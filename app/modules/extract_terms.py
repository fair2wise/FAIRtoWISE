#!/usr/bin/env python3
import os
import json
import logging
import re
import requests
import datetime
import difflib
from typing import Dict, Any, Optional
import fitz  # PyMuPDF

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class OllamaTermExtractor:
    """
    Terminology extraction agent for materials science papers using the Ollama API.

    This class processes PDFs, sends page content to the LLM via the Ollama API, and aggregates
    extracted terms with enhanced metadata including extraction date, context snippets, and fuzzy merging.
    """
    # play with repition penalty (repeated tokens)
    # length penalty (keep the output concise)
    def __init__(
        self,
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.025, # 0.1, # out of every 10 tokens, 1 is outside the normal distribution
        # if we keep temperature very low, we can compare results
        data_dir: str = "./polymer_papers_all",
        output_file: str = "./storage/terminology/extracted_terms.json",
        context_length: int = 50,  # Number of words for context snippet
    ):
        """
        Initialize the extractor.

        Parameters:
            ollama_model (str): Name of the Ollama model.
            ollama_base_url (str): Base URL for the Ollama API.
            temperature (float): Sampling temperature for the model.
            data_dir (str): Directory containing PDF files.
            output_file (str): Path to save the extracted terms.
            context_length (int): Number of words to extract for context snippet.
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature = temperature
        self.data_dir = data_dir
        self.output_file = output_file
        self.context_length = context_length

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Initialize term storage and metadata
        self.terms_dict: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {
            "extraction_date": datetime.datetime.utcnow().isoformat() + "Z",
            "processed_files": 0,
            "processed_pages": 0,
            "version": "2.0",
        }

        logger.info(f"Initialized OllamaTermExtractor with model: {ollama_model}")

        # Load existing terms if the output file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    for term in data.get("terms", []):
                        self.terms_dict[term["term"].lower()] = term
                    if "metadata" in data:
                        self.metadata.update(data["metadata"])
                logger.info(
                    f"Loaded {len(self.terms_dict)} existing terms from {output_file}"
                )
            except Exception as e:
                logger.warning(f"Could not load existing terms: {str(e)}")

    def save_terms(self) -> None:
        """Save current terms and metadata to the output file."""
        output_data = {
            "metadata": self.metadata,
            "terms": list(self.terms_dict.values()),
        }
        with open(self.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved {len(output_data['terms'])} terms to {self.output_file}")

    def extract_page_text(self, pdf_path: str, page_num: int) -> str:
        """
        Extract text from a specific page of a PDF.

        Parameters:
            pdf_path (str): Path to the PDF file.
            page_num (int): Zero-based page number.

        Returns:
            str: Extracted text.
        """
        try:
            # We can try different methods to extract text if needed
            doc = fitz.open(pdf_path)
            if 0 <= page_num < len(doc):
                return doc[page_num].get_text()
            return ""
        except Exception as e:
            logger.error(
                f"Error extracting text from page {page_num} of {pdf_path}: {str(e)}"
            )
            return ""

    def call_ollama(self, prompt: str) -> str:
        """
        Call the Ollama API with the given prompt.

        Parameters:
            prompt (str): The prompt to send to the API.

        Returns:
            str: The content of the API response.
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if response.status_code != 200:
                logger.error(f"Ollama API Error: {response.status_code}")
                return ""
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return ""

    def extract_json_from_text(self, text: str) -> Dict:
        """
        Extract and parse JSON from text using a robust regex.

        Parameters:
            text (str): The text containing JSON.

        Returns:
            Dict: Parsed JSON object with a "terms" key; empty if parsing fails.
        """
        json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        matches = list(re.finditer(json_pattern, text))
        if not matches:
            logger.warning("No JSON object found in response")
            return {"terms": []}
        matches.sort(key=lambda m: len(m.group(0)), reverse=True)
        for match in matches:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                if "terms" in data and isinstance(data["terms"], list):
                    return data
            except json.JSONDecodeError:
                continue
        logger.warning("No valid JSON with terms structure found")
        return {"terms": []}

    def normalize_term(self, term: str) -> str:
        """
        Normalize a term string for deduplication.

        Parameters:
            term (str): The term to normalize.

        Returns:
            str: A normalized (lowercase, stripped) term.
        """
        return term.strip().lower()

    def fuzzy_merge(self, new_term: Dict[str, Any]) -> Optional[str]:
        """
        Check if a similar term already exists using fuzzy matching.

        Parameters:
            new_term (Dict[str, Any]): The new term data.

        Returns:
            Optional[str]: The key of an existing similar term, or None if no match.
        """
        new_key = self.normalize_term(new_term.get("term", ""))
        for existing_key in self.terms_dict:
            similarity = difflib.SequenceMatcher(None, new_key, existing_key).ratio()
            if similarity > 0.85:
                return existing_key
        return None

    def get_context_snippet(self, text: str, term: str) -> str:
        """
        Extract a context snippet from the page text that includes the term.

        Parameters:
            text (str): The full text of the page.
            term (str): The term to locate in the text.

        Returns:
            str: A snippet (up to context_length words) from the sentence containing the term.
        """
        # Split text into sentences using a basic regex pattern.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if term.lower() in sentence.lower():
                words = sentence.split()
                if len(words) > self.context_length:
                    return " ".join(words[: self.context_length])
                else:
                    return sentence
        # Fallback: return the first context_length words of the text.
        return " ".join(text.split()[: self.context_length])

    def process_page(self, pdf_path: str, page_num: int) -> bool:
        """
        Process a single page from a PDF file.

        Parameters:
            pdf_path (str): Path to the PDF file.
            page_num (int): Zero-based page number.

        Returns:
            bool: True if new terms were added; False otherwise.
        """
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing {filename} page {page_num+1}")
        text = self.extract_page_text(pdf_path, page_num)
        if not text or len(text.split()) < 20:
            logger.info(f"Skipping page {page_num+1} - too short")
            return False

        # Look into running with Slurm for bigger models
        # Updated prompt with "relations" field.
        prompt = f"""Extract key terminology from this page of a materials science paper.

PAPER: {filename}
PAGE: {page_num+1}

CONTENT:
{text}

INSTRUCTIONS:
Identify technical terms relevant to materials, polymers, and chemical sciences on this page.

For each term, provide a JSON object with the following keys:

"term": the term name.
"definition": a brief, clear definition of the term.
"category": choose one of the following categories (inspired by the Biolink model):
chemical_substance | material | physical_property | processing_method | structural_feature | application | other
"relations": an array of relation objects. For each relation object, include:
   - "relation": a descriptive phrase that explains how the term is connected to or interacts with another term (e.g., "is a type of", "is synthesized by", "exhibits property", "is measured by", "is applied in", "is characterized by", "has structural feature", "modifies", "depends on", "interacts with").
   - "related_term": the term that is related.
Ensure that each term has at least one relation. If no relation is obvious, include a relation object with "relation": "has no identified relation" and "related_term": "".
Provide ONLY the JSON response without any additional text.

Format response as JSON:
{{
  "terms": [
    {{
      "term": "term name",
      "definition": "brief definition",
      "category": "chemical_substance|material|physical_property|processing_method|structural_feature|application|other",
      "relations": [
          {{ "relation": "descriptive relation phrase", "related_term": "related term" }}
      ]
    }}
  ]
}}
"""
        response = self.call_ollama(prompt)
        logger.debug(f"Raw API response for {filename} page {page_num+1}: {response}")
        if not response:
            return False

        terms_data = self.extract_json_from_text(response)
        if not terms_data.get("terms"):
            logger.warning(
                f"No terms found in response for {filename} page {page_num+1}"
            )
            return False

        terms_added = 0
        for term in terms_data.get("terms", []):
            term_name = term.get("term")
            if not term_name:
                continue
            normalized_key = self.normalize_term(term_name)
            # Get a context snippet specific to the term by searching the page text.
            snippet = self.get_context_snippet(text, term_name)
            existing_key = self.fuzzy_merge(term)
            if existing_key:
                # Merge with the existing term entry
                existing = self.terms_dict[existing_key]
                existing.setdefault("pages", [])
                if page_num + 1 not in existing["pages"]:
                    existing["pages"].append(page_num + 1)
                    existing["pages"].sort()
                existing.setdefault("source_papers", [])
                if filename not in existing["source_papers"]:
                    existing["source_papers"].append(filename)
                # Update definition if the new one is more detailed
                if len(term.get("definition", "")) > len(
                    existing.get("definition", "")
                ):
                    existing["definition"] = term["definition"]
                # Merge relations if provided
                if "relations" in term:
                    existing.setdefault("relations", [])
                    for new_rel in term["relations"]:
                        if new_rel not in existing["relations"]:
                            existing["relations"].append(new_rel)
                # Append the context snippet for traceability
                existing.setdefault("context_snippets", [])
                if snippet not in existing["context_snippets"]:
                    existing["context_snippets"].append(snippet)
            else:
                # Add new term entry with relations field and context snippet
                self.terms_dict[normalized_key] = {
                    "term": term_name,
                    "definition": term.get("definition", ""),
                    "category": term.get("category", "other"),
                    "relations": term.get("relations", []),
                    "pages": [page_num + 1],
                    "source_papers": [filename],
                    "context_snippets": [snippet],
                }
                terms_added += 1

        if terms_added > 0:
            self.save_terms()
            logger.info(
                f"Extracted {terms_added} new terms from {filename} page {page_num+1}"
            )
            return True
        else:
            logger.info(f"No new terms found on {filename} page {page_num+1}")
            return False

    def process_pdf(self, pdf_path: str) -> int:
        """
        Process a single PDF file.

        Parameters:
            pdf_path (str): Path to the PDF file.

        Returns:
            int: Number of pages on which new terms were added.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return 0

        try:
            doc = fitz.open(pdf_path)
            filename = os.path.basename(pdf_path)
            logger.info(f"Processing {filename} with {len(doc)} pages")
            pages_processed = 0
            for page_num in range(len(doc)):
                if self.process_page(pdf_path, page_num):
                    pages_processed += 1
            self.metadata["processed_files"] += 1
            self.metadata["processed_pages"] += len(doc)
            return pages_processed
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return 0

    def process_directory(self) -> Dict[str, Any]:
        """
        Process all PDF files in the data directory.

        Returns:
            Dict[str, Any]: A summary of the processing results.
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"Directory not found: {self.data_dir}")
            return {"status": "error", "message": "Directory not found"}

        processed_files = 0
        processed_pages = 0

        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.data_dir, filename)
                pages = self.process_pdf(pdf_path)
                if pages > 0:
                    processed_files += 1
                    processed_pages += pages

        # Update term statistics
        for term in self.terms_dict.values():
            occurrences = len(term.get("pages", []))
            term["total_occurrences"] = occurrences
            paper_count = len(term.get("source_papers", []))
            term["paper_count"] = paper_count
            if paper_count > 1 or occurrences > 5:
                term["importance"] = "high"
            elif occurrences > 2:
                term["importance"] = "medium"
            else:
                term["importance"] = "low"
            term.setdefault("metrics", {})["paper_references"] = term.get(
                "source_papers", []
            )

        self.save_terms()
        summary = {
            "status": "success",
            "processed_files": processed_files,
            "processed_pages": processed_pages,
            "unique_terms": len(self.terms_dict),
            "terms": list(self.terms_dict.values()),
            "output_file": self.output_file,
            "metadata": self.metadata,
        }
        return summary


if __name__ == "__main__":
    extractor = OllamaTermExtractor()
    result = extractor.process_directory()

    print(
        f"Processed {result['processed_files']} files, {result['processed_pages']} pages"
    )
    print(f"Extracted {result['unique_terms']} unique terms")
    print(f"Results saved to {result['output_file']}")
