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
from linkml_runtime.utils.schemaview import SchemaView

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OllamaTermExtractor")


class SchemaHelper:
    def __init__(self, schema_path="matkg_schema.yaml"):
        self.schema_path = schema_path
        self.schema_view = SchemaView(schema_path)
        self._load_schema_context()

    def _load_schema_context(self):
        """Extract schema information for LLM context"""
        # Get all classes with their descriptions
        self.classes = {}
        self.class_hierarchy = {}
        for class_name, class_def in self.schema_view.all_classes().items():
            self.classes[class_name] = {
                'name': class_name,
                'description': class_def.description or f"A {class_name} entity",
                'parent': class_def.is_a if class_def.is_a else None,
                'slots': []
            }

        # Get all slots/properties with their constraints
        self.slots = {}
        for slot_name, slot_def in self.schema_view.all_slots().items():
            self.slots[slot_name] = {
                'name': slot_name,
                'description': slot_def.description or f"Relationship: {slot_name}",
                'domain': slot_def.domain,  # What class can have this property
                'range': slot_def.range,    # What class this property points to
                'multivalued': slot_def.multivalued or False
            }

        # Associate slots with their domain classes
        for slot_name, slot_info in self.slots.items():
            if slot_info['domain']:
                domain_class = slot_info['domain']
                if domain_class in self.classes:
                    self.classes[domain_class]['slots'].append(slot_name)

        logger.info(f"Loaded schema with {len(self.classes)} classes and {len(self.slots)} slots")

    def get_schema_context_for_llm(self) -> str:
        """Generate schema context string for LLM prompt"""
        context = "=== KNOWLEDGE SCHEMA ===\n\n"
        
        # Add class definitions
        context += "ENTITY TYPES (use exactly these names):\n"
        for class_name, class_info in sorted(self.classes.items()):
            context += f"- {class_name}: {class_info['description']}\n"
            if class_info['parent']:
                context += f"  (inherits from: {class_info['parent']})\n"
        
        context += "\nVALID RELATIONSHIPS (use exactly these names):\n"
        for slot_name, slot_info in sorted(self.slots.items()):
            domain = slot_info['domain'] or "Any"
            range_class = slot_info['range'] or "Any"
            context += f"- {slot_name}: {slot_info['description']}\n"
            context += f"  Usage: {domain} --{slot_name}--> {range_class}\n"
            if slot_info['multivalued']:
                context += f"  (can have multiple values)\n"

        context += "\nIMPORTANT: Only use the exact entity type names and relationship names listed above!\n\n"
        return context

    def validate_and_fix_term(self, term_data: dict) -> dict:
        """Validate and fix extracted term against schema"""
        # Validate category/class
        category = term_data.get('category', '')
        if category not in self.classes:
            # Try to find closest match
            best_match = self._find_closest_class(category)
            if best_match:
                logger.warning(f"Fixed category '{category}' -> '{best_match}'")
                term_data['category'] = best_match
            else:
                logger.warning(f"Unknown category '{category}', keeping as-is")

        # Validate relations
        validated_relations = []
        for rel in term_data.get('relations', []):
            pred = rel.get('relation', '')
            obj_term = rel.get('related_term', '')
            
            if pred in self.slots:
                validated_relations.append(rel)
            else:
                # Try to find closest slot match
                best_slot = self._find_closest_slot(pred)
                if best_slot:
                    logger.warning(f"Fixed relation '{pred}' -> '{best_slot}'")
                    rel['relation'] = best_slot
                    validated_relations.append(rel)
                else:
                    logger.warning(f"Unknown relation '{pred}', marking as unvalidated")
                    rel['relation'] = f"unvalidated:{pred}"
                    validated_relations.append(rel)

        term_data['relations'] = validated_relations
        return term_data

    def _find_closest_class(self, target: str) -> str:
        """Find closest matching class name using fuzzy matching"""
        import difflib
        target_lower = target.lower()
        
        # First try exact match (case insensitive)
        for class_name in self.classes:
            if class_name.lower() == target_lower:
                return class_name
        
        # Then try fuzzy matching
        matches = difflib.get_close_matches(target_lower, 
                                          [c.lower() for c in self.classes.keys()], 
                                          n=1, cutoff=0.6)
        if matches:
            # Find the original case version
            for class_name in self.classes:
                if class_name.lower() == matches[0]:
                    return class_name
        return None

    def _find_closest_slot(self, target: str) -> str:
        """Find closest matching slot/relation name"""
        import difflib
        target_lower = target.lower()
        
        # First try exact match
        for slot_name in self.slots:
            if slot_name.lower() == target_lower:
                return slot_name
        
        # Then try fuzzy matching
        matches = difflib.get_close_matches(target_lower,
                                          [s.lower() for s in self.slots.keys()],
                                          n=1, cutoff=0.6)
        if matches:
            for slot_name in self.slots:
                if slot_name.lower() == matches[0]:
                    return slot_name
        return None

    def check_relation_validity(self, subj_cls: str, pred: str, obj_cls: str) -> bool:
        """Check if a relationship is valid according to schema"""
        if pred not in self.slots:
            return False
        
        slot_info = self.slots[pred]
        
        # Check domain constraint (what class can have this property)
        if slot_info['domain'] and subj_cls != slot_info['domain']:
            # Check if subject class inherits from domain
            if not self._is_subclass_of(subj_cls, slot_info['domain']):
                return False
        
        # Check range constraint (what class this property can point to)  
        if slot_info['range'] and obj_cls != slot_info['range']:
            if not self._is_subclass_of(obj_cls, slot_info['range']):
                return False
                
        return True

    def _is_subclass_of(self, child_class: str, parent_class: str) -> bool:
        """Check if child_class is a subclass of parent_class"""
        if child_class == parent_class:
            return True
        
        if child_class not in self.classes:
            return False
            
        parent = self.classes[child_class].get('parent')
        if parent == parent_class:
            return True
        elif parent:
            return self._is_subclass_of(parent, parent_class)
        
        return False


class OllamaTermExtractor:
    def __init__(
        self,
        ollama_model="mistral-small3.1:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.0,
        data_dir="./polymer_papers",
        output_file="./storage/terminology/extracted_terms.json",
        context_length=50,
        schema_path="matkg_schema.yaml",
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature = temperature
        self.data_dir = data_dir
        self.output_file = output_file
        self.context_length = context_length
        self.schema_helper = SchemaHelper(schema_path)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.terms_dict: Dict[str, Dict[str, Any]] = {}
        self.metadata = {
            "extraction_date": datetime.datetime.utcnow().isoformat() + "Z",
            "processed_files": 0,
            "processed_pages": 0,
            "version": "2.1",
        }

        if os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    data = json.load(f)
                    for term in data.get("terms", []):
                        self.terms_dict[term["term"].lower()] = term
                    self.metadata.update(data.get("metadata", {}))
                logger.info(f"Loaded {len(self.terms_dict)} existing terms")
            except Exception as e:
                logger.warning(f"Could not load existing terms: {e}")

    def save_terms(self):
        with open(self.output_file, "w") as f:
            json.dump({
                "metadata": self.metadata,
                "terms": list(self.terms_dict.values())
            }, f, indent=2)
        logger.info(f"Saved {len(self.terms_dict)} terms to {self.output_file}")

    # we should have more methods for extracting/interpreting figures, tables
    def extract_page_text(self, pdf_path, page_num):
        try:
            doc = fitz.open(pdf_path)
            if 0 <= page_num < len(doc):
                return doc[page_num].get_text()
        except Exception as e:
            logger.error(f"Page extraction error: {e}")
        return ""

    def call_ollama(self, prompt, retries=2):
        for attempt in range(retries + 1):
            try:
                timeout = 120 if attempt == 0 else 180  # Increase timeout on retry
                # add some options for additional fine-tuning (repetition penalties)
                # struct common_params_sampling {
#     uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler
#     int32_t n_prev             = 64;    // number of previous tokens to remember
#     int32_t n_probs            = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
#     int32_t min_keep           = 0;     // 0 = disabled, otherwise samplers should return at least min_keep tokens
#     int32_t top_k              = 40;    // <= 0 to use vocab size
#     float   top_p              = 0.95f; // 1.0 = disabled
#     float   min_p              = 0.05f; // 0.0 = disabled
#     float   xtc_probability    = 0.00f; // 0.0 = disabled
#     float   xtc_threshold      = 0.10f; // > 0.5 disables XTC
#     float   typ_p              = 1.00f; // typical_p, 1.0 = disabled
#     float   temp               = 0.80f; // <= 0.0 to sample greedily, 0.0 to not output probabilities
# float   dynatemp_range     = 0.00f; // 0.0 = disabled
#     float   dynatemp_exponent  = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler
#     int32_t penalty_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
#     float   penalty_repeat     = 1.00f; // 1.0 = disabled
#     float   penalty_freq       = 0.00f; // 0.0 = disabled
#     float   penalty_present    = 0.00f; // 0.0 = disabled
#     float   dry_multiplier     = 0.0f;  // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
#     float   dry_base           = 1.75f; // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
#     int32_t dry_allowed_length = 2;     // tokens extending repetitions beyond this receive penalty
#     int32_t dry_penalty_last_n = -1;    // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)

                r = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "")
            except requests.exceptions.Timeout as e:
                logger.warning(f"Ollama timeout (attempt {attempt + 1}/{retries + 1}): {e}")
                if attempt < retries:
                    logger.info(f"Retrying with longer timeout...")
                    continue
            except Exception as e:
                logger.error(f"Ollama error (attempt {attempt + 1}): {e}")
                if attempt < retries:
                    continue
        return ""

    def extract_json_from_text(self, text):
        json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        for m in sorted(re.finditer(json_pattern, text), key=lambda x: -len(x.group(0))):
            try:
                obj = json.loads(m.group(0))
                if "terms" in obj:
                    return obj
            except Exception:
                continue
        return {"terms": []}

    def normalize_term(self, term):
        return term.strip().lower()

    # is this a good merging method? can we make this more robust?
    def fuzzy_merge(self, term):
        key = self.normalize_term(term)
        for existing_key in self.terms_dict:
            if difflib.SequenceMatcher(None, key, existing_key).ratio() > 0.85:
                return existing_key
        return None

    # it looks like this is just looking to see if the term is in the context snippet
    # this should also check if the relation it found (i.e. chemical formula) is also in the snippet
    def get_context_snippet(self, text, term, filename, page_num):
        """Get context snippet with source attribution"""
        for s in re.split(r"(?<=[.!?])\s+", text):
            if term.lower() in s.lower():
                snippet_text = " ".join(s.split()[: self.context_length])
                return {
                    "text": snippet_text,
                    "source_paper": filename,
                    "page": page_num + 1
                }
        
        # Fallback if term not found in sentences
        snippet_text = " ".join(text.split()[: self.context_length])
        return {
            "text": snippet_text,
            "source_paper": filename,
            "page": page_num + 1
        }

    def process_page(self, pdf_path, page_num):
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing {filename} page {page_num+1}")
        text = self.extract_page_text(pdf_path, page_num)
        if not text or len(text.split()) < 20:
            logger.warning(f"Skipping page {page_num+1} - insufficient text")
            return False

        # Get schema context for the LLM
        schema_context = self.schema_helper.get_schema_context_for_llm()

        # Maybe we need example snippets in the json
        prompt = f"""{schema_context}

=== EXTRACTION TASK ===

Extract key terminology from this page of a materials science paper using the schema above.

PAPER: {filename}
PAGE: {page_num+1}

CONTENT:
{text}

INSTRUCTIONS:
1. Identify technical terms relevant to materials, polymers, and chemical sciences
2. Use ONLY the entity types listed in the schema above
3. Use ONLY the predicate names listed in the schema above
4. If you find something new (either entity or predicate), add it with a special label (label: "unverified").
5. Ensure that the information is in the source paper content; otherwise, don't hallucinate.
6. Make sure you capture accurate chemical formulas that are in the text. There should only be one formula per chemical.
7. For each term, return JSON in this exact format:

{{
  "terms": [
    {{
      "term": "exact term from text",
      "definition": "brief technical definition", 
      "category": "exact_entity_type_from_schema",
      "formula": "chemical formula if it is a chemical, otherwise null"
      "predicates": [
        {{
          "predicate": "exact_relationship_name_from_schema",
          "related_term": "related term name"
        }}
      ]
    }}
  ]
}}

CRITICAL: Only use entity types and relationship names that appear exactly in the schema above!
"""

        response = self.call_ollama(prompt)
        if not response:
            logger.warning(f"No response from Ollama for page {page_num+1}, continuing...")
            return False

        data = self.extract_json_from_text(response)
        if not data.get("terms"):
            logger.warning(f"No terms extracted from page {page_num+1}")
            return False
            
        added = 0
        for term in data.get("terms", []):
            name = term.get("term", "").strip()
            if not name:
                continue

            # Validate and fix term against schema
            term = self.schema_helper.validate_and_fix_term(term)

            # this just sets to lowercase. there are other tools we could implement
            # expand to normalize lexical variant (spacy)
            # keep a record of which things get normalized, list of synonyms
            key = self.normalize_term(name)
            category = term.get("category", "Thing")

            snippet = self.get_context_snippet(text, name, filename, page_num)
            existing = self.fuzzy_merge(name)
            if existing:
                entry = self.terms_dict[existing]
                entry.setdefault("pages", []).append(page_num + 1)
                entry.setdefault("source_papers", []).append(filename)
                entry.setdefault("context_snippets", []).append(snippet)
                
                # Merge relations, avoiding duplicates
                # this should be better for cases where there should only be one option (such as formula)
                # this could be a separate step, using multiple methods
                # agentic AI approach
                existing_relations = entry.setdefault("relations", [])
                for new_rel in term.get("relations", []):
                    if new_rel not in existing_relations:
                        existing_relations.append(new_rel)
                
                # Update definition if new one is longer/better
                if len(term.get("definition", "")) > len(entry.get("definition", "")):
                    entry["definition"] = term["definition"]
            else:
                self.terms_dict[key] = {
                    "term": name,
                    "definition": term.get("definition", ""),
                    "category": category,
                    "relations": term.get("relations", []),
                    "pages": [page_num + 1],
                    "source_papers": [filename],
                    "context_snippets": [snippet],
                }
                added += 1

        if added > 0:
            logger.info(f"Added {added} new terms from page {page_num+1}")
        
        # Save after processing each page to prevent data loss
        self.save_terms()
        
        return added > 0 or len(data.get("terms", [])) > 0  # Return True if we processed any terms

    def process_pdf(self, path):
        added = 0
        pages_processed = 0
        try:
            doc = fitz.open(path)
            total_pages = len(doc)
            for i in range(total_pages):
                pages_processed += 1
                if self.process_page(path, i):
                    added += 1
                # Note: save_terms() is now called after each page in process_page()
                    
            self.metadata["processed_files"] += 1
            self.metadata["processed_pages"] += total_pages
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
        return added

    def process_directory(self):
        if not os.path.exists(self.data_dir):
            logger.error(f"Directory not found: {self.data_dir}")
            return {
                "status": "error", 
                "message": "Directory not found",
                "processed_files": 0,
                "processed_pages": 0,
                "unique_terms": 0,
                "output_file": self.output_file
            }

        processed_files = 0
        processed_pages = 0
        pdf_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            
        for i, fname in enumerate(pdf_files):
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: {fname}")
            path = os.path.join(self.data_dir, fname)
            pages = self.process_pdf(path)
            if pages:
                processed_files += 1
                processed_pages += pages

        # Final processing and importance calculation
        for t in self.terms_dict.values():
            t["total_occurrences"] = len(t.get("pages", []))
            t["paper_count"] = len(set(t.get("source_papers", [])))  # Use set to avoid duplicates
            occ = t["total_occurrences"]
            papers = t["paper_count"]
            t["importance"] = "high" if papers > 1 or occ > 5 else "medium" if occ > 2 else "low"

        # Final save
        self.save_terms()
        
        logger.info(f"Processing complete! Found {len(self.terms_dict)} unique terms")
        return {
            "status": "success",
            "processed_files": processed_files,
            "processed_pages": processed_pages,
            "unique_terms": len(self.terms_dict),
            "output_file": self.output_file,
        }


if __name__ == "__main__":
    extractor = OllamaTermExtractor()
    result = extractor.process_directory()
    
    if result["status"] == "error":
        print(f"✗ Error: {result['message']}")
        print(f"  Expected directory: {extractor.data_dir}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
    else:
        print(
            f"✓ Processed {result['processed_files']} files, "
            f"{result['processed_pages']} pages, "
            f"{result['unique_terms']} unique terms → {result['output_file']}"
        )