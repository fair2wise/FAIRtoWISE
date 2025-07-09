#!/usr/bin/env python3
import os
import json
import logging
import re
import requests
import datetime
import threading
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from typing import Dict, Any, Optional, List, Union, Tuple

import fitz  # this is exactly the same as PyMuPDF

from linkml_runtime.utils.schemaview import SchemaView
from rapidfuzz import process, fuzz  # still used by SchemaHelper for class/slot matching

from agents.chebi import ChebiOboLookup
from agents.chem_checker import ChemicalFormulaValidator
from agents.properties import PhysicalPropertyExtractor, PropertyNormalizer
# extract tables and images in an agentic way
# GEMMA 3 27b
# Llama 3.2 Vision

# ----------------------------------------
# Logging Configuration
# ----------------------------------------
class _AnsiColorFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",    # cyan
        "INFO": "\033[32m",     # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
        "CRITICAL": "\033[41m", # red background
    }
    _RESET = "\033[0m"

    def format(self, record):
        level = record.levelname
        color = self._COLORS.get(level)
        if color:
            record.levelname = "{}{}{}".format(color, level, self._RESET)
            record.msg = "{}{}{}".format(color, record.getMessage(), self._RESET)
            record.args = ()
        return super(_AnsiColorFormatter, self).format(record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    _AnsiColorFormatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                       datefmt="%Y-%m-%d %H:%M:%S")
)
logger = logging.getLogger("OllamaTermExtractor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(handler)

# ----------------------------------------
# Utility: Retriable Decorator
# ----------------------------------------
def retry_on_exception(
    exceptions: Union[Tuple[type, ...], type],
    retries: int = 2,
    delay_seconds: float = 1.0,
) -> Any:
    """
    Decorator that retries a function up to `retries` times if one of `exceptions` is raised.
    Sleeps `delay_seconds * 2^attempt` between attempts.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(
                        f"Retryable error in {fn.__name__} (attempt {attempt + 1}/{retries + 1}): {e}"
                    )
                    if attempt < retries:
                        sleep_time = delay_seconds * (2 ** attempt)
                        logger.debug(f"Sleeping {sleep_time:.1f}s before retry")
                        threading.Event().wait(sleep_time)
                        continue
                    break
            raise last_exc
        return wrapper
    return decorator

# ----------------------------------------
# SchemaHelper (refined)
# ----------------------------------------
class SchemaHelper:
    """
    Loads a LinkML schema and provides:
      - RapidFuzz indexes for class‐names and slot‐names
      - Exact‐match + fuzzy suggestions
      - Domain/range validation
      - Relation filtering (drop 'description'/'category')
    """
    def __init__(self, schema_path: str = "matkg_schema.yaml", fuzzy_cutoff: int = 80):
        """
        Args:
            schema_path: Path to a LinkML YAML schema.
            fuzzy_cutoff: RapidFuzz score cutoff (0–100) for fuzzy matching.
        """
        self.schema_path = schema_path
        self.fuzzy_cutoff = fuzzy_cutoff
        self.schema_view = SchemaView(schema_path)
        self._load_classes_and_slots()
        self._build_fuzzy_indexes()

    def _load_classes_and_slots(self) -> None:
        """Load classes and slots from LinkML schema."""
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.class_parents: Dict[str, Optional[str]] = {}
        for name, cls in self.schema_view.all_classes().items():
            desc = cls.description or f"A {name} entity"
            parent = cls.is_a or None
            self.classes[name] = {"description": desc, "slots": []}
            self.class_parents[name] = parent

        self.slots: Dict[str, Dict[str, Any]] = {}
        for slot_name, slot_def in self.schema_view.all_slots().items():
            desc = slot_def.description or f"Relationship: {slot_name}"
            domain = slot_def.domain or None
            rng = slot_def.range or None
            mv = bool(slot_def.multivalued)
            self.slots[slot_name] = {
                "description": desc,
                "domain": domain,
                "range": rng,
                "multivalued": mv,
            }
            if domain and domain in self.classes:
                self.classes[domain]["slots"].append(slot_name)
        logger.info(f"Loaded schema: {len(self.classes)} classes, {len(self.slots)} slots")

    def _build_fuzzy_indexes(self) -> None:
        """Build lowercase→canonical maps for classes and slots."""
        self._class_names_lower = [c.lower() for c in self.classes]
        self._class_map_lower = {c.lower(): c for c in self.classes}
        self._slot_names_lower = [s.lower() for s in self.slots]
        self._slot_map_lower = {s.lower(): s for s in self.slots}
        logger.debug("Built fuzzy indexes for classes and slots")

    def get_schema_context_for_llm(self) -> str:
        """Generate schema context string for LLM."""
        lines: List[str] = ["=== KNOWLEDGE SCHEMA ===\n", "ENTITY TYPES (use exactly these names):"]
        for cls in sorted(self.classes):
            desc = self.classes[cls]["description"]
            parent = self.class_parents[cls]
            if parent:
                lines.append(f"- {cls}: {desc}  (inherits from: {parent})")
            else:
                lines.append(f"- {cls}: {desc}")
        lines.append("\nVALID RELATIONSHIPS (use exactly these names):")
        for slot in sorted(self.slots):
            info = self.slots[slot]
            dom = info["domain"] or "Any"
            rng = info["range"] or "Any"
            mv = "(multivalued)" if info["multivalued"] else ""
            lines.append(f"- {slot}: {info['description']}  Usage: {dom} --{slot}--> {rng} {mv}")
        lines.append("\nIMPORTANT: Do NOT use relations named 'description' or 'category'.")
        return "\n".join(lines)

    def _find_closest_class(self, target: str) -> Optional[str]:
        """Return exact or fuzzy-matched class, or None."""
        if not target:
            return None
        tl = target.strip().lower()
        if tl in self._class_map_lower:
            return self._class_map_lower[tl]
        match = process.extractOne(tl, self._class_names_lower, scorer=fuzz.QRatio, score_cutoff=self.fuzzy_cutoff)
        if match:
            found_lower, score, _ = match
            return self._class_map_lower.get(found_lower)
        return None

    def _find_closest_slot(self, target: str) -> Optional[str]:
        """Return exact or fuzzy-matched slot, or None."""
        if not target:
            return None
        tl = target.strip().lower()
        if tl in self._slot_map_lower:
            return self._slot_map_lower[tl]
        match = process.extractOne(tl, self._slot_names_lower, scorer=fuzz.QRatio, score_cutoff=self.fuzzy_cutoff)
        if match:
            found_lower, score, _ = match
            return self._slot_map_lower.get(found_lower)
        return None

    def validate_and_fix_term(self, term_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure 'category' is valid (fuzzy-fix if needed).
        Filter out any relations named 'description' or 'category'.
        For each remaining relation:
          - If slot exists → keep verified=True
          - Else if fuzzy-match → correct name, verified=True
          - Else keep original name, verified=False
        """
        # 1) Category
        cat = term_data.get("category", "").strip()
        if cat not in self.classes:
            fixed = self._find_closest_class(cat)
            if fixed:
                logger.warning(f"Fixed category '{cat}' → '{fixed}'")
                term_data["category"] = fixed
            else:
                logger.warning(f"Unknown category '{cat}' (left as-is)")

        # 2) Process relations
        cleaned_rels: List[Dict[str, Union[str, bool]]] = []
        for rel in term_data.get("relations", []):
            pred = rel.get("relation", "").strip()
            obj = rel.get("related_term", "").strip()
            if pred.lower() in ("description", "category"):
                logger.debug(f"Dropping relation '{pred}' as prohibited")
                continue

            if pred in self.slots:
                cleaned_rels.append({"relation": pred, "related_term": obj, "verified": True})
            else:
                fixed_slot = self._find_closest_slot(pred)
                if fixed_slot:
                    logger.warning(f"Fixed relation '{pred}' → '{fixed_slot}'")
                    cleaned_rels.append({"relation": fixed_slot, "related_term": obj, "verified": True})
                else:
                    logger.warning(f"Unknown relation '{pred}' → marking unverified")
                    cleaned_rels.append({"relation": pred, "related_term": obj, "verified": False})

        term_data["relations"] = cleaned_rels
        return term_data

    def _is_subclass_of(self, child: str, parent: str) -> bool:
        """Recursive check if child inherits from parent."""
        if child == parent:
            return True
        if child not in self.classes:
            return False
        p = self.class_parents.get(child)
        if not p:
            return False
        return self._is_subclass_of(p, parent)

    def check_relation_validity(self, subj_cls: str, pred: str, obj_cls: str) -> bool:
        """
        Verify domain/range of (subj_cls, pred, obj_cls). Return False if invalid.
        """
        if pred not in self.slots:
            return False
        slot = self.slots[pred]
        dom = slot["domain"]
        rng = slot["range"]
        if dom and not self._is_subclass_of(subj_cls, dom):
            return False
        if rng and not self._is_subclass_of(obj_cls, rng):
            return False
        return True

# ----------------------------------------
# OllamaTermExtractor (enhanced, frequent saves,
# with property extractor/normalizer integration + ChEBI enrichment)
# ----------------------------------------
class OllamaTermExtractor:
    """
    Extracts terms from PDFs using a local Ollama LLM + LinkML schema validation.
    - Drops 'description'/'category' relations
    - Records unverified relations with {"verified": false}
    - Ensures every ChemicalEntity gets a formula_validation entry (or 'missing')
    - Enriches chemical terms via ChEBI (formula, mass, charge, InChI, InChIKey, SMILES, etc.)
    - Incorporates PhysicalPropertyExtractor and PropertyNormalizer to attach properties on the fly
    - Saves JSON after every page that yields new terms or properties (thread-safe)
    """
    def __init__(
        self,
        ollama_model: str = "gemma3:27b",  # or "mistral-small3.1:latest"
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        data_dir: str = "./polymer_papers",
        output_file: str = "./storage/terminology/extracted_terms.json",
        context_length: int = 50,
        schema_path: str = "matkg_schema.yaml",
        max_workers: int = 50,
    ):
        """
        Args:
            ollama_model: Ollama model specifier.
            ollama_base_url: Base URL of Ollama API.
            temperature: LLM sampling temperature.
            data_dir: Directory with PDFs.
            output_file: Path to write JSON output.
            context_length: Tokens of context snippet.
            schema_path: Path to LinkML schema.
            max_workers: Parallel workers for pages.
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature = temperature
        self.data_dir = data_dir
        self.output_file = output_file
        self.context_length = context_length
        self.max_workers = max_workers

        mp_api_key = os.environ.get("MP_API_KEY", "")
        if not mp_api_key:
            logger.warning("MP_API_KEY not set; formula validation may be incomplete.")
            mp_api_key = "JziDvAj2FWxzonCe2hketK1yz4bKHRlA"
        self.formula_checker = ChemicalFormulaValidator(api_key=mp_api_key)

        self.schema_helper = SchemaHelper(schema_path=schema_path)
        self.terms_dict: Dict[str, Dict[str, Any]] = {}
        self._bk_terms: Dict[str, str] = {}  # display_text → key

        # Initialize property extractor + normalizer
        self.prop_extractor = PhysicalPropertyExtractor()
        self.prop_normalizer = PropertyNormalizer()

        # Attempt to load ChEBI ontology
        try:
            self.chebi_lookup = ChebiOboLookup("storage/ontologies/chebi.obo")
        except Exception as e:
            logger.error(f"Failed to load ChEBI ontology: {e}")
            self.chebi_lookup = None

        # Metadata
        self.metadata: Dict[str, Any] = {
            "extraction_date": datetime.datetime.utcnow().isoformat() + "Z",
            "processed_files": 0,
            "processed_pages_total": 0,
            "processed_pages_with_terms": 0,
            "version": "2.1",
        }

        self._save_lock = threading.Lock()
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as fh:
                    prev = json.load(fh)
                    for term in prev.get("terms", []):
                        key = term["term"].strip().lower()
                        self.terms_dict[key] = term
                        self._bk_terms[term["term"]] = key
                    loaded_meta = prev.get("metadata", {})
                    self.metadata.update(loaded_meta)
                logger.info(f"Loaded {len(self.terms_dict)} existing terms from {self.output_file}")
            except Exception as e:
                logger.warning(f"Could not load previous terms: {e}")

    @retry_on_exception((requests.exceptions.RequestException,), retries=2, delay_seconds=2.0)
    def call_ollama(self, prompt: str, timeout: int = 120) -> str:
        """Call Ollama chat endpoint with retries."""
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "") or ""

    @retry_on_exception((Exception,), retries=2, delay_seconds=1.0)
    def extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract largest JSON object containing "terms". Return {"terms": []} if none.
        """
        pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        matches = list(re.finditer(pattern, text))
        matches.sort(key=lambda m: -len(m.group(0)))
        for m in matches:
            snippet = m.group(0)
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict) and "terms" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
        return {"terms": []}

    def _looks_like_formula(self, s: str) -> bool:
        """Heuristic: uppercase‐lowercase blocks + digits → formula."""
        return bool(re.search(r"[A-Z][a-z]?[\d]", s or ""))

    def _postprocess_term(self, term: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Ensure every ChemicalEntity has 'formula_validation'.
        If formula string exists and looks like a formula → validate/repair.
        Otherwise set formula=None and status="missing".
        """
        formula_str = term.get("formula")
        if not formula_str or not self._looks_like_formula(formula_str):
            term["formula"] = None
            term["formula_validation"] = {"status": "missing"}
            return term

        try:
            validation = self.formula_checker.validate(formula_str)
        except Exception as e:
            logger.warning(f"Error validating formula '{formula_str}': {e}")
            validation = {"status": "error", "details": {"error": str(e)}}

        if validation.get("status") == "invalid":
            repair_prompt = (
                f"The extracted string '{formula_str}' is not a valid chemical formula.\n"
                "Based on the context below, guess the correct formula and return ONLY the formula string.\n\n"
                f"CONTEXT:\n{context}"
            )
            try:
                candidate = self.call_ollama(repair_prompt, timeout=60).strip().split()[0]
            except Exception as e:
                logger.warning(f"LLM failed to fix invalid formula '{formula_str}': {e}")
                candidate = ""
            if candidate and candidate != formula_str:
                try:
                    newval = self.formula_checker.validate(candidate)
                    if newval.get("status") != "invalid":
                        term["formula"] = candidate
                        validation = newval
                        validation["status"] = "corrected"
                    else:
                        validation["status"] = "invalid"
                except Exception as e:
                    logger.warning(f"Error re‐validating candidate '{candidate}': {e}")
        term["formula_validation"] = validation
        return term

    def normalize_term(self, term: str) -> str:
        """Lowercase + strip whitespace."""
        return term.strip().lower()

    def fuzzy_merge(self, term: str) -> Optional[str]:
        """
        Ask the LLM which existing term (if any) matches semantically.
        Return that existing key, or None.
        """
        if not self._bk_terms:
            return None

        existing_display = list(self._bk_terms.keys())
        bullets = "\n".join(f"- {disp}" for disp in existing_display)

        prompt = f"""
We have just extracted a new term:    "{term}"
Below is the list of all already‐registered terms (one per line; the first time we saw each term):
{bullets}

You must decide whether "{term}" refers to exactly the same concept as one of these, or if it is a distinct new concept.  Follow these rules:

  1. **Ignore only trivial punctuation (spaces, hyphens, slashes, brackets, parentheses, capitalization)** 
     when comparing.  For example, "GIWAXS" and "GI-WAXS" are the *same* technique and should be merged
     (choose the variant already in the list).  Likewise, "XRD" and "X-RD" (if it appeared) are identical.
     Anything beyond punctuation differences (letters, numbers, or added qualifiers) is not trivial.

  2. **Do NOT merge distinct instrument or method acronyms**.  Even if two acronyms share letters, if they are 
     known to be different techniques or materials, keep them separate.  Examples you must treat as always distinct:
       - "SEM" (scanning electron microscopy) vs. "TEM" (transmission electron microscopy)
       - "AFM" (atomic force microscopy)
       - "XPS" vs. "UPS"
       - "MoTe2" vs. "WTe2" (different compounds)
       - "Al2O3[0001]" (specific surface) vs. "Al2O3" (generic material)
     In other words, if two strings differ by more than punctuation—by letters, numbers, or explicit qualifiers—they should not be merged.

  3. **Do NOT merge general vs. specific variants**.  
     If one term is a broader concept (e.g. "band structure") and another is a specialized version (e.g. "Dirac-like band structure"), treat them as distinct.  
     Similarly, if a term includes an added qualifier or context (e.g. surface orientation "[0001]" vs. generic material), do not merge into a more general term.

  4. **If the newly extracted term is an exact punctuation‐agnostic match** to one of the existing 
     terms—i.e., removing or changing only punctuation/brackets/spaces/case makes them identical—then respond 
     with exactly that already‐registered term (preserve its original casing/spelling).  
     Otherwise, respond `"None"`.

  5. **DO merge terms if one is the acronym for the other term**, vice versa, or one term includes the acronym and the other doesn't
      For example, "angle-resolved photoelectric spectroscopy" and "ARPES" should merge to become "angle-resolved photoemission spectroscopy (ARPES)"
      Another example: "resonant soft xray scattering" or "R-SoXS" should merge to become "Resonant soft xray scattering (RSoXS)"

  6. **Your response must be exactly one line**: either the exact existing term (matching punctuation
     and case as it appears above) or the single word `None`. Don’t output anything else—no quotes,
     no extra commentary.

Here are additional examples to illustrate:

  • If the new term is `"GI-WAXS"` and the list already contains `"GIWAXS"`, respond exactly `"GIWAXS"`. 
  • If the new term is `"RSoXS"` and the list already contains `"R-SoXS"`, respond exactly `"RSoXS" as the correct term`.  
  • If the new term is `"SEM"` and the list contains `"SEM"`, respond `"SEM"`, but if the list contains 
    only `"TEM"`, respond `"None"` (distinct acronyms).  
  • If the new term is `"MoTe2"` and the list has `"WTe2"`, respond `"None"` (different compound).  
  • If the new term is `"Band-structure"` and the list has `"Dirac-like band structure"`, respond `"None"`  
    (general vs. specific).  
  • If the new term is `"Al2O3[0001]"` and the list has `"Al2O3"`, respond `"None"` (surface‐specific vs. generic).  
  • If the new term is `"photoemission"` and the list has `"angle-resolved photoemission spectroscopy (ARPES)"`,  
    respond `"None"` (general process vs. specific technique).  
  • If the new term is `"X-RD"` and the list has `"XRD"`, respond `"XRD"` (consistent acronym once punctuation is removed).
  • "organic solar cells" and "OSCs" should merge to become "Organic solar cells (OSCs)"

Now, having read the rules, please answer: which of the above existing terms is exactly the same concept
as "{term}"?  If none match, respond with `None`.
"""
        try:
            llm_response = self.call_ollama(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM merge check failed for '{term}': {e}")
            return None

        if llm_response in self._bk_terms:
            logger.debug(f"LLM‐merge '{term}' → '{llm_response}'")
            return self._bk_terms[llm_response]

        return None

    def _register_new_term(self, display_text: str) -> str:
        """
        Register new term → key, return lowercase key.
        """
        key = self.normalize_term(display_text)
        self._bk_terms[display_text] = key
        logger.debug(f"Registered new term '{display_text}' → '{key}'")
        return key

    def get_context_snippet(
        self, full_text: str, term: str, filename: str, page_num: int
    ) -> Dict[str, Union[str, int]]:
        """
        Extract ~context_length tokens around `term`. Return snippet + source/page.
        """
        sentences = re.split(r"(?<=[\.\!\?])\s+", full_text)
        low = term.lower()
        for sent in sentences:
            if low in sent.lower():
                tokens = sent.split()
                snippet = " ".join(tokens[: self.context_length])
                return {"text": snippet, "source_paper": filename, "page": page_num + 1}
        snippet = " ".join(full_text.split()[: self.context_length])
        return {"text": snippet, "source_paper": filename, "page": page_num + 1}

    def _prepare_prompt(self, schema_ctx: str, filename: str, page_num: int, text: str) -> str:
        """
        Build prompt with:
          - Schema context (no 'description'/'category' relations)
          - Few‐shot example
          - Truncated page text (~last 8000 chars)
          - JSON template
        """
        max_len = 8000
        page_text = text[-max_len:] if len(text) > max_len else text

        few_shot = r"""
### EXAMPLE
Input:
CONTENT:
"Poly(3-hexylthiophene) (P3HT) is a conjugated polymer used in organic photovoltaics."

Output:
{
  "terms": [
    {
      "term": "Poly(3-hexylthiophene) (P3HT)",
      "definition": "A conjugated polymer used in organic photovoltaics.",
      "category": "Polymer",
      "formula": "C10H14S", 
      "relations": [
        {
          "relation": "has_application",
          "related_term": "organic photovoltaics",
          "verified": true
        }
      ]
    }
  ]
}
### END-EXAMPLE
"""

        template = r"""
=== EXTRACTION TASK ===
schema_context:
{schema_ctx}

PAPER: {filename}
PAGE: {pnum}

CONTENT:
{text}

INSTRUCTIONS:
1. Extract key materials‐science terms + their relations using ONLY schema slots.]
2. Do NOT output relations named 'description' or 'category'.
3. Output JSON exactly in this structure:

{{
  "terms": [
    {{
      "term": "exact term from text",
      "definition": "brief but rich technical definition", 
      "category": "exact_entity_type_from_schema",
      "formula": "valid chemical formula or null", 
      "relations": [
        {{
          "relation": "exact_predicate_name_from_schema",
          "related_term": "related term name",
          "verified": true
        }}
      ]
    }}
  ]
}}
"""
        return (
            f"{template.format(schema_ctx=schema_ctx, filename=filename, pnum=page_num+1, text=page_text)}\n"
            f"{few_shot}"
        )

    def _save_terms_threadsafe(self) -> None:
        """Acquire lock and write JSON to disk."""
        with self._save_lock:
            try:
                # Ensure each term has "properties" key (even if empty)
                terms_out = []
                for t in self.terms_dict.values():
                    if "properties" not in t:
                        t["properties"] = []
                    terms_out.append(t)
                out = {"metadata": self.metadata, "terms": terms_out}
                with open(self.output_file, "w") as fh:
                    json.dump(out, fh, indent=2)
                logger.debug(f"Saved {len(self.terms_dict)} terms to {self.output_file}")
            except Exception as e:
                logger.error(f"Failed to save terms: {e}")

    @retry_on_exception((Exception,), retries=1, delay_seconds=1.0)
    def process_page(self, doc: fitz.Document, pdf_path: str, page_num: int) -> bool:
        """
        Process one page:
        - Extract text
        - Call LLM for JSON
        - Validate/fix each term
        - Enrich chemical terms with ChEBI data (formula, mass, charge, InChI, InChIKey, SMILES, etc.)
        - Drop prohibited relations
        - Repair/fill formula & formula_validation
        - Merge duplicates via LLM
        - Extract & normalize properties for all known materials on this page
        - Attach properties to term entries
        - If new/updated → save immediately (thread-safe)
        Returns True if new/updated terms or properties were added.
        """
        filename = os.path.basename(pdf_path)
        logger.debug(f">> Starting process_page: {filename} (page {page_num+1})")
        try:
            page = doc.load_page(page_num)
            raw_text = page.get_text()
        except Exception as e:
            logger.error(f"Error reading page {page_num+1} of {filename}: {e}")
            return False

        if not raw_text or len(raw_text.split()) < 20:
            logger.info(f"Skipping page {page_num+1} (insufficient text).")
            return False

        schema_ctx = self.schema_helper.get_schema_context_for_llm()
        prompt = self._prepare_prompt(schema_ctx, filename, page_num, raw_text)

        start = datetime.datetime.utcnow()
        try:
            logger.debug(f"Calling LLM for {filename} page {page_num+1}")
            response = self.call_ollama(prompt)
            latency = (datetime.datetime.utcnow() - start).total_seconds()
            logger.debug(f"LLM response for {filename} page {page_num+1} in {latency:.2f}s")
        except Exception as e:
            logger.error(f"Ollama failed for {filename} page {page_num+1}: {e}")
            return False

        try:
            data = self.extract_json_from_text(response)
            logger.debug(f"Parsed JSON for {filename} page {page_num+1}")
        except Exception as e:
            logger.error(f"JSON parsing failed on {filename} page {page_num+1}: {e}")
            return False

        if not data.get("terms"):
            logger.info(f"No terms found on {filename} page {page_num+1}.")
            # Even if no new terms, we may still want to extract properties if existing materials appear
            new_or_updated = self._extract_and_attach_properties(raw_text)
            return new_or_updated

        added_or_updated = False
        page_terms: List[str] = []

        for raw_term in data["terms"]:
            name = raw_term.get("term", "").strip()
            if not name:
                continue

            page_terms.append(name)
            # 1) Validate/fix category & drop prohibited relations
            fixed_term = self.schema_helper.validate_and_fix_term(raw_term)

            # 2) Ensure formula & validation
            snippet = self.get_context_snippet(raw_text, name, filename, page_num)
            validated_term = self._postprocess_term(fixed_term, snippet["text"])

            # maybe only do this if it's a material or chemical...
            # it found an entry for Fill Factor (FF)
            # 3) Enrich with ChEBI, if available
            if self.chebi_lookup:
                try:
                    chebi_info = self.chebi_lookup.lookup(name)
                    if chebi_info:
                        validated_term["chebi"] = chebi_info
                        # If no formula from LLM, pull from ChEBI
                        if not validated_term.get("formula") and chebi_info.get("formula"):
                            validated_term["formula"] = chebi_info["formula"]
                            validated_term["formula_validation"] = {"status": "from_chebi"}
                        # Add SMILES, charge, InChI, InChIKey if missing
                        for key in ("smiles", "charge", "inchi", "inchikey", "mass"):
                            if chebi_info.get(key) and not validated_term.get(key):
                                validated_term[key] = chebi_info[key]
                except Exception as e:
                    logger.warning(f"ChEBI lookup failed for '{name}': {e}")

            # 4) Merge into terms_dict
            key = self.normalize_term(name)
            existing_key = self.fuzzy_merge(name)

            # Helper to compare relation tuples
            def relation_tuple(rel: Dict[str, Any]) -> Tuple[str, str]:
                return (rel["relation"], rel["related_term"])

            if existing_key:
                entry = self.terms_dict[existing_key]
                if (page_num + 1) not in entry.get("pages", []):
                    entry.setdefault("pages", []).append(page_num + 1)
                    entry.setdefault("source_papers", []).append(filename)
                    entry.setdefault("context_snippets", []).append(snippet)
                    added_or_updated = True

                ex_rels = entry.setdefault("relations", [])
                existing_rel_tups = {relation_tuple(r) for r in ex_rels}
                for new_rel in validated_term.get("relations", []):
                    tup = relation_tuple(new_rel)
                    if tup not in existing_rel_tups:
                        subj_cat = entry.get("category", "")
                        obj_key = self.normalize_term(new_rel["related_term"])
                        obj_entry = self.terms_dict.get(obj_key)
                        obj_cat = obj_entry.get("category") if obj_entry else ""
                        if new_rel.get("verified", False) and subj_cat and obj_cat:
                            if not self.schema_helper.check_relation_validity(
                                subj_cat, new_rel["relation"], obj_cat
                            ):
                                logger.warning(
                                    f"Relation '{new_rel['relation']}' invalid for {subj_cat}→{obj_cat}, marking verified=false"
                                )
                                new_rel["verified"] = False
                        ex_rels.append(new_rel)
                        added_or_updated = True

                new_def = validated_term.get("definition", "")
                if len(new_def) > len(entry.get("definition", "")):
                    entry["definition"] = new_def
                    added_or_updated = True

                # Update any missing chemical details from ChEBI
                if validated_term.get("chebi"):
                    for chem_key in ("chebi", "formula", "formula_validation", "smiles", "charge", "inchi", "inchikey", "mass"):
                        if validated_term.get(chem_key) and not entry.get(chem_key):
                            entry[chem_key] = validated_term[chem_key]
                            added_or_updated = True

            else:
                new_key = self._register_new_term(name)
                entry: Dict[str, Any] = {
                    "term": name,
                    "definition": validated_term.get("definition", ""),
                    "category": validated_term.get("category", "Thing"),
                    "formula": validated_term.get("formula"),
                    "formula_validation": validated_term.get("formula_validation"),
                    "relations": [],
                    "pages": [page_num + 1],
                    "source_papers": [filename],
                    "context_snippets": [snippet],
                }
                # include any ChEBI enrichment
                for chem_key in ("chebi", "smiles", "charge", "inchi", "inchikey", "mass"):
                    if validated_term.get(chem_key):
                        entry[chem_key] = validated_term[chem_key]

                # add relations
                for rel in validated_term.get("relations", []):
                    subj_cat = entry["category"]
                    obj_key = self.normalize_term(rel["related_term"])
                    obj_entry = self.terms_dict.get(obj_key)
                    obj_cat = obj_entry.get("category") if obj_entry else ""
                    if rel.get("verified", False) and subj_cat and obj_cat:
                        if not self.schema_helper.check_relation_validity(
                            subj_cat, rel["relation"], obj_cat
                        ):
                            logger.warning(
                                f"Relation '{rel['relation']}' invalid for {subj_cat}→{obj_cat}, marking verified=false"
                            )
                            rel["verified"] = False
                    entry["relations"].append(rel)

                self.terms_dict[new_key] = entry
                added_or_updated = True
                logger.info(f"Added new term '{name}' (page {page_num+1} of {filename})")

        # 5) After terms are merged, extract + normalize properties for all known materials on this page
        prop_updated = self._extract_and_attach_properties(raw_text)

        if added_or_updated or prop_updated:
            self._save_terms_threadsafe()

        return added_or_updated or prop_updated

    def _extract_and_attach_properties(self, full_text: str) -> bool:
        """
        Find numeric property mentions for any known material names in full_text,
        normalize them, and merge into `self.terms_dict`.
        Returns True if any properties were added or updated.
        """
        if not self.terms_dict:
            return False

        material_names = [t["term"] for t in self.terms_dict.values()]
        raw_props = self.prop_extractor.extract(full_text, material_names)
        if not raw_props:
            return False

        normalized_props = self.prop_normalizer.normalize(raw_props)
        updated = False

        for p in normalized_props:
            mat = p["material"]
            mat_key = self.normalize_term(mat)
            if mat_key not in self.terms_dict:
                continue

            term_record = self.terms_dict[mat_key]
            props_list = term_record.setdefault("properties", [])

            existing_tuples = {
                (pr["property"], pr["value"], pr["unit"], pr["context"]) for pr in props_list
            }
            prop_tuple = (p["property"], p["normalized_value"], p["normalized_unit"], p["context"])
            if prop_tuple not in existing_tuples:
                props_list.append({
                    "property": p["property"],
                    "value": p["normalized_value"],
                    "unit": p["normalized_unit"],
                    "uncertainty": p.get("uncertainty_value"),
                    "context": p["context"],
                    "verified": not p["unit_conversion_failed"]
                })
                logger.info(f"Attached property '{p['property']}' to material '{mat}'")
                updated = True

        return updated

    def process_pdf(self, pdf_path: str) -> int:
        """
        Open PDF and process all pages in parallel.
        Returns number of pages that added/updated ≥1 term or property.
        Updates metadata['processed_pages_total'] and 'processed_pages_with_terms'.
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Cannot open PDF {pdf_path}: {e}")
            return 0

        total_pages = doc.page_count
        self.metadata["processed_pages_total"] += total_pages
        pages_with_terms = 0

        logger.debug(f"Processing '{pdf_path}' ({total_pages} pages) with {self.max_workers} workers")
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {
                exe.submit(self.process_page, doc, pdf_path, i): i for i in range(total_pages)
            }
            for fut in as_completed(futures):
                page_i = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        pages_with_terms += 1
                        logger.debug(
                            f"Page {page_i+1}/{total_pages} of {os.path.basename(pdf_path)} yielded terms or properties"
                        )
                    else:
                        logger.debug(
                            f"Page {page_i+1}/{total_pages} of {os.path.basename(pdf_path)} had no new terms or properties"
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing page {page_i+1} of {os.path.basename(pdf_path)}: {e}"
                    )

        self.metadata["processed_files"] += 1
        self.metadata["processed_pages_with_terms"] += pages_with_terms
        logger.info(
            f"Finished '{os.path.basename(pdf_path)}': {pages_with_terms}/{total_pages} pages yielded terms or properties"
        )
        return pages_with_terms

    def process_directory(self) -> Dict[str, Any]:
        """
        Walk data_dir for all PDFs, process them, then compute final importance metrics.
        Saves output JSON one final time.
        """
        if not os.path.isdir(self.data_dir):
            msg = f"Directory not found: {self.data_dir}"
            logger.error(msg)
            return {"status": "error", "message": msg}

        pdfs = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]
        if not pdfs:
            logger.warning(f"No PDFs in {self.data_dir}")

        total_files = 0
        for idx, fname in enumerate(pdfs, start=1):
            fullpath = os.path.join(self.data_dir, fname)
            logger.info(f"[{idx}/{len(pdfs)}] Processing file: {fname}")
            self.process_pdf(fullpath)
            total_files += 1

        # Assign importance
        for term_data in self.terms_dict.values():
            occ = len(term_data.get("pages", []))
            papers = len(set(term_data.get("source_papers", [])))
            if papers > 1 or occ > 5:
                term_data["importance"] = "high"
            elif occ > 2:
                term_data["importance"] = "medium"
            else:
                term_data["importance"] = "low"

        # Final save
        self._save_terms_threadsafe()
        logger.info(
            f"Done! Files processed: {total_files}, Pages total: {self.metadata['processed_pages_total']}, "
            f"Pages w/ terms: {self.metadata['processed_pages_with_terms']}, Unique terms: {len(self.terms_dict)}"
        )
        return {
            "status": "success",
            "processed_files": total_files,
            "processed_pages_total": self.metadata["processed_pages_total"],
            "processed_pages_with_terms": self.metadata["processed_pages_with_terms"],
            "unique_terms": len(self.terms_dict),
            "output_file": self.output_file,
        }

    def save_terms(self) -> None:
        """Legacy method; not used. Thread-safe saving is done via `_save_terms_threadsafe`."""
        raise NotImplementedError("Use `_save_terms_threadsafe` instead.")

# ----------------------------------------
# Script entrypoint
# ----------------------------------------
if __name__ == "__main__":
    extractor = OllamaTermExtractor(
        ollama_model="mistral-small3.1:latest",
        ollama_base_url="http://localhost:11434",
        temperature=0.0,
        data_dir="./polymer_papers",
        output_file="./storage/terminology/extracted_terms.json",
        context_length=50,
        schema_path="matkg_schema.yaml",
        max_workers=16,
    )
    result = extractor.process_directory()
    if result.get("status") == "error":
        print(f"✗ Error: {result.get('message')}")
        print(f"  Expected directory: {extractor.data_dir}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
    else:
        print(
            f"✓ Processed {result['processed_files']} files, "
            f"{result['processed_pages_total']} total pages, "
            f"{result['processed_pages_with_terms']} pages with terms, "
            f"{result['unique_terms']} unique terms → {result['output_file']}"
        )
