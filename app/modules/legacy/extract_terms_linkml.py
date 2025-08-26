#!/usr/bin/env python3
"""extract_terms_linkml.py – FAST PDF → LinkML term extractor

Run: ``python extract_terms_linkml.py -i ./pdf_dir -o ./terms.json``
"""
from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import os
import re
import json
import sys
import logging
import datetime
import threading
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss                            # ANN (GPU‑aware)
import fitz                             # == PyMuPDF
import numpy as np
import requests
import torch                            # required by SentenceTransformer
from linkml_runtime.utils.schemaview import SchemaView
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import nvtx
from nvtx import annotate

# --- local helpers -----------------------------------------------------------
from agents.chebi import ChebiOboLookup
from agents.chem_checker import ChemicalFormulaValidator
from agents.properties import PhysicalPropertyExtractor, PropertyNormalizer

###############################################################################
# Global LLM PROMPTS  (VERBATIM – do not edit)
###############################################################################

_FEW_SHOT_EXAMPLE = r"""
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

_EXTRACTION_TEMPLATE = r"""
=== EXTRACTION TASK ===
schema_context:
{schema_ctx}

PAPER: {filename}
PAGE: {pnum}

CONTENT:
{text}

INSTRUCTIONS:
1. Extract key materials‐science terms + their relations using ONLY schema slots.
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

_MERGE_PROMPT_TEMPLATE = r"""
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

###############################################################################
# Logging & retry decorator
###############################################################################

class _AnsiColorFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    _RESET = "\033[0m"

    def format(self, record):
        if (c := self._COLORS.get(record.levelname)):
            record.levelname = f"{c}{record.levelname}{self._RESET}"
            record.msg = f"{c}{record.getMessage()}{self._RESET}"
            record.args = ()
        return super().format(record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(_AnsiColorFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("OllamaTermExtractor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(handler)

TExc = Tuple[type, ...] | type

def retry_on_exception(exceptions: TExc, retries: int = 2, delay_seconds: float = 1.0):
    def dec(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            last: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return fn(*a, **kw)
                except exceptions as e:
                    last = e
                    logger.warning("%s failed (%d/%d): %s", fn.__name__, attempt + 1, retries + 1, e)
                    if attempt < retries:
                        threading.Event().wait(delay_seconds * (2 ** attempt))
            raise last
        return wrap
    return dec

###############################################################################
# ANNIndexer
###############################################################################

class ANNIndexer:
    """Incremental cosine‑similarity ANN using FAISS (GPU if available)."""
    
    @annotate("ANNIndexer_init")
    def __init__(self, model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2", k: int = 12):
        self.model = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
        dim = self.model.get_sentence_embedding_dimension()
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.index_cpu_to_all_gpus(base) if torch.cuda.is_available() else base
        self.display: list[str] = []
        self.k = k
        self._lock = threading.Lock()

    @annotate("_enc")
    def _enc(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32", copy=False)
    
    @annotate("add")
    def add(self, term: str) -> None:
        with self._lock:
            self.index.add(self._enc([term]))
            self.display.append(term)

    @annotate("add_many")
    def add_many(self, terms: list[str], batch: int = 2048) -> None:
        for i in range(0, len(terms), batch):
            vec = self._enc(terms[i:i+batch])
            with self._lock:
                self.index.add(vec)
                self.display.extend(terms[i:i+batch])

    @annotate("query")
    def query(self, text: str, k: int | None = None) -> list[str]:
        if not self.display:
            return []
        with self._lock:
            _, idx = self.index.search(self._enc([text]), min(k or self.k, len(self.display)))
        return [self.display[i] for i in idx[0] if i != -1]

###############################################################################
# Helper regex & canonical key
###############################################################################
_PUNC_RE = re.compile(r"[\s_\-/\[\]\(\)]")

@annotate("_canon")
def _canon(s: str) -> str:
    return _PUNC_RE.sub("", s).casefold()

###############################################################################
# SchemaHelper (unchanged)
###############################################################################
class SchemaHelper:
    """LinkML schema utilities (exact copy of original implementation)."""
    @annotate("SchemaHelper_init")
    def __init__(self, schema_path: str = "matkg_schema.yaml", fuzzy_cutoff: int = 80):
        self.schema_view = SchemaView(schema_path)
        self.fuzzy_cutoff = fuzzy_cutoff
        self._load()
        self._build_maps()

    # ---------------- private loaders ----------------
    @annotate("_load")
    def _load(self):
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.class_parents: Dict[str, Optional[str]] = {}
        for n, c in self.schema_view.all_classes().items():
            self.classes[n] = {"description": c.description or f"A {n} entity", "slots": []}
            self.class_parents[n] = c.is_a or None
        self.slots: Dict[str, Dict[str, Any]] = {}
        for n, s in self.schema_view.all_slots().items():
            self.slots[n] = {
                "description": s.description or f"Relationship: {n}",
                "domain": s.domain or None,
                "range": s.range or None,
                "multivalued": bool(s.multivalued),
            }
            if s.domain and s.domain in self.classes:
                self.classes[s.domain]["slots"].append(n)
        logger.info("Loaded schema: %d classes, %d slots", len(self.classes), len(self.slots))
    @annotate("_build_maps")
    def _build_maps(self):
        self._class_map_lower = {c.lower(): c for c in self.classes}
        self._slot_map_lower = {s.lower(): s for s in self.slots}
        self._class_names_lower = list(self._class_map_lower)
        self._slot_names_lower = list(self._slot_map_lower)

    # ---------------- context for LLM ----------------
    @annotate("get_schema_context_for_llm")
    def get_schema_context_for_llm(self) -> str:
        lines: List[str] = ["=== KNOWLEDGE SCHEMA ===\n", "ENTITY TYPES (use exactly these names):"]
        for cls in sorted(self.classes):
            desc = self.classes[cls]["description"]
            parent = self.class_parents[cls]
            lines.append(f"- {cls}: {desc}" + (f"  (inherits from: {parent})" if parent else ""))
        lines.append("\nVALID RELATIONSHIPS (use exactly these names):")
        for slot in sorted(self.slots):
            info = self.slots[slot]
            dom = info["domain"] or "Any"
            rng = info["range"] or "Any"
            mv  = " (multivalued)" if info["multivalued"] else ""
            lines.append(f"- {slot}: {info['description']}  Usage: {dom} --{slot}--> {rng}{mv}")
        lines.append("\nIMPORTANT: Do NOT use relations named 'description' or 'category'.")
        return "\n".join(lines)

    # ---------------- fuzzy helpers ----------------
    @annotate("_find_closest_class")
    def _find_closest_class(self, target: str) -> Optional[str]:
        tl = target.strip().lower();
        if tl in self._class_map_lower:
            return self._class_map_lower[tl]
        m = process.extractOne(tl, self._class_names_lower, scorer=fuzz.QRatio, score_cutoff=self.fuzzy_cutoff)
        return self._class_map_lower.get(m[0]) if m else None

    @annotate("_find_closest_slot")
    def _find_closest_slot(self, target: str) -> Optional[str]:
        tl = target.strip().lower();
        if tl in self._slot_map_lower:
            return self._slot_map_lower[tl]
        m = process.extractOne(tl, self._slot_names_lower, scorer=fuzz.QRatio, score_cutoff=self.fuzzy_cutoff)
        return self._slot_map_lower.get(m[0]) if m else None

    # ---------------- public validation helpers ----------------
    @annotate("validate_and_fix_term")
    def validate_and_fix_term(self, t: Dict[str, Any]) -> Dict[str, Any]:
        cat = t.get("category", "").strip()
        if cat not in self.classes:
            fixed = self._find_closest_class(cat)
            if fixed:
                logger.warning("Fixed category '%s' → '%s'", cat, fixed)
                t["category"] = fixed
            else:
                logger.warning("Unknown category '%s' (left as‑is)", cat)
        cleaned: List[Dict[str, Union[str, bool]]] = []
        for rel in t.get("relations", []):
            pred = rel.get("relation", "").strip()
            obj  = rel.get("related_term", "").strip()
            if pred.lower() in ("description", "category"):
                continue
            if pred in self.slots:
                cleaned.append({"relation": pred, "related_term": obj, "verified": True})
            else:
                fixed_slot = self._find_closest_slot(pred)
                if fixed_slot:
                    logger.warning("Fixed relation '%s' → '%s'", pred, fixed_slot)
                    cleaned.append({"relation": fixed_slot, "related_term": obj, "verified": True})
                else:
                    cleaned.append({"relation": pred, "related_term": obj, "verified": False})
        t["relations"] = cleaned
        return t

    @annotate("_is_subclass_of")
    def _is_subclass_of(self, child: str, parent: str) -> bool:
        if child == parent:
            return True
        p = self.class_parents.get(child)
        return self._is_subclass_of(p, parent) if p else False

    @annotate("check_relation_validity")
    def check_relation_validity(self, subj: str, pred: str, obj: str) -> bool:
        slot = self.slots.get(pred)
        if not slot:
            return False
        d, r = slot["domain"], slot["range"]
        return (not d or self._is_subclass_of(subj, d)) and (not r or self._is_subclass_of(obj, r))

###############################################################################
# OllamaTermExtractor – full pipeline
###############################################################################
class OllamaTermExtractor:
    """PDF → LinkML term extractor with ANN‑accelerated duplicate handling."""
    @annotate("OllamaTermExtractor_init")
    def __init__(self, *,
                 ollama_model: str = "mistral-small3.1-32k", # "mistral-small3.1:latest",
                 ollama_base_url: str = "http://localhost:11434",
                 temperature: float = 0.0,
                 data_dir: str = "./polymer_papers",
                 output_file: str = "./storage/terminology/extracted_terms.json",
                 context_length: int = 50,
                 schema_path: str = "matkg_schema.yaml",
                 max_workers: int = 16):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature = temperature
        self.data_dir = data_dir
        self.output_file = output_file
        self.context_length = context_length
        self.max_workers = max_workers

        self.schema_helper = SchemaHelper(schema_path=schema_path)
        self.prop_extractor = PhysicalPropertyExtractor()
        self.prop_normalizer = PropertyNormalizer()
        self.formula_checker = ChemicalFormulaValidator(api_key=os.getenv("MP_API_KEY", ""))
        try:
            self.chebi_lookup = ChebiOboLookup("storage/ontologies/chebi.obo")
        except Exception as e:
            logger.warning("ChEBI load failed: %s", e)
            self.chebi_lookup = None

        self.terms_dict: Dict[str, Dict[str, Any]] = {}
        self._bk_terms: Dict[str, str] = {}
        self._canon_map: Dict[str, str] = {}
        self._ann = ANNIndexer()

        # Load existing terms if present
        if os.path.exists(self.output_file):
            try:
                prev = json.load(open(self.output_file))
                for t in prev.get("terms", []):
                    key = t["term"].strip().lower()
                    self.terms_dict[key] = t
                    self._bk_terms[t["term"]] = key
                    self._canon_map[_canon(t["term"])] = t["term"]
                self._ann.add_many(list(self._bk_terms))
                logger.info("Loaded %d existing terms", len(self._bk_terms))
            except Exception as e:
                logger.warning("Could not load previous terms: %s", e)

        self.metadata = {
            "extraction_date": datetime.datetime.utcnow().isoformat() + "Z",
            "processed_files": 0,
            "processed_pages_total": 0,
            "processed_pages_with_terms": 0,
            "version": "3.0",
        }
        self._save_lock = threading.Lock()

    # ---------------- duplicate handling ----------------
    @annotate("_register_new_term")
    def _register_new_term(self, display: str) -> str:
        key = display.strip().lower()
        self._bk_terms[display] = key
        self._canon_map[_canon(display)] = display
        self._ann.add(display)
        return key

    @annotate("fuzzy_merge")
    def fuzzy_merge(self, term: str) -> Optional[str]:
        if not self._bk_terms:
            return None
        disp = self._canon_map.get(_canon(term))
        if disp:
            return self._bk_terms[disp]
        cands = self._ann.query(term)
        if not cands:
            return None
        prompt = _MERGE_PROMPT_TEMPLATE.format(term=term, bullets="\n".join(f"- {c}" for c in cands))
        try:
            resp = self.call_ollama(prompt).strip()
        except Exception as e:
            logger.warning("LLM merge failed: %s", e)
            return None
        return self._bk_terms.get(resp)

    # ---------------- LLM helpers ----------------
    @annotate("call_ollama")
    @retry_on_exception((requests.exceptions.RequestException,), retries=2, delay_seconds=2.0)
    def call_ollama(self, prompt: str, timeout: int = 240) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        r = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "") or ""

    @annotate("extract_json_from_text")
    @retry_on_exception((Exception,), retries=2, delay_seconds=1.0)
    def extract_json_from_text(self, text: str) -> Dict[str, Any]:
        pat = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        matches = sorted(re.finditer(pat, text), key=lambda m: -len(m.group(0)))
        for m in matches:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "terms" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
        return {"terms": []}

    # ---------------- formula / context helpers ----------------
    @annotate("_looks_like_formula")
    def _looks_like_formula(self, s: str) -> bool:
        return bool(re.search(r"[A-Z][a-z]?[\d]", s or ""))

    @annotate("_postprocess_term")
    def _postprocess_term(self, t: Dict[str, Any], ctx: str) -> Dict[str, Any]:
        f = t.get("formula")
        if not f or not self._looks_like_formula(f):
            t["formula"] = None
            t["formula_validation"] = {"status": "missing"}
            return t
        try:
            val = self.formula_checker.validate(f)
        except Exception as e:
            logger.warning("Formula validation error '%s': %s", f, e)
            val = {"status": "error", "details": {"error": str(e)}}
        if val.get("status") == "invalid":
            repair_prompt = (
                f"The extracted string '{f}' is not a valid chemical formula.\n"
                "Based on the context below, guess the correct formula and return ONLY the formula string.\n\n"
                f"CONTEXT:\n{ctx}"
            )
            try:
                cand = self.call_ollama(repair_prompt, timeout=120).strip().split()[0]
                newval = self.formula_checker.validate(cand)
                if newval.get("status") != "invalid":
                    t["formula"] = cand
                    val = newval | {"status": "corrected"}
            except Exception as e:
                logger.warning("LLM formula repair failed: %s", e)
        t["formula_validation"] = val
        return t

    # ---------------- context & prompt ----------------
    @annotate("get_context_snippet")
    def get_context_snippet(self, full: str, term: str, fname: str, page: int):
        sents = re.split(r"(?<=[.!?])\s+", full)
        low = term.lower()
        for s in sents:
            if low in s.lower():
                return {
                    "text": " ".join(s.split()[: self.context_length]),
                    "source_paper": fname,
                    "page": page + 1,
                }
        return {
            "text": " ".join(full.split()[: self.context_length]),
            "source_paper": fname,
            "page": page + 1,
        }
    
    @annotate("_prepare_prompt")
    def _prepare_prompt(self, schema_ctx: str, fname: str, pnum: int, text: str) -> str:
        page_text = text[-8000:] if len(text) > 8000 else text
        return _EXTRACTION_TEMPLATE.format(schema_ctx=schema_ctx, filename=fname, pnum=pnum + 1, text=page_text) + "\n" + _FEW_SHOT_EXAMPLE

    # ---------------- thread‑safe save ----------------
    @annotate("_save_terms_threadsafe")
    def _save_terms_threadsafe(self):
        with self._save_lock:
            try:
                for t in self.terms_dict.values():
                    t.setdefault("properties", [])
                out = {"metadata": self.metadata, "terms": list(self.terms_dict.values())}
                json.dump(out, open(self.output_file, "w"), indent=2)
                logger.debug("Saved %d terms", len(self.terms_dict))
            except Exception as e:
                logger.error("Save failed: %s", e)

    # ---------------- property extraction helper ----------------
    @annotate("_extract_and_attach_properties")
    def _extract_and_attach_properties(self, text: str) -> bool:
        if not self.terms_dict:
            return False
        mats = [t["term"] for t in self.terms_dict.values()]
        raw = self.prop_extractor.extract(text, mats)
        if not raw:
            return False
        norm = self.prop_normalizer.normalize(raw)
        changed = False
        for p in norm:
            key = p["material"].strip().lower()
            rec = self.terms_dict.get(key)
            if not rec:
                continue
            tpl = (p["property"], p["normalized_value"], p["normalized_unit"], p["context"])
            existing = {(r["property"], r["value"], r["unit"], r["context"]) for r in rec["properties"]}
            if tpl not in existing:
                rec["properties"].append({
                    "property": p["property"],
                    "value": p["normalized_value"],
                    "unit": p["normalized_unit"],
                    "uncertainty": p.get("uncertainty_value"),
                    "context": p["context"],
                    "verified": not p["unit_conversion_failed"],
                })
                changed = True
        return changed

    # ---------------- main page processor ----------------
    @annotate("process_page")
    def process_page(self, doc: fitz.Document, path: str, num: int) -> bool:
        fname = os.path.basename(path)
        try:
            with nvtx.annotate("load_page", color="green"):
                page = doc.load_page(num)
                text = page.get_text()
        except Exception as e:
            logger.error("Read error %s page %d: %s", fname, num + 1, e)
            return False
        if not text or len(text.split()) < 20:
            return False
        prompt = self._prepare_prompt(self.schema_helper.get_schema_context_for_llm(), fname, num, text)
        try:
            response = self.call_ollama(prompt)
            data = self.extract_json_from_text(response)
        except Exception as e:
            logger.error("LLM/JSON failure on %s page %d: %s", fname, num + 1, e)
            return False
        updated = False
        
        terms_rng = nvtx.start_range(message="process_terms", color="blue")
        if data.get("terms"):
            for raw in data["terms"]:
                name = raw.get("term", "").strip()
                if not name:
                    continue
                prep_rng = nvtx.start_range(message="preprocess_clean_new_terms", color="blue")
                fixed = self.schema_helper.validate_and_fix_term(raw)
                snippet = self.get_context_snippet(text, name, fname, num)
                term_pp = self._postprocess_term(fixed, snippet["text"])
                nvtx.end_range(prep_rng)

                # ChEBI enrich
                
                chebi_rng = nvtx.start_range(message="chebi_rng", color="blue")
                if self.chebi_lookup:
                    try:
                        info = self.chebi_lookup.lookup(name)
                        if info:
                            term_pp["chebi"] = info
                            if not term_pp.get("formula") and info.get("formula"):
                                term_pp["formula"] = info["formula"]
                                term_pp["formula_validation"] = {"status": "from_chebi"}
                            for k in ("smiles", "charge", "inchi", "inchikey", "mass"):
                                if info.get(k) and not term_pp.get(k):
                                    term_pp[k] = info[k]
                    except Exception as e:
                        logger.warning("ChEBI lookup failed for '%s': %s", name, e)
                nvtx.end_range(chebi_rng)

                
                merge_rng = nvtx.start_range(message="merge_new_terms", color="blue")
                existing_key = self.fuzzy_merge(name)
                key_norm = name.lower()
                if existing_key:
                    rec = self.terms_dict[existing_key]
                else:
                    existing_key = self._register_new_term(name)
                    rec = {
                        "term": name,
                        "definition": "",
                        "category": "Thing",
                        "formula": None,
                        "formula_validation": None,
                        "relations": [],
                        "pages": [],
                        "source_papers": [],
                        "context_snippets": [],
                        "properties": [],
                    }
                    self.terms_dict[existing_key] = rec

                
                # merge basic fields
                if term_pp.get("definition") and len(term_pp["definition"]) > len(rec.get("definition", "")):
                    rec["definition"] = term_pp["definition"]
                rec.setdefault("category", term_pp.get("category", "Thing"))
                for k in ("formula", "formula_validation", "chebi", "smiles", "charge", "inchi", "inchikey", "mass"):
                    if term_pp.get(k) and not rec.get(k):
                        rec[k] = term_pp[k]
                nvtx.end_range(merge_rng)

                # pages / context
                if num + 1 not in rec["pages"]:
                    rec["pages"].append(num + 1)
                    rec["source_papers"].append(fname)
                    rec["context_snippets"].append(snippet)
                    updated = True
                
                # relations
                rel_rng = nvtx.start_range(message="process_related_terms", color="blue")
                existing_tuples = {(r["relation"], r["related_term"]) for r in rec["relations"]}
                for rel in term_pp.get("relations", []):
                    tup = (rel["relation"], rel["related_term"])
                    if tup not in existing_tuples:
                        rec["relations"].append(rel)
                        updated = True
                nvtx.end_range(rel_rng)
        # properties
        updated |= self._extract_and_attach_properties(text)
        nvtx.end_range(terms_rng)
        if updated:
            self._save_terms_threadsafe()
        return updated

    # ---------------- PDF & directory drivers ----------------
    @annotate("process_pdf")
    def process_pdf(self, path: str) -> int:
        try:
            with nvtx.annotate("open_pdf", color="green"):
                doc = fitz.open(path)
        except Exception as e:
            logger.error("Cannot open PDF %s: %s", path, e)
            return 0
        total = doc.page_count
        logger.info("Processing %s (%d pages)", os.path.basename(path), total)
        self.metadata["processed_pages_total"] += total
        pages_done = 0
        pages_with_terms = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.process_page, doc, path, i): i for i in range(total)}
            for fut in as_completed(futs):
                pages_done += 1
                try:
                    if fut.result():
                        pages_with_terms += 1
                except Exception as e:
                    logger.error("Worker error on %s page %d: %s", path, futs[fut] + 1, e)
                if pages_done % 25 == 0 or pages_done == total:
                    logger.info("…%s progress: %d/%d pages finished (%d with terms)", os.path.basename(path), pages_done, total, pages_with_terms)
        self.metadata["processed_pages_with_terms"] += pages_with_terms
        self.metadata["processed_files"] += 1
        logger.info("Finished %s: %d/%d pages w/ terms", os.path.basename(path), pages_with_terms, total)
        return pages_with_terms

    @annotate("process_directory")
    def process_directory(self) -> Dict[str, Any]:
        if not os.path.isdir(self.data_dir):
            return {"status": "error", "message": f"Directory not found: {self.data_dir}"}
        pdfs = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]
        if not pdfs:
            logger.warning("No PDFs found under %s", self.data_dir)
        for i, f in enumerate(pdfs, 1):
            logger.info("[%d/%d] %s", i, len(pdfs), f)
            self.process_pdf(os.path.join(self.data_dir, f))
        # importance flag
        for t in self.terms_dict.values():
            occ = len(t["pages"])
            papers = len(set(t["source_papers"]))
            t["importance"] = "high" if papers > 1 or occ > 5 else "medium" if occ > 2 else "low"
        self._save_terms_threadsafe()
        return {
            "status": "success",
            "processed_files": self.metadata["processed_files"],
            "processed_pages_total": self.metadata["processed_pages_total"],
            "processed_pages_with_terms": self.metadata["processed_pages_with_terms"],
            "unique_terms": len(self.terms_dict),
            "output_file": self.output_file,
        }

    # legacy
    def save_terms(self):
        raise NotImplementedError("Use internal save")

###############################################################################
# CLI
###############################################################################
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="./polymer_paper", help="PDF directory")
    p.add_argument("-o", "--output", default="./storage/terminology/extracted_terms_profile.json", help="JSON output path")
    args = p.parse_args()

    ext = OllamaTermExtractor(data_dir=args.input, output_file=args.output)
    res = ext.process_directory()
    if res.get("status") != "success":
        sys.exit("Extraction failed: " + res.get("message", "unknown error"))
    print(json.dumps(res, indent=2))
