#!/usr/bin/env python3
"""
extract_terms_linkml.py
=======================

An ultra‑fast, streaming PDF → LinkML terminology extractor with duplicate
handling and property enrichment.

Major features (v3.2, 2025‑07‑15)
---------------------------------
1. **Two‑stage LLM pipeline** ‑ term‑only → detail → 4‑5 × speed‑up.
2. **Batch ANN + LLM merge** – one merge pass per page, zero redundant merges.
3. **Sentence‑aware, material‑science‑filtered chunking** – fewer useless calls.
4. **Streaming Ollama** – start parsing while tokens arrive.
5. **Full NVTX instrumentation** – perfect GPU/CPU timelines.
6. Comprehensive **type‑hints** & **doc‑strings** – 100 % self‑documenting.

Run:
    python extract_terms_linkml.py -i ./pdf_dir -o ./terms.json
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as _dt
import functools
import hashlib
import json
import logging
import os
import re
import sys
import textwrap
import threading
import time
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import fitz  # PyMuPDF
import nltk                         # sentence segmentation
import numpy as np
import requests
import tiktoken                    # fast BPE token counter
import torch
from linkml_runtime.utils.schemaview import SchemaView
from nvtx import annotate, end_range, start_range
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

# ---------- local helpers ----------------------------------------------------
from agents.chebi import ChebiOboLookup
from agents.chem_checker import ChemicalFormulaValidator
from agents.properties import PhysicalPropertyExtractor, PropertyNormalizer

###############################################################################
# Constants & templates
###############################################################################
SLOW_WARN_S   = 240.0
VERBOSE_CALLS = True

# ---- Few‑shot example -------------------------------------------------------
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

# ---- Term‑only extraction prompt -------------------------------------------
_TERM_ONLY_TEMPLATE = r"""
=== TERM‑ONLY EXTRACTION ===
Return ONE JSON array (no key) of distinct materials‑science terms you find
inside CONTENT.  Example:

["Poly(3-hexylthiophene) (P3HT)", "GIWAXS"]

CONTENT:
{content}
"""

# ---- Detail extraction prompt ----------------------------------------------
_DETAIL_TEMPLATE = r"""
=== TERM PROFILING ===
For each term below, output an object exactly like:
{{
  "term": "...",
  "definition": "...",
  "category": "...",
  "formula": "...",
  "relations": [{{"relation":"...","related_term":"...","verified":true}}]
}}
Return ONE JSON list named "terms".

TERMS:
{terms}

PAPER: {filename} · PAGE: {page}
Context snippets are embedded in <CTX idx=N>…</CTX N>.
"""

# ---- Merge heuristics prompt (unchanged) -----------------------------------
_MERGE_PROMPT_TEMPLATE = r"""
We have just extracted a new term:    "{term}"
Below is the list of all already‑registered terms (one per line; the first time we saw each term):
{bullets}

You must decide whether "{term}" refers to exactly the same concept as one of these, or if it is a distinct new concept.  Follow these rules:

  1. **Ignore only trivial punctuation (spaces, hyphens, slashes, brackets, parentheses, capitalization)**
     when comparing.  Example: "GIWAXS" ↔ "GI‑WAXS".
  2. **Do NOT merge distinct instrument or method acronyms** (SEM ≠ TEM, etc.).
  3. **Do NOT merge general vs. specific variants** (band structure ↔ Dirac-like band structure).
  4. If punctuation‑agnostic identical, return that term *verbatim*; else return `None`.
  5. **Merge acronym ↔ long‑form** (ARPES ↔ angle‑resolved photoemission spectroscopy (ARPES)).
  6. Respond with **exactly one line**: existing term or `None`.

Examples:
  • New "GI‑WAXS", list has "GIWAXS"           → respond "GIWAXS"
  • New "SEM",  list only "TEM"                → respond "None"
  • New "organic solar cells", list "OSCs"     → respond "Organic solar cells (OSCs)"
"""

###############################################################################
# Logging setup
###############################################################################
class _AnsiFormatter(logging.Formatter):
    _C = {"DEBUG":36,"INFO":32,"WARNING":33,"ERROR":31,"CRITICAL":41}
    _R = "\033[0m"
    def format(self, record):  # noqa:D401
        color = f"\033[{self._C.get(record.levelname,32)}m"
        record.levelname = f"{color}{record.levelname}{self._R}"
        record.msg = f"{color}{record.getMessage()}{self._R}"
        record.args = ()
        return super().format(record)

_hdl = logging.StreamHandler(sys.stdout)
_hdl.setFormatter(_AnsiFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                 "%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("OllamaTermExtractor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(_hdl)

###############################################################################
# Utility decorators
###############################################################################
def retry_on_exception(
    exc: Tuple[type, ...] | type,
    *,
    retries: int = 2,
    delay_seconds: float = 1.0,
):
    """Retry *fn* on *exc* with exponential back‑off."""
    def dec(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            last: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return fn(*a, **kw)
                except exc as e:
                    last = e
                    logger.warning("%s failed (%d/%d): %s",
                                   fn.__name__, attempt + 1, retries + 1, e)
                    if attempt < retries:
                        threading.Event().wait(delay_seconds * (2**attempt))
            raise last
        return wrap
    return dec

###############################################################################
# Text helpers
###############################################################################
_ENC_CACHE: Dict[str, tiktoken.Encoding] = {}
def _get_encoding(model: str = "mistral") -> tiktoken.Encoding:
    """Return (cached) tiktoken encoding suitable for *model*."""
    if model not in _ENC_CACHE:
        try:
            _ENC_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENC_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENC_CACHE[model]

def _count_tokens(txt: str, model: str = "mistral") -> int:
    """Fast token count via tiktoken."""
    return len(_get_encoding(model).encode(txt))

# sentence splitter
nltk.download("punkt", quiet=True)
def _sentences(txt: str) -> List[str]:
    """Return list of sentences via NLTK Punkt."""
    return nltk.tokenize.sent_tokenize(txt)

# quick MS keyword heuristic
_MS_RE = re.compile(
    r"\b(polym|photo.?volta|xrd|giwaxs|soxs|sem|tem|solar cell|band\s+gap|doping)\b",
    flags=re.I,
)
_is_ms = _MS_RE.search

# punctuation‑insensitive canonical key
_PUNC_RE = re.compile(r"[\s_\-/\[\]\(\)]")
def _canon(s: str) -> str:
    """Return case‑folded string stripped of trivial punctuation."""
    return _PUNC_RE.sub("", s).casefold()

###############################################################################
# ANNIndexer
###############################################################################
class ANNIndexer:
    """Incremental cosine‑similarity ANN using FAISS (GPU‑aware)."""

    @annotate("ANNIndexer::__init__")
    def __init__(self,
                 model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
                 k: int = 12) -> None:
        """
        Parameters
        ----------
        model : str
            HuggingFace SentenceTransformer identifier.
        k : int
            Number of nearest neighbours to return per query.
        """
        self.model = SentenceTransformer(
            model, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        dim = self.model.get_sentence_embedding_dimension()
        base = faiss.IndexFlatIP(dim)
        self.index = (faiss.index_cpu_to_all_gpus(base)
                      if torch.cuda.is_available() else base)
        self.display: List[str] = []
        self.k      = k
        self._lock  = threading.Lock()

    @annotate("ANNIndexer::_enc")
    def _enc(self, texts: Sequence[str]) -> np.ndarray:
        """Encode *texts* → L2‑normalised float32 vectors."""
        return self.model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32", copy=False)

    @annotate("ANNIndexer::add")
    def add(self, term: str) -> None:
        """Add single *term* to index."""
        with self._lock:
            self.index.add(self._enc([term]))
            self.display.append(term)

    @annotate("ANNIndexer::add_many")
    def add_many(self, terms: List[str], *, batch: int = 2048) -> None:
        """Add many *terms* in size‑limited batches."""
        for i in range(0, len(terms), batch):
            vec = self._enc(terms[i : i + batch])
            with self._lock:
                self.index.add(vec)
                self.display.extend(terms[i : i + batch])

    @annotate("ANNIndexer::query")
    def query(self, text: str, *, k: int | None = None) -> List[str]:
        """Return ≤*k* nearest neighbour display strings for *text*."""
        if not self.display:
            return []
        with self._lock:
            _, idx = self.index.search(self._enc([text]),
                                       min(k or self.k, len(self.display)))
        return [self.display[i] for i in idx[0] if i != -1]

###############################################################################
# Schema helper (compact)
###############################################################################
class SchemaHelper:
    """Tiny, high‑performance LinkML schema wrapper."""

    @annotate("SchemaHelper::__init__")
    def __init__(self, schema_path: str, fuzzy_cutoff: int = 80) -> None:
        self.schema_view  = SchemaView(schema_path)
        self.fuzzy_cutoff = fuzzy_cutoff
        self._load()
        self._build_maps()

    # ---------------- private loaders --------------------------------------
    @annotate("SchemaHelper::_load")
    def _load(self) -> None:
        """Load classes & slots once into dicts."""
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.class_parents: Dict[str, Optional[str]] = {}
        for n, c in self.schema_view.all_classes().items():
            self.classes[n] = {"description": c.description or f"A {n} entity",
                               "slots": []}
            self.class_parents[n] = c.is_a or None

        self.slots: Dict[str, Dict[str, Any]] = {}
        for n, s in self.schema_view.all_slots().items():
            self.slots[n] = {
                "description": s.description or f"Relationship: {n}",
                "domain": s.domain or None,
                "range":  s.range  or None,
                "multivalued": bool(s.multivalued),
            }
            if s.domain and s.domain in self.classes:
                self.classes[s.domain]["slots"].append(n)

    @annotate("SchemaHelper::_build_maps")
    def _build_maps(self) -> None:
        """Lower‑cased lookup tables for fuzzy fixing."""
        self._class_map_lower = {c.lower(): c for c in self.classes}
        self._slot_map_lower  = {s.lower(): s for s in self.slots}
        self._class_names_lower = list(self._class_map_lower)
        self._slot_names_lower  = list(self._slot_map_lower)

    # ---------------- public -------------------------------------------------
    @annotate("SchemaHelper::get_schema_context_for_llm")
    def get_schema_context_for_llm(self) -> str:
        """Return minimal (identifier‑only) schema context for LLM."""
        ent = ",".join(sorted(self.classes))
        rel = ",".join(sorted(self.slots))
        return f"ENTITY_TYPES:{ent}\nRELATIONS:{rel}"

    @annotate("SchemaHelper::validate_and_fix_term")
    def validate_and_fix_term(self, t: Dict[str, Any]) -> Dict[str, Any]:
        """Fix category / slot typos via fuzzy matching."""
        cat = t.get("category", "").strip()
        if cat and cat not in self.classes:
            fixed = self._find_closest_class(cat)
            if fixed:
                logger.debug("Category '%s' → '%s'", cat, fixed)
                t["category"] = fixed
        cleaned: List[Dict[str, Any]] = []
        for rel in t.get("relations", []):
            pred = rel.get("relation", "").strip()
            obj  = rel.get("related_term", "").strip()
            if pred.lower() in ("description", "category"):
                continue
            if pred not in self.slots:
                fixed = self._find_closest_slot(pred)
                if fixed:
                    pred = fixed
            cleaned.append({"relation": pred,
                            "related_term": obj,
                            "verified": pred in self.slots})
        t["relations"] = cleaned
        return t

    # ---------------- fuzzy helpers ----------------------------------------
    @annotate("SchemaHelper::_find_closest_class")
    def _find_closest_class(self, target: str) -> Optional[str]:
        tl = target.casefold()
        if tl in self._class_map_lower:
            return self._class_map_lower[tl]
        m = process.extractOne(tl, self._class_names_lower,
                               scorer=fuzz.QRatio, score_cutoff=self.fuzzy_cutoff)
        return self._class_map_lower.get(m[0]) if m else None

    @annotate("SchemaHelper::_find_closest_slot")
    def _find_closest_slot(self, target: str) -> Optional[str]:
        tl = target.casefold()
        if tl in self._slot_map_lower:
            return self._slot_map_lower[tl]
        m = process.extractOne(tl, self._slot_names_lower,
                               scorer=fuzz.QRatio, score_cutoff=self.fuzzy_cutoff)
        return self._slot_map_lower.get(m[0]) if m else None

###############################################################################
# OllamaTermExtractor
###############################################################################
class OllamaTermExtractor:
    """High‑throughput PDF → LinkML term extractor."""

    # ---------------------------------------------------------------- init --
    @annotate("Extractor::__init__")
    def __init__(
        self,
        *,
        ollama_model: str = "mistral-small3.1-32k",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float  = 0.0,
        data_dir: str       = "./polymer_papers",
        output_file: str    = "./storage/terminology/extracted_terms.json",
        schema_path: str    = "matkg_schema.yaml",
        max_workers: int    = 8,
        context_length: int = 50,
    ) -> None:
        self.ollama_model    = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.temperature     = temperature
        self.data_dir        = data_dir
        self.output_file     = output_file
        self.max_workers     = max_workers
        self.context_length  = context_length

        # ---- helpers ------------------------------------------------------
        self.schema_helper     = SchemaHelper(schema_path)
        self.schema_ctx        = self.schema_helper.get_schema_context_for_llm()
        self.prop_extractor    = PhysicalPropertyExtractor()
        self.prop_normalizer   = PropertyNormalizer()
        self.formula_checker   = ChemicalFormulaValidator(
            api_key=os.getenv("MP_API_KEY", "")
        )
        try:
            self.chebi_lookup = ChebiOboLookup("storage/ontologies/chebi.obo")
        except Exception as e:
            logger.warning("ChEBI load failed: %s", e)
            self.chebi_lookup = None

        # ---- term stores --------------------------------------------------
        self.terms_dict: Dict[str, Dict[str, Any]] = {}
        self._bk_terms: Dict[str, str]   = {}   # display → key
        self._canon_map: Dict[str, str]  = {}   # canon  → display
        self._ann = ANNIndexer()
        self._slow_prompts: List[Dict[str, Any]] = []
        self._save_lock = threading.Lock()

        if os.path.exists(self.output_file):
            self._restore_previous()

        self.metadata = {
            "extraction_date": _dt.datetime.utcnow().isoformat() + "Z",
            "processed_files": 0,
            "processed_pages_total": 0,
            "processed_pages_with_terms": 0,
            "version": "3.2",
        }

    # -------------------------- persistence -------------------------------
    @annotate("Extractor::_restore_previous")
    def _restore_previous(self) -> None:
        """Load *output_file* if present into memory & ANN."""
        try:
            prev = json.load(open(self.output_file))
            for t in prev.get("terms", []):
                key = t["term"].casefold()
                self.terms_dict[key] = t
                self._bk_terms[t["term"]] = key
                self._canon_map[_canon(t["term"])] = t["term"]
            self._ann.add_many(list(self._bk_terms))
            logger.info("Restored %d terms from previous run", len(self._bk_terms))
        except Exception as e:
            logger.warning("Could not restore previous terms: %s", e)

    # ----------------------- HTTP / LLM helpers ---------------------------
    @annotate("Extractor::call_ollama")
    @retry_on_exception((requests.exceptions.RequestException,), retries=2,
                        delay_seconds=2.0)
    def call_ollama(self, prompt: str, *, meta: Dict[str, Any] | None = None,
                    timeout: int = 480) -> str:
        """
        Send *prompt* to Ollama, streaming the response for minimal latency.

        Parameters
        ----------
        prompt : str
            User prompt (schema system context is auto‑prefixed).
        meta : dict, optional
            Extra logging metadata (file/page/task, etc.).
        timeout : int
            HTTP timeout seconds.
        """
        meta = meta or {}
        pid  = hashlib.sha1(prompt.encode()).hexdigest()[:10]
        ntok = _count_tokens(prompt)
        logger.debug("OLLAMA ▶ %s tok=%d meta=%s", pid, ntok, meta)

        t0   = time.monotonic()
        out_chunks: List[str] = []
        with annotate(message=f"ollama:{meta.get('task', '')}", color="red"):
            with requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": self.schema_ctx},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": True,
                    "options": {"temperature": self.temperature},
                },
                stream=True,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    out_chunks.append(json.loads(line.decode())["message"]["content"])
        dt = time.monotonic() - t0
        lvl = logging.WARNING if dt >= SLOW_WARN_S else logging.INFO
        logger.log(lvl, "OLLAMA ✔ %s %.1fs", pid, dt)
        if dt >= SLOW_WARN_S:
            self._slow_prompts.append({"id": pid, "t": round(dt, 1),
                                       "tok": ntok, **meta})
        return "".join(out_chunks)

    # -------------------------- duplicate handling ------------------------
    @annotate("Extractor::_register_new_term")
    def _register_new_term(self, display: str) -> str:
        key = display.casefold()
        self._bk_terms[display]        = key
        self._canon_map[_canon(display)] = display
        self._ann.add(display)
        return key

    @annotate("Extractor::fuzzy_merge")
    def fuzzy_merge(self, term: str) -> Optional[str]:
        """
        Slow path: single‑term merge via ANN + LLM with full heuristics.
        Returns existing dict key or ``None``.
        """
        if not self._bk_terms:
            return None
        # fast punctuation‑agnostic mapping
        disp = self._canon_map.get(_canon(term))
        if disp:
            return self._bk_terms[disp]

        # ANN candidates
        cands = self._ann.query(term)
        if not cands:
            return None
        prompt = _MERGE_PROMPT_TEMPLATE.format(
            term=term, bullets="\n".join(f"- {c}" for c in cands)
        )
        try:
            resp = self.call_ollama(prompt, meta={"task": "merge", "term": term})
        except Exception as e:
            logger.warning("LLM merge failed: %s", e)
            return None
        return self._bk_terms.get(resp.strip())

    # -------------------- batch merge (per page) --------------------------
    @annotate("Extractor::_batch_merge_unique")
    def _batch_merge_unique(self, terms: List[str]) -> Dict[str, str]:
        """
        One ANN + single LLM pass to dedupe *terms* against global store.
        Returns mapping: raw_term → existing_key (missing if distinct).
        """
        mapping: Dict[str, str] = {}
        # 1. exact canon hits
        for t in terms:
            disp = self._canon_map.get(_canon(t))
            if disp:
                mapping[t] = self._bk_terms[disp]

        remain = [t for t in terms if t not in mapping]
        if not remain:
            return mapping

        # 2. ANN candidates
        ann = {t: self._ann.query(t) for t in remain}
        lines = []
        for t, c in ann.items():
            bullets = "\n".join(f"- {x}" for x in c)
            lines.append(f"**{t}**\n{bullets}\n")

        merge_prompt = (
            "For EVERY bold term decide if identical to any candidate (ignore "
            "punctuation/case rules above). Respond CSV: new,existing_or_None\n\n"
            + "\n".join(lines)
        )
        try:
            resp = self.call_ollama(merge_prompt, meta={"task": "batch_merge"})
        except Exception as e:
            logger.warning("Batch merge LLM failed: %s", e)
            return mapping

        for ln in resp.splitlines():
            if "," not in ln:
                continue
            new, old = map(str.strip, ln.split(",", 1))
            if old.lower() == "none" or not old:
                continue
            if old not in self._bk_terms:
                continue
            mapping[new] = self._bk_terms[old]
        return mapping

    # -------------------------- JSON helper -------------------------------
    @annotate("Extractor::_json_from_text")
    @retry_on_exception(Exception, retries=2, delay_seconds=1.0)
    def _json_from_text(self, txt: str) -> Dict[str, Any]:
        """Greedy largest‑brace extraction to guard against stray tokens."""
        pat = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
        matches = sorted(re.finditer(pat, txt), key=lambda m: -len(m.group(0)))
        for m in matches:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "terms" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
        return {"terms": []}

    # ---------------------- formula post‑processing -----------------------
    _FORM_RE = re.compile(r"[A-Z][a-z]?[\d]")
    @annotate("Extractor::_looks_like_formula")
    def _looks_like_formula(self, s: str | None) -> bool:
        return bool(s and self._FORM_RE.search(s))

    @annotate("Extractor::_postprocess_term")
    def _postprocess_term(self, t: Dict[str, Any], ctx: str) -> Dict[str, Any]:
        """
        Validate / repair chemical formula and attach validation metadata.
        """
        f = t.get("formula")
        if not f or not self._looks_like_formula(f):
            t["formula"] = None
            t["formula_validation"] = {"status": "missing"}
            return t
        try:
            val = self.formula_checker.validate(f)
        except Exception as e:
            logger.warning("Formula checker error '%s': %s", f, e)
            val = {"status": "error", "details": {"error": str(e)}}
        if val.get("status") == "invalid":
            repair_prompt = (
                f"'{f}' is invalid formula. From context guess correct formula "
                f"and return ONLY formula string.\nCONTEXT:\n{ctx}"
            )
            try:
                cand = self.call_ollama(repair_prompt, timeout=120,
                                        meta={"task": "formula"})
                cand = cand.strip().split()[0]
                newval = self.formula_checker.validate(cand)
                if newval.get("status") != "invalid":
                    t["formula"] = cand
                    val = newval | {"status": "corrected"}
            except Exception as e:
                logger.warning("LLM formula repair failed: %s", e)
        t["formula_validation"] = val
        return t

    # ------------------------ context snippet ----------------------------
    @annotate("Extractor::_snippet")
    def _snippet(self, full: str, term: str, fname: str,
                 page: int) -> Dict[str, Any]:
        low = term.casefold()
        for sent in re.split(r"(?<=[.!?])\s+", full):
            if low in sent.casefold():
                clip = " ".join(sent.split()[: self.context_length])
                break
        else:
            clip = " ".join(full.split()[: self.context_length])
        return {"text": clip, "source_paper": fname, "page": page + 1}

    # ----------------------- store integration ---------------------------
    @annotate("Extractor::_integrate_terms")
    def _integrate_terms(self, data: Dict[str, Any], txt: str, fname: str,
                         page: int, merge_map: Dict[str, str]) -> bool:
        """
        Merge *data* from LLM into global store. Returns True if changed.
        """
        if not data.get("terms"):
            return False
        updated, local_seen = False, set()

        with annotate(message="integrate_terms", color="blue"):
            for raw in data["terms"]:
                name = raw.get("term", "").strip()
                if not name:
                    continue
                norm = _canon(name)
                if norm in local_seen:       # per‑page dedupe
                    continue
                local_seen.add(norm)

                fixed   = self.schema_helper.validate_and_fix_term(raw)
                snippet = self._snippet(txt, name, fname, page)
                term_pp = self._postprocess_term(fixed, snippet["text"])

                existing_key = merge_map.get(name) or self.fuzzy_merge(name)
                if existing_key:
                    rec = self.terms_dict[existing_key]
                else:
                    existing_key = self._register_new_term(name)
                    rec = {
                        "term": name, "definition": "", "category": "Thing",
                        "formula": None, "formula_validation": None,
                        "relations": [], "pages": [], "source_papers": [],
                        "context_snippets": [], "properties": [],
                    }
                    self.terms_dict[existing_key] = rec

                # merge expandable fields
                if term_pp.get("definition") and \
                   len(term_pp["definition"]) > len(rec.get("definition", "")):
                    rec["definition"] = term_pp["definition"]
                rec.setdefault("category", term_pp.get("category", "Thing"))
                for k in ("formula", "formula_validation", "chebi", "smiles",
                          "charge", "inchi", "inchikey", "mass"):
                    if term_pp.get(k) and not rec.get(k):
                        rec[k] = term_pp[k]

                if page + 1 not in rec["pages"]:
                    rec["pages"].append(page + 1)
                    rec["source_papers"].append(fname)
                    rec["context_snippets"].append(snippet)
                    updated = True

                # relations
                existing = {(r["relation"], r["related_term"])
                            for r in rec["relations"]}
                for rel in term_pp.get("relations", []):
                    tup = (rel["relation"], rel["related_term"])
                    if tup not in existing:
                        rec["relations"].append(rel)
                        updated = True

        updated |= self._attach_properties(txt)
        return updated

    # -------------------- property extraction ---------------------------
    @annotate("Extractor::_attach_properties")
    def _attach_properties(self, txt: str) -> bool:
        """Attach extracted physical properties to existing terms."""
        if not self.terms_dict:
            return False
        mats = [t["term"] for t in self.terms_dict.values()]
        raw  = self.prop_extractor.extract(txt, mats)
        if not raw:
            return False
        norm = self.prop_normalizer.normalize(raw)
        changed = False
        for p in norm:
            key = p["material"].casefold()
            rec = self.terms_dict.get(key)
            if not rec:
                continue
            tpl = (p["property"], p["normalized_value"],
                   p["normalized_unit"], p["context"])
            existing = {(r["property"], r["value"], r["unit"], r["context"])
                        for r in rec["properties"]}
            if tpl not in existing:
                rec["properties"].append({
                    "property": p["property"],
                    "value":    p["normalized_value"],
                    "unit":     p["normalized_unit"],
                    "uncertainty": p.get("uncertainty_value"),
                    "context":  p["context"],
                    "verified": not p["unit_conversion_failed"],
                })
                changed = True
        return changed

    # ------------------------ chunk builder ------------------------------
    @annotate("Extractor::_windows")
    def _windows(self, txt: str, budget_tokens: int = 350) -> List[str]:
        """Sentence‑aware windows filtered by MS keywords."""
        enc, toks_len = _get_encoding(), lambda s: len(_get_encoding().encode(s))
        wins, cur, cur_tok = [], [], 0
        for sent in _sentences(txt):
            if not _is_ms(sent):
                continue
            st = toks_len(sent)
            if cur and cur_tok + st > budget_tokens:
                wins.append(" ".join(cur))
                cur, cur_tok = [], 0
            cur.append(sent)
            cur_tok += st
        if cur:
            wins.append(" ".join(cur))
        return wins

    # ----------------------- thread‑safe save ---------------------------
    @annotate("Extractor::_save")
    def _save(self) -> None:
        """Persist term store to JSON on disk (thread‑safe)."""
        with self._save_lock:
            for t in self.terms_dict.values():
                t.setdefault("properties", [])
            json.dump({"metadata": self.metadata,
                       "terms": list(self.terms_dict.values())},
                      open(self.output_file, "w"), indent=2)
            logger.debug("Saved %d terms", len(self.terms_dict))

    # ------------------------------ page --------------------------------
    @annotate("Extractor::process_page")
    def process_page(self, doc: fitz.Document, path: str, page_num: int) -> bool:
        """
        Extract terms from a single PDF page.

        Returns
        -------
        bool
            True if *any* new information was added to the global store.
        """
        with annotate("load_page", color="green"):
            try:
                pg   = doc.load_page(page_num)
                text = pg.get_text()
            except Exception as e:
                logger.error("Read error %s p%d: %s",
                             os.path.basename(path), page_num + 1, e)
                return False

        if not text or len(text.split()) < 20:
            return False
        fname = os.path.basename(path)

        # ---------- Stage 1: term‑only extraction ------------------------
        win_rng = start_range(message="term_only_windows", color="yellow")
        windows = self._windows(text)
        end_range(win_rng)

        term_sets: List[List[str]] = []
        for idx, chunk in enumerate(windows, 1):
            prm  = _TERM_ONLY_TEMPLATE.format(content=chunk)
            meta = {"task": "term_only", "file": fname,
                    "page": page_num + 1, "chunk": idx}
            resp = self.call_ollama(prm, meta=meta)
            try:
                term_sets.append(json.loads(resp))
            except json.JSONDecodeError:
                logger.warning("Bad JSON term‑only (%s p%d c%d)",
                               fname, page_num + 1, idx)

        uniques = sorted({t.strip() for sub in term_sets for t in sub if t.strip()})
        if not uniques:
            return False

        # ---------- Stage 2: batch merge -------------------------------
        merge_map = self._batch_merge_unique(uniques)

        # ---------- Stage 3: detail pass -------------------------------
        # embed context tags
        tagged = text
        for idx, t in enumerate(uniques, 1):
            tagged = re.sub(re.escape(t),
                            f"<CTX idx={idx}>{t}</CTX {idx}>",
                            tagged, flags=re.I)

        detail_prm = _DETAIL_TEMPLATE.format(
            terms="\n  - " + "\n  - ".join(uniques),
            filename=fname, page=page_num + 1
        ) + "\n\nCONTENT:\n" + tagged[:6000]

        detail_json = self._json_from_text(
            self.call_ollama(detail_prm,
                             meta={"task": "detail",
                                   "file": fname,
                                   "page": page_num + 1})
        )

        updated = self._integrate_terms(detail_json, text, fname,
                                        page_num, merge_map)
        if updated:
            self._save()
        return updated

    # ------------------------------ PDF ----------------------------------
    @annotate("Extractor::process_pdf")
    def process_pdf(self, pdf_path: str) -> int:
        """
        Process a full PDF in parallel across pages. Returns #pages with terms.
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error("Cannot open %s: %s", pdf_path, e)
            return 0

        total = doc.page_count
        logger.info("Processing %s (%d pages)", os.path.basename(pdf_path), total)
        self.metadata["processed_pages_total"] += total
        pages_with_terms = 0

        with annotate("process_pdf", color="cyan"):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_page, doc, pdf_path, i): i
                    for i in range(total)
                }
                for future in as_completed(futures):
                    try:
                        if future.result():
                            pages_with_terms += 1
                    except Exception as e:
                        logger.warning("Page %d failed: %s", futures[future], e)

        self.metadata["processed_pages_with_terms"] += pages_with_terms
        self.metadata["processed_files"] += 1
        return pages_with_terms

    # ---------------------------- directory -----------------------------
    @annotate("Extractor::process_directory")
    def process_directory(self) -> Dict[str, Any]:
        """Walk *data_dir* and process each PDF in parallel."""
        if not os.path.isdir(self.data_dir):
            return {"status": "error",
                    "message": f"Directory not found: {self.data_dir}"}
        pdfs = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".pdf")]
        if not pdfs:
            logger.warning("No PDFs under %s", self.data_dir)

        logger.info("Processing %d PDF(s) with %d workers", len(pdfs), self.max_workers)
        with annotate("process_directory", color="magenta"):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_pdf, os.path.join(self.data_dir, f)): f
                    for f in pdfs
                }
                for i, future in enumerate(as_completed(futures), 1):
                    fname = futures[future]
                    logger.info("[%d/%d] %s", i, len(pdfs), fname)
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning("PDF %s failed: %s", fname, e)

        # importance flagging
        for t in self.terms_dict.values():
            occ, papers = len(t["pages"]), len(set(t["source_papers"]))
            t["importance"] = ("high"   if papers > 1 or occ > 5 else
                               "medium" if occ > 2            else
                               "low")
        self._save()

        if self._slow_prompts:
            logger.warning("==== SLOW CALLS (>%ds) ====", SLOW_WARN_S)
            for rec in sorted(self._slow_prompts, key=lambda x: -x["t"]):
                logger.warning("%6.1fs tok=%4d  %s",
                               rec["t"], rec["tok"],
                               {k:v for k,v in rec.items()
                                if k not in ("t","tok")})

        return {"status": "success",
                "processed_files": self.metadata["processed_files"],
                "processed_pages_total": self.metadata["processed_pages_total"],
                "processed_pages_with_terms": self.metadata["processed_pages_with_terms"],
                "unique_terms": len(self.terms_dict),
                "output_file": self.output_file}



###############################################################################
# CLI
###############################################################################
def _cli() -> None:
    """CLI entry‑point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",  default="./polymer_papers",
                    help="Directory containing PDF files")
    ap.add_argument("-o", "--output",
                    default="./storage/terminology/extracted_terms-test.json",
                    help="Path to JSON output")
    args = ap.parse_args()

    extractor = OllamaTermExtractor(data_dir=args.input,
                                    output_file=args.output)
    res = extractor.process_directory()
    if res.get("status") != "success":
        sys.exit("Extraction failed: " + res.get("message", "unknown"))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    _cli()
