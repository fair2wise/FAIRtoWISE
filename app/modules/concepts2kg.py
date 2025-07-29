#!/usr/bin/env python3
"""
concepts2kg.py

Convert extracted_terms.json → MatKG graph.json
with optional **LLM-based relation validation/enrichment** at scale.

Usage
-----
python concepts2kg.py extracted_terms.json mat_kg.json \
    --validate-relations \
    --schema matkg_schema.yaml \
    --max-workers 16 \
    --ctx-chars 300 \
    --topk 8 \
    --autosave-every 100 \
    --dump-bad bad_rel_cases \
    --very-verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from nvtx import annotate

# ---------------------------------------------------------------------------
# Color logger (ported/trimmed from extractor)
# ---------------------------------------------------------------------------
class _AnsiFormatter(logging.Formatter):
    _C = {"DEBUG":36,"INFO":32,"WARNING":33,"ERROR":31,"CRITICAL":41}
    _R = "\033[0m"
    def format(self, record):  # noqa:D401
        color = f"\033[{self._C.get(record.levelname,32)}m"
        record.levelname = f"{color}{record.levelname}{self._R}"
        record.msg = f"{color}{record.getMessage()}{self._R}"
        record.args = ()
        return super().format(record)

_log_hdl = logging.StreamHandler(sys.stdout)
_log_hdl.setFormatter(_AnsiFormatter("%(asctime)s [%(levelname)s] %(message)s",
                                     "%H:%M:%S"))
logger = logging.getLogger("MatKG_json2kg")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(_log_hdl)

# ---------------------------------------------------------------------------
# Lightweight token counter (best-effort; falls back if tiktoken missing)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _ntok(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:  # pragma: no cover
    def _ntok(s: str) -> int:
        return max(1, len(s)//4)

# ---------------------------------------------------------------------------
# Embedding + ANN (borrowed pattern)
# ---------------------------------------------------------------------------
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import faiss
except Exception as e:  # pragma: no cover
    logger.error("FAISS import failed: %s", e)
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    logger.error("sentence_transformers import failed: %s", e)
    raise

_EMB_LOCK = threading.Lock()
_EMB_MODEL = None

@annotate("_get_emb_model")
def _get_emb_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMB_MODEL
    with _EMB_LOCK:
        if _EMB_MODEL is None:
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            _EMB_MODEL = SentenceTransformer(model_name, device=device)
        return _EMB_MODEL

@annotate("_embed")
def _embed(texts: Sequence[str]) -> np.ndarray:
    """Return L2-normalized float32 embeddings; safe GPU→CPU fallback."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    try:
        m = _get_emb_model()
        arr = m.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
        return arr.astype("float32", copy=False)
    except Exception as e:  # GPU oom / init failure fallback
        logger.warning("Embed error (%s) → falling back to CPU.", e)
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        arr = m.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
        return arr.astype("float32", copy=False)


class _ANN:
    """Wrapper around FAISS inner-product ANN."""
    @annotate("_ANN::__init__")
    def __init__(self, dim: int):
        base = faiss.IndexFlatIP(dim)
        if torch and torch.cuda.is_available():  # try GPU shards
            try:
                self.index = faiss.index_cpu_to_all_gpus(base)
            except Exception:  # pragma: no cover
                self.index = base
        else:
            self.index = base
        self.display: List[str] = []
        self._lock = threading.Lock()

    @annotate("_ANN::add_many")
    def add_many(self, vecs: np.ndarray, labels: Sequence[str]):
        with self._lock:
            self.index.add(vecs)
            self.display.extend(labels)

    @annotate("_ANN::query")
    def query(self, txt: str, k: int = 8) -> List[str]:
        if not self.display:
            return []
        with self._lock:
            q = _embed([txt])
            _, idx = self.index.search(q, min(k, len(self.display)))
        return [self.display[i] for i in idx[0] if i >= 0]

# ---------------------------------------------------------------------------
# Schema helper (minimal; we only need predicate names)
# ---------------------------------------------------------------------------
from linkml_runtime.utils.schemaview import SchemaView

class _SchemaHelper:
    @annotate("_SchemaHelper::__init__")
    def __init__(self, schema_path: str):
        self.schema_view = SchemaView(schema_path)
        self.allowed_preds = sorted(
            cls.name for cls in self.schema_view.all_classes().values()
            if cls.is_a == "Association"
        )
    @annotate("_SchemaHelper::allowed_predicates_str")
    def allowed_predicates_str(self) -> str:
        return ", ".join(self.allowed_preds)

# ---------------------------------------------------------------------------
# Ollama call (no streaming; structured; meta unused but kept for parity)
# ---------------------------------------------------------------------------
import requests

@annotate("default_call_fn")
def default_call_fn(prompt: str, meta: dict | None = None,
                    *, model: str = "mistral-small3.1-32k",
                    system: str = "You are a schema-aware scientific assistant. "
                                  "Only use schema-defined relations. "
                                  "Return ONLY JSON.") -> str:
    pid = hashlib.sha1(prompt.encode()).hexdigest()[:10]
    t0 = time.monotonic()
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                "stream": False,
            },
            timeout=300,
        )
        resp.raise_for_status()
        parsed = resp.json()
        msg = parsed.get("message", {}).get("content", "")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed ({pid}): {e}") from e
    dt = time.monotonic() - t0
    logger.debug("OLLAMA ✔ %s (%.1fs) %s", pid, dt, meta or {})
    return msg

# ---------------------------------------------------------------------------
# JSON parse helpers
# ---------------------------------------------------------------------------
_JSON_ARRAY_RE = None  # lazy compile


@annotate("_json_greedy_array")
def _json_greedy_array(txt: str) -> List[Any]:
    """
    Greedy-extract the *largest* JSON array substring in txt; fallback [].
    Accepts leading/trailing garbage from LLMs.
    """
    global _JSON_ARRAY_RE
    if _JSON_ARRAY_RE is None:
        import re
        _JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
    m = _JSON_ARRAY_RE.search(txt)
    if not m:
        return []
    frag = m.group(0)
    try:
        out = json.loads(frag)
        return out if isinstance(out, list) else []
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------
@annotate("_diff_relations")
def _diff_relations(orig: List[Dict[str, Any]],
                    new:  List[Dict[str, Any]]) -> Tuple[int,int,int]:
    """Return (#added, #removed, #kept)."""
    orig_set = {(r.get("relation"), r.get("related_term")) for r in orig}
    new_set  = {(r.get("relation"), r.get("related_term")) for r in new}
    added   = len(new_set - orig_set)
    removed = len(orig_set - new_set)
    kept    = len(orig_set & new_set)
    return added, removed, kept

# ---------------------------------------------------------------------------
# SmartRelationLLMValidator
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

class SmartRelationLLMValidator:
    """
    Scalable relation enrichment/repair.

    Workflow per term
    -----------------
    1. ANN-based candidate harvesting (topK by embedding distance).
    2. Prompt with (term minimal summary) + truncated context + numbered candidate list + allowed preds.
    3. Parse JSON array of {relation, related_term, verified}.
    4. Diff vs original relations → stats.
    """
    @annotate("SmartRelationLLMValidator::__init__")
    def __init__(self,
                 schema_path: str,
                 call_fn: Callable[[str, dict], str] = default_call_fn,
                 *,
                 ctx_chars: int = 300,
                 topk: int = 8,
                 max_workers: int = 8,
                 very_verbose: bool = False,
                 dump_bad_dir: Optional[Path] = None):
        self.schema = _SchemaHelper(schema_path)
        self.call_fn = call_fn
        self.ctx_chars = ctx_chars
        self.topk = topk
        self.max_workers = max_workers
        self.very_verbose = very_verbose
        self.dump_bad_dir = Path(dump_bad_dir) if dump_bad_dir else None
        if self.dump_bad_dir:
            self.dump_bad_dir.mkdir(parents=True, exist_ok=True)

        self.ann: Optional[_ANN] = None
        self._ann_ready = False
        self._cand_terms: List[str] = []
        self._cand_ctx: Dict[str,str] = {}

    # ------------------------------ ANN prep ---------------------------
    @annotate("SmartRelationLLMValidator::init_ann")
    def init_ann(self, terms: Iterable[Dict[str, Any]]) -> None:
        if self._ann_ready:
            return
        names: List[str] = []
        for t in terms:
            nm = t.get("term") or t.get("name")
            if not nm:
                continue
            names.append(nm)
            # stash minimal context for candidate description scoring fallback
            self._cand_ctx[nm] = (t.get("definition") or "")[:200]
        if not names:
            logger.warning("No term names; skipping ANN init.")
            self._ann_ready = True
            return

        logger.info("📦 Embedding %d term names ...", len(names))
        vecs = _embed(names)
        self.ann = _ANN(vecs.shape[1])
        self.ann.add_many(vecs, names)
        self._cand_terms = names
        self._ann_ready = True
        logger.info("✅ Candidate ANN ready (%d)", len(names))

    # ------------------------------ candidate ------------------------
    @annotate("SmartRelationLLMValidator::_topk")
    def _topk(self, query: str) -> List[str]:
        if not self._ann_ready or not self.ann:
            return []
        return self.ann.query(query, k=self.topk)

    # ------------------------------ prompt ---------------------------
    @annotate("SmartRelationLLMValidator::_make_prompt")
    def _make_prompt(self,
                     term_name: str,
                     term_category: str,
                     term_def: str,
                     ctx: str,
                     candidates: List[str],
                     orig_rels: List[Dict[str, Any]]) -> str:
        """
        Ultra-compact prompt.
        """
        # compress orig relations inline
        rel_lines = []
        for r in orig_rels:
            rel_lines.append(f"- {r.get('relation','?')} -> {r.get('related_term','?')}")
        rel_str = "\n".join(rel_lines) if rel_lines else "(none)"

        cand_lines = []
        for i, c in enumerate(candidates, 1):
            # include short context snip from candidate if we cached
            snip = self._cand_ctx.get(c, "")
            if snip:
                snip = snip.replace("\n"," ")[:80]
                cand_lines.append(f"{i}. {c} :: {snip}")
            else:
                cand_lines.append(f"{i}. {c}")
        cand_str = "\n".join(cand_lines)

        prompt = (
f"TERM: {term_name}\n"
f"CATEGORY: {term_category}\n"
f"DEF: {term_def[:200]}\n"
f"CTX: {ctx[:self.ctx_chars]}\n\n"
f"ORIG_RELATIONS:\n{rel_str}\n\n"
f"CANDIDATE_TERMS (choose any #):\n{cand_str}\n\n"
f"ALLOWED_PREDICATES: {self.schema.allowed_predicates_str()}\n\n"
"Return ONLY a JSON array of objects like:\n"
'[{"relation":"MaterialHasProperty","related_term":"band gap","verified":true}, ...]\n'
"If no valid relations, return []."
        )
        return prompt

    # ------------------------------ enrich one -----------------------
    @annotate("SmartRelationLLMValidator::enrich_one")
    def enrich_one(self, term: Dict[str, Any], ctx: str) -> Tuple[List[Dict[str, Any]], Dict[str,int], str]:
        name = term.get("term") or term.get("name") or "UNKNOWN"
        category = term.get("category","Unknown")
        definition = term.get("definition","") or ""
        orig_rels = term.get("relations") or []

        # Compose query string for ANN candidate harvesting
        ann_query = f"{name} {definition} {ctx}"
        candidates = [c for c in self._topk(ann_query) if c != name]

        prompt = self._make_prompt(name, category, definition, ctx, candidates, orig_rels)
        toks = _ntok(prompt)
        if toks > 6000:  # safety clip
            prompt = prompt[:24000]  # char clip as final guard

        try:
            raw = self.call_fn(prompt, meta={"task":"relation_enrich","term":name})
        except Exception as e:
            logger.warning("Ollama error %s: %s", name, e)
            return orig_rels, {"added":0,"removed":0,"kept":len(orig_rels),"error":1}, "http"

        parsed = _json_greedy_array(raw)
        if not parsed:
            if self.dump_bad_dir:
                (self.dump_bad_dir / f"{name}.txt").write_text(raw)
            logger.warning("Relation enrich fail %s: parse error", name)
            return orig_rels, {"added":0,"removed":0,"kept":len(orig_rels),"error":1}, "parse"

        # sanitize & filter
        clean: List[Dict[str, Any]] = []
        for r in parsed:
            if not isinstance(r, dict):
                continue
            rel = r.get("relation")
            rel_term = r.get("related_term")
            if not rel or not rel_term:
                continue
            clean.append({
                "relation": rel,
                "related_term": rel_term,
                "verified": bool(r.get("verified", False))
            })

        if not clean:
            return [], {"added":0,"removed":len(orig_rels),"kept":0,"error":0}, "ok"

        a,r,k = _diff_relations(orig_rels, clean)
        return clean, {"added":a,"removed":r,"kept":k,"error":0}, "ok"

    # ------------------------------ bulk enrich ----------------------
    @annotate("SmartRelationLLMValidator::enrich_many")
    def enrich_many(self, terms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parallel enrich; returns term_name -> relation list.
        Emits progress logs.
        """
        self.init_ann(terms)

        # Pre-build context strings (joined snippet text)
        ctx_map: Dict[str,str] = {}
        for t in terms:
            nm = t.get("term") or t.get("name") or "UNKNOWN"
            ctx_list = t.get("context_snippets") or []
            ctx = " ".join(s.get("text","")[:100] for s in ctx_list if isinstance(s, dict))[:self.ctx_chars]
            ctx_map[nm] = ctx

        # Submit tasks
        total = len(terms)
        logger.info("🔍 LLM relation enrichment … (%d terms)", total)
        results: Dict[str, List[Dict[str, Any]]] = {}
        stats_tot = {"added":0,"removed":0,"kept":0,"error":0}
        done = 0
        lock = threading.Lock()

        def _work(t: Dict[str, Any]) -> Tuple[str, List[Dict[str,Any]], Dict[str,int], str]:
            nm = t.get("term") or t.get("name") or "UNKNOWN"
            rels, stats, status = self.enrich_one(t, ctx_map[nm])
            return nm, rels, stats, status

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(_work, t): t for t in terms}
            for f in as_completed(futs):
                nm, rels, stats, status = f.result()
                results[nm] = rels
                with lock:
                    done += 1
                    for k,v in stats.items():
                        stats_tot[k] += v
                if self.very_verbose:
                    logger.info("   [%d/%d] %s  +%d  -%d  =%d  %s",
                                done, total, nm, stats["added"], stats["removed"],
                                stats["kept"], "ERR" if stats["error"] else "")
                else:
                    # lightweight heartbeat every 50
                    if done % 50 == 0 or status != "ok":
                        logger.info("   … %d/%d done (+%d -%d err=%d)",
                                    done, total, stats_tot["added"],
                                    stats_tot["removed"], stats_tot["error"])
        logger.info("✅ Enrichment complete: +%d -%d kept=%d err=%d",
                    stats_tot["added"], stats_tot["removed"],
                    stats_tot["kept"], stats_tot["error"])
        return results

# ---------------------------------------------------------------------------
# ID + list helpers
# ---------------------------------------------------------------------------
@annotate("make_id")
def make_id(term: str) -> str:
    return "matkg:" + "".join(c for c in term if c.isalnum() or c == "-")

@annotate("ensure_list")
def ensure_list(val: Any) -> List[Any]:
    return val if isinstance(val, list) else [val] if val else []

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
@annotate("build_graph")
def build_graph(raw_terms: List[Dict[str, Any]],
                *,
                validate_relations: bool = False,
                schema_path: str = "matkg_schema.yaml",
                ctx_chars: int = 300,
                topk: int = 8,
                max_workers: int = 8,
                very_verbose: bool = False,
                autosave_every: int = 0,
                dump_bad_dir: Optional[Path] = None,
                autosave_path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:

    logger.info("⚙️  Build graph (validate_relations=%s)", validate_relations)

    validator = None
    rel_map: Dict[str, List[Dict[str, Any]]] = {}
    if validate_relations:
        validator = SmartRelationLLMValidator(
            schema_path,
            call_fn=default_call_fn,
            ctx_chars=ctx_chars,
            topk=topk,
            max_workers=max_workers,
            very_verbose=very_verbose,
            dump_bad_dir=dump_bad_dir,
        )
        rel_map = validator.enrich_many(raw_terms)

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen: Set[Tuple[str,str,str]] = set()

    def _autosave_if_needed(n_done: int):
        if not autosave_every or not autosave_path:
            return
        if n_done % autosave_every == 0:
            partial = {"things": list(nodes.values()), "associations": edges}
            autosave_path.write_text(json.dumps(partial, indent=2))
            logger.info("💾 Autosaved partial KG at %d terms → %s", n_done, autosave_path)

    for i, term in enumerate(raw_terms, 1):
        name = term.get("term") or term.get("name") or "UNKNOWN"
        tid = make_id(name)

        if tid not in nodes:
            nodes[tid] = {
                "id": tid,
                "name": name,
                "category": term.get("category", "Unknown"),
                "description": term.get("definition", "") or "N/A",
                "pages": ensure_list(term.get("pages")),
                "source_papers": ensure_list(term.get("source_papers")),
                "context_snippets": ensure_list(term.get("context_snippets")),
                "formula": term.get("formula", "") or "",
                "formula_validation": term.get("formula_validation", {}) or {},
                "properties": ensure_list(term.get("properties")),
            }

        rels = rel_map.get(name) if validator else ensure_list(term.get("relations"))
        for rel in rels:
            tgt = rel.get("related_term")
            if not tgt:
                continue
            rid = make_id(tgt)
            if rid not in nodes:
                nodes[rid] = {
                    "id": rid,
                    "name": tgt,
                    "category": "Unknown",
                    "description": "",
                    "pages": [],
                    "source_papers": [],
                    "context_snippets": [],
                    "formula": "",
                    "formula_validation": {},
                    "properties": [],
                }
            pred = "rel:" + rel.get("relation", "RELATED_TO")
            sig = (tid, pred, rid)
            if sig in seen:
                continue
            seen.add(sig)
            edges.append({
                "subject": tid,
                "predicate": pred,
                "object": rid,
                "has_evidence": (
    "; ".join(ensure_list(rel.get("evidence")))
    if rel.get("evidence")
    else (
        " || ".join(
            f"{s.get('source_paper', 'unknown')} pg {s.get('page','?')}: {s.get('text','').replace(chr(10),' ').strip()}"
            for s in ensure_list(term.get("context_snippets"))
        ) or None
    )
),
            })

        if very_verbose and validate_relations:
            logger.debug("   Node %s now has %d relations.", name, len(rels))
        _autosave_if_needed(i)

    return {"things": list(nodes.values()), "associations": edges}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert extracted_terms → MatKG with optional relation validation.")
    p.add_argument("input_json", type=Path)
    p.add_argument("output_json", type=Path)
    p.add_argument("--validate-relations", action="store_true",
                   help="Run LLM-based relation validation/enrichment.")
    p.add_argument("--schema", default="matkg_schema.yaml")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--ctx-chars", type=int, default=300,
                   help="Context character budget per term prompt.")
    p.add_argument("--topk", type=int, default=8,
                   help="Top-K candidate terms for LLM prompt.")
    p.add_argument("--autosave-every", type=int, default=10,
                   help="Autosave partial KG every N terms (0=off).")
    p.add_argument("--autosave-path", type=Path, default="storage/kg/autosave.json",
                   help="Path for autosave file (defaults to output + '.autosave.json').")
    p.add_argument("--dump-bad", type=Path, default=None,
                   help="Directory to dump bad LLM JSON responses.")
    p.add_argument("--very-verbose", action="store_true",
                   help="Log per-term enrichment diff details.")
    p.add_argument("--verbose", action="store_true",
                   help="Set log level DEBUG (overrides).")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    if args.verbose or args.very_verbose:
        logger.setLevel(logging.DEBUG)

    autosave_path = args.autosave_path or Path(str(args.output_json) + ".autosave.json")

    try:
        data_txt = args.input_json.read_text()
        data = json.loads(data_txt)
        terms = data["terms"] if isinstance(data, dict) and "terms" in data else data
    except Exception as e:
        logger.error("Failed to read input: %s", e)
        sys.exit(1)

    graph = build_graph(
        terms,
        validate_relations=args.validate_relations,
        schema_path=args.schema,
        ctx_chars=args.ctx_chars,
        topk=args.topk,
        max_workers=args.max_workers,
        very_verbose=args.very_verbose,
        autosave_every=args.autosave_every,
        dump_bad_dir=args.dump_bad,
        autosave_path=autosave_path,
    )

    try:
        args.output_json.write_text(json.dumps(graph, indent=2))
    except Exception as e:
        logger.error("Write failed: %s", e)
        sys.exit(1)

    logger.info("Wrote %d nodes and %d edges → %s",
                len(graph["things"]), len(graph["associations"]), args.output_json)

if __name__ == "__main__":
    main()
