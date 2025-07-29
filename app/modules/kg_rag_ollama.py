#!/usr/bin/env python
"""
kg_rag_ollama.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp
import faiss  # type: ignore
import fitz  # PyMuPDF
import numpy as np
import torch
from colorama import Fore, Style, init as colorama_init
from nvtx import annotate
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# ───────────────────── optional noun‑phrase extraction ─────────────────────
try:
    import nltk
    from nltk import word_tokenize, pos_tag
    from nltk.chunk import RegexpParser

    _NLTK_OK = True
    for corp in ("punkt", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"tokenizers/{corp}")
        except LookupError:
            nltk.download(corp, quiet=True)
except Exception:
    _NLTK_OK = False

# ───────────────────── configuration ─────────────────────
OLLAMA_MODEL = os.environ.get("KG_RAG_OLLAMA_MODEL", "gemma3:27b")
OLLAMA_API_URL = os.environ.get("KG_RAG_OLLAMA_URL", "http://localhost:11434/api/chat")
GRAPH_FILE = os.environ.get(
    "KG_RAG_GRAPH",
    "/pscratch/sd/d/dabramov/fair2wise/storage/kg/mat_kg_jul18.json",
)
PDF_DIR = os.environ.get("KG_RAG_PDF_DIR", "polymer_papers")

DEFAULT_K = int(os.environ.get("KG_RAG_TOPK", "12"))
EMBED_MODEL = os.environ.get("KG_RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
USER_BATCH_OVERRIDE: Optional[str] = os.environ.get("KG_RAG_BATCH")

PDF_SNIPPET_LEN = int(os.environ.get("KG_RAG_SNIP", "1_000"))
CONTEXT_CHAR_BUDGET = int(os.environ.get("KG_RAG_CTX_CHARS", "16_000"))
CTX_SOFT_LIMIT = int(CONTEXT_CHAR_BUDGET * 0.75)

FORCE_CPU = bool(os.environ.get("KG_RAG_FORCE_CPU"))
MAX_TEXT_CHARS = int(os.environ.get("KG_RAG_MAX_TEXT_CHARS", "1024"))

#  Retrieval & ranking
ENABLE_BFS = bool(int(os.environ.get("KG_RAG_ENABLE_BFS", "1")))
BFS_SEED_TOPK = int(os.environ.get("KG_RAG_BFS_TOPK", str(DEFAULT_K * 2)))
MAX_BFS_HOPS = int(os.environ.get("KG_RAG_MAX_HOPS", "1"))

STEPWISE = bool(int(os.environ.get("KG_RAG_STEPWISE", "1")))
STEPWISE_MAX_STEPS = int(os.environ.get("KG_RAG_STEPWISE_MAX_STEPS", "6"))

PRP_W_SEM, PRP_W_DEPTH, PRP_W_LEX, PRP_W_EVID = (
    0.8,
    0.6,
    0.6,
    0.3,
)  # retuned

GENERIC_PENALTY = float(os.environ.get("KG_RAG_GENERIC_PENALTY", "0.8"))
CTX_VOLUME_TRIPLES = int(os.environ.get("KG_RAG_CONTEXT_VOLUME", "150"))
STRUCT_CTX = bool(int(os.environ.get("KG_RAG_STRUCT_CTX", "1")))

#  Misc
DEBUG = bool(int(os.environ.get("KG_RAG_DEBUG", "0")))
MAX_PDF_CACHE = int(os.environ.get("KG_RAG_PDF_CACHE", "256"))

# ───────────────────── logging ─────────────────────
class _Fmt(logging.Formatter):
    @annotate('_Fmt::format')
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        t = Fore.CYAN + self.formatTime(record) + Style.RESET_ALL
        return f"{t} {Fore.MAGENTA}{record.levelname}{Style.RESET_ALL}: {record.getMessage()}"


_hdl = logging.StreamHandler(sys.stdout)
_hdl.setFormatter(_Fmt())
logger = logging.getLogger("kg_rag_ollama")
logger.addHandler(_hdl)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

#  Terminal colours
colorama_init(autoreset=True)

# ───────────────────── helpers ─────────────────────
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
GENERIC_PAT = re.compile(r"(generic|material|property|parameter|technique|process|device)s?$", re.I)


@annotate('_strip_ansi')
def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


@annotate('_tokenize')
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


@annotate('_noun_phrases')
def _noun_phrases(text: str) -> List[str]:
    if not _NLTK_OK:
        return [text] if text.strip() else []
    toks = word_tokenize(text)
    tags = pos_tag(toks)
    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    tree = RegexpParser(grammar).parse(tags)
    out: List[str] = []
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        phrase = " ".join(w for w, _ in subtree.leaves()).strip()
        if phrase:
            out.append(phrase)
    return out or [text]


@annotate('extract_query_entities')
def extract_query_entities(q: str) -> List[str]:
    """Return deduplicated noun phrases + ≥3‑char tokens."""
    ents: List[str] = []
    ents.extend([np for np in _noun_phrases(q) if len(np) >= 3])
    ents.extend([tok for tok in _tokenize(q) if len(tok) >= 3])
    seen: set[str] = set()
    uniq: List[str] = []
    for e in ents:
        k = e.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)
    return uniq


@annotate('auto_device')
def auto_device() -> str:
    if FORCE_CPU:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@annotate('cuda_warmup')
def cuda_warmup(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.cuda.init()
            a = torch.empty((4096, 4096), device=device).normal_()
            _ = a @ a.t()
        except Exception as exc:  # pragma: no cover
            logger.warning("CUDA warm‑up failed: %s", exc)


@annotate('snippet_text')
def snippet_text(txt: str, length: int, hints: Sequence[str] | None) -> str:
    if not txt or length <= 0:
        return ""
    if len(txt) <= length:
        return txt
    if hints:
        low = txt.lower()
        hits = [low.find(h.lower()) for h in hints if low.find(h.lower()) != -1]
        if hits:
            i = min(hits)
            start = max(i - length // 4, 0)
            return txt[start : start + length]
    return txt[:length]


# ───────────────────── dataclasses ─────────────────────
@dataclass(slots=True)
class NodeScore:
    id: str
    score: float
    depth: int = 0


@dataclass(slots=True)
class NodeInfo:
    id: str
    name: str
    category: str
    description: str
    score_sem: float
    score_graph: float
    depth: int
    lexical_overlap: float
    evidence_ct: int

    @property
    @annotate('NodeInfo::score_prp')
    def score_prp(self) -> float:
        depth_fac = 1.0 / (1.0 + self.depth)
        evid = np.tanh(self.evidence_ct / 5.0)
        return (
            PRP_W_SEM * self.score_sem
            + PRP_W_DEPTH * depth_fac * self.score_graph
            + PRP_W_LEX * self.lexical_overlap
            + PRP_W_EVID * evid
        )


# ───────────────────── PDF cache ─────────────────────
from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=MAX_PDF_CACHE)
@annotate('load_pdf_text')
def load_pdf_text(path: str) -> str:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("PDF open failed %s – %s", path, exc)
        return ""
    try:
        txt = "".join(pg.get_text() for pg in doc)
    finally:
        doc.close()
    return txt


# ───────────────────── KnowledgeGraph ─────────────────────
class KnowledgeGraph:
    @annotate('KnowledgeGraph::__init__')
    def __init__(self, graph_file: str, embed_model: str = EMBED_MODEL) -> None:  # noqa: D401
        logger.info(Fore.YELLOW + "Loading KG…" + Style.RESET_ALL)
        with open(graph_file, "r") as fh:
            data = json.load(fh)
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in data["things"]}
        self.out_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in data["associations"]:
            self.out_edges[e["subject"]].append(e)

        self._canon_to_id: Dict[str, str] = {}
        for nid, n in self.nodes.items():
            canon = re.sub(r"[^a-z0-9]", "", n.get("name", "").lower())
            self._canon_to_id.setdefault(canon, nid)

        device = auto_device()
        cuda_warmup(device)
        logger.info("Loading embedding model %s on %s…", embed_model, device)
        self.embed_model = SentenceTransformer(embed_model, device=device)

        try:
            _ = self.embed_model.encode(
                ["_smoke_"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.error("Initial encode failed on %s (%s) – switching to CPU", device, exc)
            self.embed_model = SentenceTransformer(embed_model, device="cpu")
            device = "cpu"

        self.batch_size = int(USER_BATCH_OVERRIDE or (16 if device == "cuda" else 32))
        logger.info("Encode batch size = %d", self.batch_size)

        texts, self.ids = [], []
        for nid, n in self.nodes.items():
            txt = f"{n.get('name','')} {n.get('description','')}".strip()[:MAX_TEXT_CHARS]
            texts.append(txt)
            self.ids.append(nid)

        logger.info("Encoding %d nodes (≤%d chars)…", len(texts), MAX_TEXT_CHARS)
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            try:
                vecs = self.embed_model.encode(
                    chunk,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                if device == "cuda":
                    logger.error("GPU encode failed (%s) → retry CPU…", exc)
                    self.embed_model = SentenceTransformer(embed_model, device="cpu")
                    vecs = self.embed_model.encode(
                        chunk,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                else:
                    raise
            embs.append(vecs)
        self._emb = np.vstack(embs).astype("float32")
        self._build_faiss_index(self._emb)
        self.id_map = np.asarray(self.ids)
        self._cache: Dict[str, List[NodeScore]] = {}
        logger.info(Fore.GREEN + f"KG ready ({len(self.ids)} vectors)." + Style.RESET_ALL)

    #  FAISS index ----------------------------------------------------------
    @annotate('KnowledgeGraph::_build_faiss_index')
    def _build_faiss_index(self, emb: np.ndarray) -> None:  # noqa: D401
        dim, N = emb.shape[1], emb.shape[0]
        nlist = max(64, int(np.sqrt(N) * 2))
        logger.info("Building IVF‑Flat: dim=%d nlist=%d vectors=%d", dim, nlist, N)
        cpu_index = faiss.index_factory(dim, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
        use_gpu = (not FORCE_CPU) and faiss.get_num_gpus() > 0
        if use_gpu:
            res = faiss.StandardGpuResources()
            try:
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                try:
                    self.index.train(emb)
                except faiss.FaissException:
                    self.index = faiss.GpuIndexFlatIP(res, dim)
                self.index.add(emb)
                self.index.nprobe = min(32, nlist // 4)
                return
            except Exception as exc:
                logger.error("GPU FAISS build failed (%s) → CPU.", exc)
        try:
            cpu_index.train(emb)
        except faiss.FaissException:
            cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(emb)
        self.index = cpu_index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(32, nlist // 4)  # type: ignore[attr-defined]

    #  Semantic search ------------------------------------------------------
    @annotate('KnowledgeGraph::_norm')
    def _norm(self, d: np.ndarray) -> np.ndarray:
        return np.clip((d + 1.0) * 0.5, 0.0, 1.0)

    @annotate('KnowledgeGraph::semantic_search')
    def semantic_search(self, q: str, topk: int = DEFAULT_K * 2) -> List[NodeScore]:
        if q in self._cache:
            return self._cache[q]
        q_vec = self.embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        dists, idx = self.index.search(q_vec.astype("float32"), topk)
        hits = [
            NodeScore(self.id_map[i], float(s), depth=0)
            for i, s in zip(idx[0], self._norm(dists[0]))
        ]
        #  canonical de‑dup
        seen: set[str] = set()
        uniq: List[NodeScore] = []
        for h in hits:
            canon = re.sub(r"[^a-z0-9]", "", self.nodes[h.id].get("name", "").lower())
            if canon in seen:
                continue
            seen.add(canon)
            uniq.append(h)
        self._cache[q] = uniq
        return uniq

    #  Weighted BFS ---------------------------------------------------------
    @annotate('KnowledgeGraph::weighted_bfs')
    def weighted_bfs(self, seeds: Sequence[NodeScore], hops: int) -> List[NodeScore]:
        if not seeds:
            return []
        visited: Dict[str, float] = {}
        depths: Dict[str, int] = {}
        dq: Deque[Tuple[str, float, int]] = deque((s.id, s.score, 0) for s in seeds)
        while dq:
            nid, score, depth = dq.popleft()
            if depth > hops:
                continue
            if score <= visited.get(nid, 0.0):
                continue
            visited[nid] = score
            depths[nid] = depth
            for e in self.out_edges.get(nid, []):
                nbr = e["object"]
                pred = e["predicate"]
                edge_w = 1.5 if not pred.endswith("RELATED_TO") else 1.2
                edge_w += fuzz.partial_ratio(pred, "RELATED_TO") / 100.0
                if GENERIC_PAT.search(self.nodes.get(nbr, {}).get("name", "")):
                    edge_w *= GENERIC_PENALTY
                nxt_score = score * edge_w / (depth + 1.0)
                if nxt_score > visited.get(nbr, 0.0):
                    dq.append((nbr, nxt_score, depth + 1))
        return sorted(
            (NodeScore(nid, sc, depth=depths[nid]) for nid, sc in visited.items()),
            key=lambda x: x.score,
            reverse=True,
        )

    #  NodeInfo build -------------------------------------------------------
    @annotate('KnowledgeGraph::build_nodeinfo')
    def build_nodeinfo(
        self, sem: Sequence[NodeScore], graph: Sequence[NodeScore], q_tokens: Sequence[str]
    ) -> List[NodeInfo]:
        qt = [t.lower() for t in q_tokens if t]
        gmap, smap = {n.id: n for n in graph}, {n.id: n for n in sem}
        ids = set(gmap) | set(smap)
        out: List[NodeInfo] = []
        for nid in ids:
            raw = self.nodes[nid]
            name = raw.get("name", nid)
            desc = raw.get("description", "")
            txt_low = f"{name} {desc}".lower()
            hit = sum(1 for t in qt if t in txt_low)
            lex = np.sqrt(hit) / max(1, len(qt))
            evid = len(raw.get("source_papers", [])) + len(self.out_edges.get(nid, []))
            sem_sc = smap.get(nid, NodeScore(nid, 0.0)).score
            g_sc = gmap.get(nid, NodeScore(nid, 0.0)).score
            depth = gmap.get(nid, NodeScore(nid, 0, 0)).depth
            if GENERIC_PAT.search(name):
                sem_sc *= GENERIC_PENALTY
                g_sc *= GENERIC_PENALTY
            out.append(
                NodeInfo(
                    id=nid,
                    name=name,
                    category=raw.get("category", "?"),
                    description=desc,
                    score_sem=sem_sc,
                    score_graph=g_sc,
                    depth=depth,
                    lexical_overlap=lex,
                    evidence_ct=evid,
                )
            )
        return out

    #  Context assembly -----------------------------------------------------
    @annotate('KnowledgeGraph::build_context')
    def build_context(
        self,
        nodes: Sequence[NodeInfo],
        include_structured: bool,
        char_budget: int,
        hint_terms: Sequence[str] | None,
    ) -> str:
        parts: List[str] = []
        chars = 0

        if include_structured:
            triples: List[str] = []
            for ni in nodes:
                for e in self.out_edges.get(ni.id, []):
                    tgt = self.nodes.get(e["object"], {})
                    triples.append(
                        f"({ni.name}) -[{e['predicate'].split(':')[-1]}]-> ({tgt.get('name', e['object'])})"
                    )
                    if len(triples) >= CTX_VOLUME_TRIPLES:
                        break
                if len(triples) >= CTX_VOLUME_TRIPLES:
                    break
            blk = "Structured_KG_Facts:\n" + "\n".join(triples)
            parts.append(blk)
            chars += len(blk)

        for ni in nodes:
            raw = self.nodes[ni.id]
            lines = [
                f"## {ni.name} ({ni.category})",
                f"Combined_Score: {ni.score_prp:.3f}",
            ]
            if ni.description:
                lines.append(f"Description: {ni.description}")
            if raw.get("formula"):
                lines.append(f"Formula: {raw['formula']}")
            for pdf in raw.get("source_papers", [])[:3]:
                path = str(Path(PDF_DIR) / pdf)
                txt = load_pdf_text(path)
                snip = snippet_text(txt, PDF_SNIPPET_LEN, hint_terms)
                if snip:
                    lines.append(f"[PDF {pdf}]\n{snip}")
            if self.out_edges.get(ni.id):
                lines.append("Relations:")
                for e in sorted(self.out_edges[ni.id], key=lambda x: x["predicate"]):
                    tgt = self.nodes.get(e["object"], {})
                    pred = e["predicate"].split(":")[-1]
                    lines.append(f"- {pred}: {tgt.get('name', e['object'])}")
            sec = "\n".join(lines)
            parts.append(sec)
            chars += len(sec)
            if chars >= char_budget:
                break
        return _strip_ansi("\n\n".join(parts))


# ───────────────────── retrieval orchestrator ─────────────────────
@annotate('decompose')
def decompose(q: str) -> List[str]:
    segs = re.split(r"[?;,]|\\band\\b|\\bthen\\b", q, flags=re.I)
    out = [s.strip() for s in segs if len(s.strip()) >= 3]
    return out or [q]


@annotate('retrieve_nodes')
def retrieve_nodes(q: str, kg: KnowledgeGraph) -> List[NodeInfo]:
    ents = extract_query_entities(q)
    seeds = kg.semantic_search(q)[: DEFAULT_K * 2]

    if STEPWISE:
        for sub in decompose(q)[:STEPWISE_MAX_STEPS]:
            seeds.extend(kg.semantic_search(sub)[:DEFAULT_K])

    #  keep highest score per node
    s_map: Dict[str, NodeScore] = {}
    for ns in seeds:
        cur = s_map.get(ns.id)
        if cur is None or ns.score > cur.score:
            s_map[ns.id] = ns
    sem = list(s_map.values())

    graph: List[NodeScore] = []
    if ENABLE_BFS:
        graph = kg.weighted_bfs(
            sorted(sem, key=lambda x: x.score, reverse=True)[:BFS_SEED_TOPK],
            hops=MAX_BFS_HOPS,
        )

    infos = kg.build_nodeinfo(sem, graph, ents)
    ranked = sorted(infos, key=lambda x: x.score_prp, reverse=True)
    #  evidence‑aware trimming
    ranked = sorted(
        ranked,
        key=lambda x: (x.score_prp, x.evidence_ct),
        reverse=True,
    )[:DEFAULT_K]
    return ranked


# ───────────────────── Ollama client ─────────────────────
class OllamaClient:
    @annotate('OllamaClient::__init__')
    def __init__(self, url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL) -> None:  # noqa: D401
        self.url, self.model = url, model
        self.timeout = aiohttp.ClientTimeout(total=300)

    async def chat(self, messages: Sequence[Dict[str, str]]) -> str:
        async with aiohttp.ClientSession(timeout=self.timeout) as sess:
            r = await sess.post(
                self.url,
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": list(messages),
                    "options": {"temperature": 0.0},
                },
            )
            r.raise_for_status()
            js = await r.json()
        return js.get("message", {}).get("content", "")


# ───────────────────── conversation helpers ─────────────────────
class Conversation:
    @annotate('Conversation::__init__')
    def __init__(self, system_prompt: str) -> None:  # noqa: D401
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    @annotate('Conversation::add')
    def add(self, user: str, assistant: str) -> None:
        self.messages.append({"role": "user", "content": user})
        self.messages.append({"role": "assistant", "content": assistant})

    @annotate('Conversation::build')
    def build(self, user: str, prepend: str | None = None) -> List[Dict[str, str]]:
        msgs = list(self.messages)
        if prepend:
            msgs.append({"role": "system", "content": prepend})
        msgs.append({"role": "user", "content": user})
        return msgs


BASELINE_SYSTEM = (
    "You are an expert materials‑science assistant. Answer clearly and concisely. "
    "If unsure, say so."
)
RAG_SYSTEM = (
    "You are an expert materials‑science assistant with access to a retrieved KG context. "
    "Use it as evidence, but flag gaps if context is missing or noisy."
)


@annotate('build_baseline_prompt')
def build_baseline_prompt(q: str) -> str:
    return f"Question: {q}\n\nAnswer:"


@annotate('build_rag_prompt')
def build_rag_prompt(q: str, ctx: str) -> str:
    return (
        f"User Question:\n{q}\n\n"
        f"Retrieved Context (may be noisy):\n{ctx}\n\n"
        "Instructions:\n"
        "- Fully answer the question, citing relevant context.\n"
        "- If context is insufficient, rely on domain knowledge and note assumptions.\n"
        "- Organise with short headings/bullets; be concise.\n\n"
        "Answer:"
    )


# ───────────────────── main Q&A loop ─────────────────────
async def answer(
    q: str, kg: KnowledgeGraph, cli: OllamaClient, rag_c: Conversation, base_c: Conversation
) -> None:
    print(Fore.MAGENTA + f"\nQ: {q}" + Style.RESET_ALL)
    infos = retrieve_nodes(q, kg)
    print(
        Fore.CYAN
        + "Selected: "
        + str([f"{n.id}:{n.score_prp:.2f}" for n in infos])
        + Style.RESET_ALL
    )
    ctx = kg.build_context(
        infos,
        include_structured=STRUCT_CTX,
        char_budget=CTX_SOFT_LIMIT,
        hint_terms=_tokenize(q),
    )

    base_prompt = build_baseline_prompt(q)
    rag_prompt = build_rag_prompt(q, ctx)

    base_resp = await cli.chat(base_c.build(base_prompt))
    rag_resp = await cli.chat(rag_c.build(rag_prompt))

    base_c.add(base_prompt, base_resp)
    rag_c.add(rag_prompt, rag_resp)

    print(Fore.GREEN + "\n[Baseline]\n" + base_resp + Style.RESET_ALL)
    print(Fore.GREEN + "\n[KG‑RAG]\n" + rag_resp + Style.RESET_ALL)


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=GRAPH_FILE)
    ap.add_argument("--question", type=str, help="One‑shot question, then exit")
    args = ap.parse_args()

    kg = KnowledgeGraph(str(args.graph))
    cli = OllamaClient()
    rag_c = Conversation(RAG_SYSTEM)
    base_c = Conversation(BASELINE_SYSTEM)

    if args.question:
        await answer(args.question, kg, cli, rag_c, base_c)
        return

    while True:
        try:
            q = input(Fore.YELLOW + "Ask (exit to quit): " + Style.RESET_ALL).strip()
        except EOFError:
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        await answer(q, kg, cli, rag_c, base_c)


@annotate('main')
def main() -> None:  # pragma: no cover
    asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover
    main()
