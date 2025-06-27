#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_rag_ollama.py –– Enhanced KG-RAG integration with Ollama with terminal color,
now including PDF snippets from any node’s listed source papers in polymer_papers/.
"""
import argparse
import asyncio
import json
import logging
import sys
import fitz              # PyMuPDF for PDF text extraction
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import faiss
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from colorama import init as colorama_init, Fore, Style

# === CONFIGURATION ===
OLLAMA_MODEL     = "gemma3:27b"
OLLAMA_API_URL   = "http://localhost:11434/api/chat"
GRAPH_FILE       = "/pscratch/sd/d/dabramov/fair2wise/storage/kg/matkg-jun6.json"
PDF_DIR          = "polymer_papers"    # directory where the PDFs live
DEFAULT_K        = 4
DEFAULT_MAX_HOPS = 3
EMBED_MODEL      = "all-MiniLM-L6-v2"
PDF_SNIPPET_LEN  = 1000   # characters to pull from each PDF

# Initialize colorama
colorama_init(autoreset=True)

# Setup logging with colored timestamps
class ColorFormatter(logging.Formatter):
    def format(self, record):
        time = Fore.CYAN + self.formatTime(record) + Style.RESET_ALL
        level = record.levelname
        msg = record.getMessage()
        return f"{time} {Fore.MAGENTA}{level}{Style.RESET_ALL}: {msg}"

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter())
logger = logging.getLogger("kg_rag_ollama")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class NodeScore:
    id: str
    score: float

class KnowledgeGraph:
    """Lightweight in-memory directed KG with embeddings, now pulling PDF snippets."""
    def __init__(self, graph_file: str, embed_model: str = EMBED_MODEL):
        logger.info(Fore.YELLOW + "Loading KG..." + Style.RESET_ALL)
        with open(graph_file) as f:
            data = json.load(f)
        self.nodes = {n["id"]: n for n in data["things"]}
        self.out_edges = defaultdict(list)
        for e in data["associations"]:
            self.out_edges[e["subject"]].append(e)

        # Build FAISS index on node texts
        self.embed_model = SentenceTransformer(embed_model)
        texts, self.ids = [], []
        for nid, node in self.nodes.items():
            texts.append(f"{node.get('name','')} {node.get('description','')}")
            self.ids.append(nid)
        emb = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb.astype("float32"))
        self.id_map = np.array(self.ids)
        self._cache = {}
        logger.info(Fore.GREEN + f"KG loaded: {len(self.nodes)} nodes, {sum(len(v) for v in self.out_edges.values())} edges." + Style.RESET_ALL)

    def semantic_search(self, query: str, topk: int = DEFAULT_K*2) -> List[NodeScore]:
        if query in self._cache:
            return self._cache[query]
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        _, idx = self.index.search(q_emb.astype("float32"), topk)
        hits = [NodeScore(id=self.id_map[i], score=1.0) for i in idx[0]]
        self._cache[query] = hits
        return hits

    def weighted_bfs(self, seeds: List[NodeScore], max_hops: int = DEFAULT_MAX_HOPS) -> List[NodeScore]:
        visited = {}
        queue = deque((s.id, s.score, 0) for s in seeds)
        while queue:
            nid, base_score, depth = queue.popleft()
            if depth > max_hops:
                continue
            visited[nid] = max(visited.get(nid, 0), base_score)
            for edge in self.out_edges.get(nid, []):
                nbr = edge["object"]
                w = 1.2 if edge["predicate"].endswith("RELATED_TO") else 1.5
                rel_score = fuzz.token_sort_ratio(edge["predicate"], "RELATED_TO") / 100.0
                score = base_score * w + rel_score / (depth + 1)
                if score >= 0.01:
                    queue.append((nbr, score, depth+1))
        ranked = sorted((NodeScore(id=n, score=s) for n,s in visited.items()),
                        key=lambda x: x.score, reverse=True)
        return ranked[:DEFAULT_K]

    def build_context(self, nodes: List[NodeScore]) -> str:
        sections = []
        for ns in nodes:
            n = self.nodes[ns.id]
            header = Fore.BLUE + f"## {n.get('name')} ({n.get('category')})" + Style.RESET_ALL
            lines = [header]

            # PDF snippets from polymer_papers/
            for paper in n.get("source_papers", []):
                pdf_path = Path(PDF_DIR) / paper
                try:
                    doc = fitz.open(str(pdf_path))
                    text = "".join(p.get_text() for p in doc)
                    snippet = text[:PDF_SNIPPET_LEN].replace("\n", " ")
                    lines.append(Fore.YELLOW + f"[PDF snippet from {pdf_path}]" + Style.RESET_ALL)
                    lines.append(snippet + "…")
                except Exception as e:
                    lines.append(Fore.RED + f"[Could not load PDF {pdf_path}: {e}]" + Style.RESET_ALL)

            # Node description and formula
            lines.append(f"Description: {n.get('description','N/A')}")
            if n.get("formula"):
                lines.append(f"Formula: {n['formula']}")

            # Outgoing edges
            for e in sorted(self.out_edges.get(ns.id, []), key=lambda x: x["predicate"]):
                tgt = self.nodes.get(e["object"], {})
                pred = e["predicate"].split(":")[-1]
                entry = f"- {pred}: {tgt.get('name', e['object'])}"
                if e.get("has_evidence"):
                    entry += f" ({e['has_evidence']})"
                lines.append(entry)

            sections.append("\n".join(lines))

        return "\n\n".join(sections)


class OllamaClient:
    """Async client for Ollama."""
    def __init__(self, url=OLLAMA_API_URL, model=OLLAMA_MODEL, temp=0.0):
        self.url, self.model, self.temp = url, model, temp

    async def chat(self, prompt: str, history: List[Dict[str,str]]) -> str:
        msgs = history + [{"role":"user","content":prompt}]
        async with aiohttp.ClientSession() as sess:
            resp = await sess.post(
                self.url,
                json={
                    "model":    self.model,
                    "stream":   False,               # ← disable streaming
                    "messages": msgs,
                    "options":  {"temperature": self.temp}
                },
                timeout=90
            )
            txt = await resp.text()
            try:
                data = json.loads(txt)
                content = data["message"]["content"]
            except Exception:
                content = txt
            history.append({"role":"assistant","content":content})
            return content


rag_history: List[Dict[str,str]] = []
baseline_history: List[Dict[str,str]] = []

async def answer_question(question: str,
                          kg: KnowledgeGraph,
                          client: OllamaClient):
    print(Fore.MAGENTA + f"\nQuestion: {question}" + Style.RESET_ALL)

    seeds    = kg.semantic_search(question)
    print(Fore.YELLOW + f"Seeds: {[s.id for s in seeds]}" + Style.RESET_ALL)

    relevant = kg.weighted_bfs(seeds)
    print(Fore.CYAN + f"Selected: {[r.id for r in relevant]}" + Style.RESET_ALL)

    context = kg.build_context(relevant)

    # KG-RAG call
    if not rag_history:
        rag_history.append({"role":"system","content":"You are an expert materials-science assistant."})
    rag_resp = await client.chat(f"Use this KG context:\n\n{context}\n\nQ: {question}\nA:", rag_history)

    # Baseline call
    if not baseline_history:
        baseline_history.append({"role":"system","content":"You are an expert materials-science assistant without KG."})
    base_resp = await client.chat(f"Answer concisely:\n\nQ: {question}\nA:", baseline_history)

    # Print both
    print(Fore.GREEN + "\n[Baseline Answer]\n" + base_resp + Style.RESET_ALL)
    print(Fore.GREEN + "\n[KG-RAG Answer]\n" + rag_resp  + Style.RESET_ALL)


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=GRAPH_FILE)
    parser.add_argument("--question", type=str)
    args = parser.parse_args()

    kg     = KnowledgeGraph(str(args.graph))
    client = OllamaClient()

    if args.question:
        await answer_question(args.question, kg, client)
    else:
        while True:
            q = input(Fore.YELLOW + "Enter question (or 'exit'): " + Style.RESET_ALL).strip()
            if not q or q.lower()=="exit":
                break
            await answer_question(q, kg, client)


if __name__ == "__main__":
    asyncio.run(main_async())
