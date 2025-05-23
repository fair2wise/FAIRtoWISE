import json
from typing import List, Dict, Any, Set
from collections import defaultdict
import requests
import numpy as np
from rapidfuzz import fuzz  # fuzzy keyword boost
from sentence_transformers import SentenceTransformer
import faiss

# === CONFIGURATION ===
OLLAMA_MODEL = "llama3.2"  # Adjust if you use a different tag
OLLAMA_API_URL = "http://localhost:11434/api/chat"
GRAPH_FILE = "storage/kg/matkg_graph.json"
K_NEIGHBORS = 3              # how many KG nodes to pull per query
HOPS = 4                     # default neighborhood size
EMBED_MODEL = "all-MiniLM-L6-v2"  # 80 MB, fast & solid

# === KNOWLEDGE GRAPH =====================================================
class KnowledgeGraph:
    """Lightweight in‑memory property‑graph wrapper."""

    def __init__(self, graph_file: str):
        with open(graph_file, "r") as f:
            data = json.load(f)
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in data["things"]}
        self.edges: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for e in data["associations"]:
            self.edges[e["subject"].strip()].append(e)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.nodes.get(node_id, {})

    def get_neighbors(self, node_id: str, hops: int = 1) -> Set[str]:
        visited, frontier = set(), {node_id}
        for _ in range(hops):
            nxt = set()
            for nid in frontier:
                visited.add(nid)
                for e in self.edges.get(nid, []):
                    nxt.add(e["object"])
            frontier = nxt
        return visited.union(frontier)

    def build_context(self, anchor_id: str, hops: int = 1) -> str:
        lines: List[str] = []
        for nid in sorted(self.get_neighbors(anchor_id, hops)):
            n = self.get_node(nid)
            if not n:
                continue
            lines.append(f"\n## {n.get('name', nid)} ({n.get('category', 'Unknown')})")
            lines.append(f"Description: {n.get('description', 'N/A')}")
            if n.get('formula'):
                lines.append(f"Formula: {n['formula']}")
            for e in self.edges.get(nid, []):
                m = self.get_node(e["object"])
                if not m:
                    continue
                pred = e["predicate"].split(":")[-1]
                entry = f"- {pred}: {m.get('name', e['object'])}"
                if e.get("has_evidence"):
                    entry += f" ({e['has_evidence']})"
                lines.append(entry)
        return "\n".join(lines)

# === NODE SEARCHER =======================================================
class NodeSearcher:
    """Hybrid embedding + fuzzy keyword retrieval over KG nodes."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.model = SentenceTransformer(EMBED_MODEL)

        texts, ids = [], []
        for nid, node in kg.nodes.items():
            blob = f"{node.get('name', '')} {node.get('description', '')}".strip()
            if blob:
                texts.append(blob)
                ids.append(nid)

        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs.astype("float32"))
        self.id_arr = np.array(ids)

    def _exact_name_match(self, query: str, name: str) -> bool:
        q, n = query.lower(), name.lower()
        return f" {n} " in f" {q} "

    def find(self, query: str, k: int = K_NEIGHBORS) -> List[str]:
        q_vec = self.model.encode([query], normalize_embeddings=True)
        _, idx = self.index.search(q_vec.astype("float32"), k * 6)
        cand_ids = self.id_arr[idx[0]].tolist()

        scored: List[tuple[int, int, str]] = []
        for cid in cand_ids:
            node = self.kg.get_node(cid)
            name = node.get("name", "")
            text = f"{name} {node.get('description', '')}"
            fuzzy = fuzz.token_set_ratio(query, text)
            exact = 1 if self._exact_name_match(query, name) else 0
            scored.append((exact, fuzzy, cid))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best, seen = [], set()
        for _, _, cid in scored:
            if cid not in seen:
                best.append(cid)
                seen.add(cid)
            if len(best) >= k:
                break
        return best

# === OLLAMA CALL =========================================================
def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.0) -> str:
    try:
        r = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=90,
        )
        if r.status_code != 200:
            return f"[LLM Error {r.status_code}]"
        try:
            data = r.json()
        except ValueError:
            for line in reversed([ln for ln in r.text.splitlines() if ln.strip()]):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
            else:
                return "[Cannot parse LLM response]"
        return data.get("message", {}).get("content", "[No content]")
    except Exception as e:
        return f"[Exception: {e}]"

# === PROMPT HELPERS ======================================================

def ask_rag(question: str, context: str) -> str:
    p = f"""Use the following knowledge‑graph context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    return _call_ollama(p)


def ask_baseline(question: str) -> str:
    return _call_ollama(f"Answer using your general knowledge:\n\nQuestion: {question}\nAnswer:")

# === CLI LOOP ============================================================

def interactive():
    print("Loading KG …", end=" ")
    kg = KnowledgeGraph(GRAPH_FILE)
    searcher = NodeSearcher(kg)
    print("done.")

    while True:
        q = input("\nAsk a materials‑science question (or 'exit'): ")
        if q is None or q.strip().lower() == "exit":
            break
        q = q.strip()
        if not q:
            continue

        hits = searcher.find(q)
        print("Top KG matches:", hits)
        context = "\n---\n".join(kg.build_context(nid, HOPS) for nid in hits)

        mode = input("Mode — rag | baseline | compare [rag]: ").strip().lower() or "rag"
        if mode == "baseline":
            print("\n[Baseline]\n" + ask_baseline(q))
        elif mode == "compare":
            print("\n[KG‑RAG]\n" + ask_rag(q, context))
            print("\n[Baseline]\n" + ask_baseline(q))
        else:
            print("\n[KG‑RAG]\n" + ask_rag(q, context))


if __name__ == "__main__":
    interactive()
