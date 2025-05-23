

import json
from typing import List, Dict, Any, Set
from collections import defaultdict
import requests

# === CONFIGURATION ===
OLLAMA_MODEL = "llama3.2"  # Adjust to match the model tag you pulled
OLLAMA_API_URL = "http://localhost:11434/api/chat"
GRAPH_FILE = "storage/kg/matkg_graph.json"

# === KNOWLEDGE GRAPH LOADER ===
class KnowledgeGraph:
    """Light‑weight in‑memory property‑graph wrapper around the uploaded JSON KG."""

    def __init__(self, graph_file: str):
        with open(graph_file, "r") as f:
            data = json.load(f)

        # Node and edge indices for O(1) lookup
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in data["things"]}
        self.edges: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for e in data["associations"]:
            self.edges[e["subject"]].append(e)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.nodes.get(node_id, {})

    def get_neighbors(self, node_id: str, hops: int = 1) -> Set[str]:
        """Return the set of node_ids in the ≤ `hops`‑hop neighborhood of `node_id`."""
        visited, frontier = set(), {node_id}
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                visited.add(nid)
                for edge in self.edges.get(nid, []):
                    next_frontier.add(edge["object"])
            frontier = next_frontier
        return visited.union(frontier)

    def build_context(self, node_id: str, hops: int = 1) -> str:
        """Serialise the local subgraph around `node_id` into readable markdown."""
        out: List[str] = []
        for nid in sorted(self.get_neighbors(node_id, hops)):
            node = self.get_node(nid)
            if not node:
                continue
            out.append(f"\n## {node.get('name', nid)} ({node.get('category', 'Unknown')})")
            out.append(f"Description: {node.get('description', 'N/A')}")
            if node.get("formula"):
                out.append(f"Formula: {node['formula']}")
            for edge in self.edges.get(nid, []):
                obj = self.get_node(edge["object"])
                if not obj:
                    continue
                pred = edge["predicate"].split(":")[-1]
                line = f"- {pred}: {obj.get('name', edge['object'])}"
                if edge.get("has_evidence"):
                    line += f" ({edge['has_evidence']})"
                out.append(line)
        return "\n".join(out)

# === OLLAMA CALL HELPERS ===

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.0) -> str:
    """POST the prompt to the local Ollama server and return the assistant message."""
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return f"[LLM Error {resp.status_code}]"
        return resp.json().get("message", {}).get("content", "[No content]")
    except Exception as exc:
        return f"[Exception calling Ollama: {exc}]"

# --- RAG & BASELINE WRAPPERS ---------------------------------------------

def ask_with_rag(question: str, context: str) -> str:
    rag_prompt = f"""Use the following knowledge‑graph context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    return _call_ollama(rag_prompt)


def ask_baseline(question: str) -> str:
    baseline_prompt = f"""Answer the following question using your general knowledge:

Question: {question}
Answer:"""
    return _call_ollama(baseline_prompt)

# === CLI DRIVER ===

def interactive():
    print("Loading Knowledge Graph …", end=" ")
    kg = KnowledgeGraph(GRAPH_FILE)
    print(f"done. {len(kg.nodes)} nodes loaded.\n")

    while True:
        print("--- KG‑RAG playground ---")
        entity = input("Node ID (e.g., matkg:P3HT) or 'exit': ").strip()
        if entity.lower() == "exit":
            break
        if entity not in kg.nodes:
            print("[!] Node not found.\n")
            continue

        hops_in = input("Neighborhood hops [default 1]: ").strip()
        hops = int(hops_in) if hops_in.isdigit() else 1
        context = kg.build_context(entity, hops)
        print("[Context built]\n")

        question = input("Your question: ").strip()
        mode = input("Mode — rag | baseline | compare [default rag]: ").strip().lower() or "rag"

        if mode == "baseline":
            print("\n[Baseline Answer]\n" + ask_baseline(question))
        elif mode == "compare":
            print("\n[KG‑RAG Answer]\n" + ask_with_rag(question, context))
            print("\n[Baseline Answer]\n" + ask_baseline(question))
        else:  # rag
            print("\n[KG‑RAG Answer]\n" + ask_with_rag(question, context))
        print()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    interactive()
