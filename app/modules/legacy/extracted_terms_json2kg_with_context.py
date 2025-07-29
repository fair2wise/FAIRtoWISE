#!/usr/bin/env python3
"""
json2kg.py  ‑‑  Convert extracted_terms*.json → MatKG graph.json

Works on BOTH:
  • original file (maps mixed with strings)
  • flattened file (lists of plain strings)

Output schema is exactly what KnowledgeGraph expects.

Author: ChatGPT License: MIT
"""
from __future__ import annotations
import json, argparse, hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Iterable

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def slug(term: str) -> str:
    """Stable, URL‑safe node id derived from the term name."""
    return hashlib.sha1(term.encode("utf‑8")).hexdigest()[:16]

def ensure_list(val: Any) -> List[Any]:
    """Return [] for null, [val] for scalar, val for list."""
    if val is None:
        return []
    return val if isinstance(val, list) else [val]

# --------------------------------------------------------------------------- #
# Core                                                                        #
# --------------------------------------------------------------------------- #
def build_graph(raw_terms: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Transform raw term records into {things, associations} dict."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen_edges: Set[tuple[str, str, str]] = set()

    for term in raw_terms:
        tname: str = term.get("term") or term.get("name") or "UNKNOWN"
        tid        = slug(tname)

        # ---------------- Nodes ------------------------------------------------
        if tid not in nodes:
            nodes[tid] = {
                "id":   tid,
                "name": tname,
                "category": term.get("category", "Unknown"),
                "description": term.get("definition", "") or "N/A",
                "pages": term.get("pages", []),
                "source_papers": ensure_list(term.get("source_papers")),
                "context_snippets": ensure_list(term.get("context_snippets")),
            }

        # ---------------- Edges ------------------------------------------------
        for rel in ensure_list(term.get("relations")):
            target_name = rel.get("related_term")
            if not target_name:
                continue
            rid = slug(target_name)

            # stub the target node if it hasn't appeared yet
            if rid not in nodes:
                nodes[rid] = {
                    "id": rid,
                    "name": target_name,
                    "category": "Unknown",
                    "description": "",
                    "pages": [],
                    "source_papers": [],
                    "context_snippets": [],
                }

            predicate = f"rel:{rel.get('relation','RELATED_TO')}"
            sig = (tid, predicate, rid)
            if sig in seen_edges:
                continue
            seen_edges.add(sig)

            edges.append({
                "subject": tid,
                "predicate": predicate,
                "object": rid,
                "has_evidence": "; ".join(ensure_list(rel.get("evidence"))) or None,
            })

    return {"things": list(nodes.values()), "associations": edges}

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Convert extracted_terms*.json to MatKG graph.json")
    ap.add_argument("input_json",  type=Path, help="Original or flattened JSON file")
    ap.add_argument("output_json", type=Path, help="Destination graph JSON")
    args = ap.parse_args()

    with args.input_json.open() as f:
        data = json.load(f)

    # tolerate both top‑level {"terms":[...]} and flat list
    terms = data["terms"] if isinstance(data, dict) and "terms" in data else data
    graph = build_graph(terms)

    args.output_json.write_text(json.dumps(graph, indent=2))
    print(f"Wrote {len(graph['things']):,} nodes and {len(graph['associations']):,} "
          f"edges → {args.output_json}")

if __name__ == "__main__":
    main()