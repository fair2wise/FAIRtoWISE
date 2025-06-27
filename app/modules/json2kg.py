#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json2kg.py -- Optimized conversion of extracted_terms JSON → MatKG graph.json

Features:
  - Precompiled regex for ID cleaning
  - Efficient list handling
  - Structured logging with configurable verbosity
  - Robust error handling
  - Type hints and concise docstrings
  - Full utilization of extracted term fields: formula, formula_validation, properties
  - Pytest test suite included below
"""
import json
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# Precompile regex pattern for performance
_CLEAN_PATTERN = re.compile(r"[^A-Za-z0-9\-]")


def make_id(term: str) -> str:
    """
    Convert a human-readable term into a MatKG node ID.

    - Prepends "matkg:"
    - Removes all characters except letters, digits, and hyphens
    - Removes spaces
    """
    cleaned = _CLEAN_PATTERN.sub("", term.replace(" ", ""))
    return f"matkg:{cleaned}"


def ensure_list(val: Any) -> List[Any]:
    """
    Guarantee that the return is a list:
      - None     → []
      - scalar   → [val]
      - list     → val
    """
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def build_graph(
    raw_terms: Iterable[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a MatKG-compatible graph from raw term records.

    Returns a dict with keys:
      - "things": list of node dicts
      - "associations": list of edge dicts
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for term in raw_terms:
        name = term.get("term") or term.get("name") or "UNKNOWN"
        tid = make_id(name)

        # Create node if new
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

        # Process relations
        for rel in ensure_list(term.get("relations")):
            tgt = rel.get("related_term")
            if not tgt:
                continue
            rid = make_id(tgt)

            # stub for unseen target
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

            pred = f"rel:{rel.get('relation', 'RELATED_TO')}"
            sig = (tid, pred, rid)
            if sig in seen:
                continue
            seen.add(sig)

            evidence = ensure_list(rel.get("evidence"))
            edges.append({
                "subject": tid,
                "predicate": pred,
                "object": rid,
                "has_evidence": "; ".join(evidence) if evidence else None,
            })

    return {"things": list(nodes.values()), "associations": edges}


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert extracted_terms JSON → MatKG graph JSON"
    )
    parser.add_argument(
        "input_json", type=Path,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "output_json", type=Path,
        help="Path to output graph JSON file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Increase output verbosity"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level, format="%(levelname)s: %(message)s")

    try:
        with args.input_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        terms = data.get("terms") if isinstance(data, dict) and "terms" in data else data
        graph = build_graph(terms)
        args.output_json.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info(
            "Wrote %d nodes and %d edges → %s",
            len(graph["things"]), len(graph["associations"]), args.output_json
        )
    except Exception as e:
        logging.error("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ----------------------- Pytest Test Suite -----------------------
# To run: pytest test_json2kg.py

def test_make_id_simple():
    assert make_id("P3HT") == "matkg:P3HT"
    assert make_id("Bulk Heterojunction OPV") == "matkg:BulkHeterojunctionOPV"
    assert make_id("pAQM-2TV") == "matkg:pAQM-2TV"


def test_ensure_list():
    assert ensure_list(None) == []
    assert ensure_list(5) == [5]
    assert ensure_list([1, 2, 3]) == [1, 2, 3]


def test_build_graph_fields():
    raw = [{
        "term": "X",
        "definition": "Def",
        "category": "Cat",
        "formula": "H2O",
        "formula_validation": {"status": "ok"},
        "properties": [{"property": "density", "value": 1}]
    }]
    graph = build_graph(raw)
    node = {n['id']: n for n in graph['things']}['matkg:X']
    assert node['formula'] == "H2O"
    assert node['formula_validation']['status'] == "ok"
    assert node['properties'][0]['property'] == "density"


def test_build_graph_minimal(tmp_path):
    raw = [{"term": "A", "relations": [{"related_term": "B", "relation": "TEST"}]}]
    graph = build_graph(raw)
    assert len(graph["things"]) == 2
    assert len(graph["associations"]) == 1
    edge = graph["associations"][0]
    assert edge["predicate"] == "rel:TEST"
    assert edge["has_evidence"] is None


def test_cli(tmp_path, capsys):
    in_json = tmp_path / "in.json"
    out_json = tmp_path / "out.json"
    data = {"terms": [{"term": "X"}]} 
    in_json.write_text(json.dumps(data))
    sys.argv = ["json2kg.py", str(in_json), str(out_json)]
    main()
    captured = capsys.readouterr()
    assert "Wrote 1 nodes and 0 edges" in captured.out
    out = json.loads(out_json.read_text())
    assert "things" in out and "associations" in out
