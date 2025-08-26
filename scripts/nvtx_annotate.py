#!/usr/bin/env python3
"""
nvtx_annotate.py

Safely adds @annotate decorators and inserts `from nvtx import annotate` after the module docstring and existing imports,
preserving original formatting and docstrings.
"""

import ast
import os
import re
import sys
import argparse
import subprocess
from typing import List, Tuple


def ensure_nvtx():
    try:
        import nvtx
    except ImportError:
        print("[INFO] nvtx not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nvtx"])


def get_function_positions_with_class(source: str) -> List[Tuple[int, str]]:
    """Return a list of (line_number, full_label) for each function (with optional class)."""
    tree = ast.parse(source)
    positions = []
    class_stack = []

    class FuncVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node):
            label = f"{class_stack[-1]}::{node.name}" if class_stack else node.name
            positions.append((node.lineno, label))

    FuncVisitor().visit(tree)
    return positions


def insert_import_line(lines: List[str]) -> List[str]:
    """Insert `from nvtx import annotate` after module docstring and imports."""
    if any("from nvtx import annotate" in l for l in lines):
        return lines

    # Find end of docstring
    idx = 0
    if lines[0].strip().startswith(('"""', "'''")):
        quote = lines[0][:3]
        for i in range(1, len(lines)):
            if lines[i].strip().endswith(quote):
                idx = i + 1
                break

    # Find end of import block
    while idx < len(lines) and (
        lines[idx].startswith("import ") or lines[idx].startswith("from ")
    ):
        idx += 1

    lines.insert(idx, "from nvtx import annotate")
    return lines


def add_decorators(source_lines: List[str], positions: List[Tuple[int, str]]) -> List[str]:
    """Add @annotate decorators just before function defs."""
    # Adjust lines from 1-based AST line numbers
    adjusted_lines = source_lines[:]
    offset = 0
    for lineno, label in sorted(positions):
        insert_at = lineno - 1 + offset

        # Avoid duplicate
        if "@annotate" in adjusted_lines[insert_at - 1]:
            continue

        indent = re.match(r'^(\s*)', adjusted_lines[insert_at]).group(1)
        annotation_line = f"{indent}@annotate('{label}')"
        adjusted_lines.insert(insert_at, annotation_line)
        offset += 1
    return adjusted_lines


def annotate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.splitlines()
    positions = get_function_positions_with_class(source)

    # Decorate functions
    lines = add_decorators(lines, positions)

    # Insert import
    lines = insert_import_line(lines)

    # Save
    backup = filepath + ".bak"
    os.rename(filepath, backup)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[✅] Annotated: {filepath} (backup: {backup})")


def main():
    parser = argparse.ArgumentParser(description="Add @annotate decorators to all functions.")
    parser.add_argument("script", help="Python script to modify")
    args = parser.parse_args()

    ensure_nvtx()
    annotate_file(args.script)


if __name__ == "__main__":
    main()
