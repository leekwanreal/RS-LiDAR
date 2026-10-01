#!/usr/bin/env python3
"""
check_notebook_syntax.py - Standalone Guardrail for Jupyter Notebook Python Syntax

Validates that all Python code cells in target .ipynb files have 100% valid Python syntax.
Handles IPython magics (%, %%) and shell commands (!, !) including multiline continuations.

Usage:
    python check_notebook_syntax.py
    python check_notebook_syntax.py colab/LiDAR_Table2_Replication_Colab.ipynb
    python check_notebook_syntax.py colab/*.ipynb kaggle/*.ipynb
"""

import os
import sys
import glob
import json
import ast
import argparse

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check_single_notebook(path):
    """Checks all code cells in a single .ipynb file for Python syntax errors.
    Returns: (is_valid: bool, error_messages: list[str])
    """
    if not os.path.exists(path):
        return False, [f"File not found: {path}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return False, [f"Invalid JSON in {path}: {e}"]

    errors = []
    cells = nb.get("cells", [])

    for cell_idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        lines = cell.get("source", [])
        if isinstance(lines, str):
            lines = lines.splitlines(keepends=True)

        cleaned_lines = []
        in_continuation = False
        cmd_indent = ""

        for line in lines:
            indent = " " * (len(line) - len(line.lstrip()))
            stripped = line.strip()

            if in_continuation:
                cleaned_lines.append(cmd_indent + "pass\n")
                if not stripped.endswith("\\"):
                    in_continuation = False
            elif stripped.startswith("!") or stripped.startswith("%"):
                cmd_indent = indent
                cleaned_lines.append(indent + "pass\n")
                if stripped.endswith("\\"):
                    in_continuation = True
            else:
                cleaned_lines.append(line)

        code_str = "".join(cleaned_lines)
        if not code_str.strip():
            continue

        try:
            ast.parse(code_str, filename=f"{path} [Cell {cell_idx}]")
        except SyntaxError as e:
            errors.append(
                f"[Cell {cell_idx}] {e.__class__.__name__} at line {e.lineno}: {e.msg}\n"
                f"   >>> {e.text.strip() if e.text else ''}"
            )

    return (len(errors) == 0), errors


def main():
    parser = argparse.ArgumentParser(description="Validate Python syntax in Jupyter Notebooks (.ipynb)")
    parser.add_argument("paths", nargs="*", default=[], help="Notebook paths or globs to validate")
    parser.add_argument("--quiet", action="store_true", help="Only show failing notebooks")
    args = parser.parse_args()

    target_files = []
    if args.paths:
        for p in args.paths:
            matched = glob.glob(p, recursive=True)
            if matched:
                target_files.extend(matched)
            elif os.path.exists(p):
                target_files.append(p)
    else:
        # Default: scan all notebooks in current repository
        target_files = glob.glob("**/*.ipynb", recursive=True)

    target_files = sorted(list(set(target_files)))
    if not target_files:
        print("⚠️ No .ipynb files found to validate.")
        return 0

    all_passed = True
    passed_count = 0

    print("=" * 70)
    print(f"🔍 JUPYTER NOTEBOOK SYNTAX GUARDRAIL: Validating {len(target_files)} notebooks")
    print("=" * 70)

    for nb_path in target_files:
        is_valid, errors = check_single_notebook(nb_path)
        if is_valid:
            passed_count += 1
            if not args.quiet:
                print(f"✅ [PASS] {nb_path}")
        else:
            all_passed = False
            print(f"❌ [FAIL] {nb_path}")
            for err in errors:
                print(f"   • {err}")

    print("=" * 70)
    print(f"📊 SUMMARY: {passed_count}/{len(target_files)} notebooks passed.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
