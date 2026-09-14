# Agent Instructions for Diffusion-LiDAR-Sampling

## Git Workflow Rule
- Whenever performing Git operations (staging, committing, and pushing), **always combine `git add`, `git commit`, and `git push` into a single command line** (e.g. `git add . ; git commit -m "<message>" ; git push`) to minimize round-trips and save execution time.

## Project Context
- **Repository Overview & Fixes History**: Refer to `README_CONTEXT.md` (kept in `.gitignore`) for a summary of the project architecture, key scripts (`lookahead_sampling.py`, `LiDAR_sampling.py`), and past resolved bugs/fixes.
- Always maintain and update `README_CONTEXT.md` with any newly resolved issues or architectural changes.

## Mandatory Onboarding for "Read/Understand Repo" Requests
Whenever the user asks you to read, understand, or familiarize yourself with the repo (e.g. "đọc hiểu repo", "hiểu codebase", "read repo context"), you MUST ALWAYS immediately read the following 3 foundational files using `view_file` BEFORE answering or taking any further action:
1. `README_CONTEXT.md`: Architecture overview, key scripts, and full history of all resolved issues and fixes.
2. `txt/LIDAR_PAPER_EXTRACTED.txt`: Paper theorems (Theorem 3.1 & 3.3), Table 2 benchmarks, and mathematical formulas.
3. `LiDAR_Table2_Replication_Colab.ipynb`: Official Colab execution workflow, setup, hyperparameter configurations, and evaluation logic.

## Code Integrity & Safe Modification Rules (Strict - Never Violate)
- **Zero Accidental Deletions**: When editing, replacing, or refactoring code or notebooks, **NEVER delete, truncate, or omit surrounding lines** (such as variable/object initializations like `parser = argparse.ArgumentParser()`, imports, working directory switches like `%cd`, or environment configs).
- **Minimal Diff Only**: Apply surgical, minimal modifications strictly to the targeted lines. Do not rewrite surrounding blocks or entire functions/cells when only modifying specific parameters or lines.
- **Mandatory Diff Verification Before Commit**: Always run `git diff` on modified files before staging/committing to verify that no essential lines were unintentionally dropped or wiped out.
- **Python Syntax & Import Sanity Check**: After modifying Python scripts, perform a quick verification (`python -c "import ..."` or syntax check) to prevent runtime `NameError`, missing dependencies, or broken entrypoints.

## Anti-Regression & Notebook Integrity Rules
- **No Brittle In-Place Monkey-Patching in Notebooks**: NEVER write or retain notebook cells that perform blind string/regex replacements (`code.replace(...)` or modifying `.py` files on disk). All bug fixes and architectural changes must be applied cleanly and directly to the repository source files. Notebook cells should only perform verification (`py_compile.compile`), never mutate source code at runtime.
- **Substring Match Collision Prevention**: When writing or refactoring Python code, avoid using fragile expressions that could partially match legacy substring search-and-replace patterns, which can corrupt file indentation and produce fatal runtime `IndentationError`s.

