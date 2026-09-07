# Agent Instructions for Diffusion-LiDAR-Sampling

## Git Workflow Rule
- Whenever performing Git operations (staging, committing, and pushing), **always combine `git add`, `git commit`, and `git push` into a single command line** (e.g. `git add . ; git commit -m "<message>" ; git push`) to minimize round-trips and save execution time.

## Project Context
- **Repository Overview & Fixes History**: Refer to `README_CONTEXT.md` (kept in `.gitignore`) for a summary of the project architecture, key scripts (`lookahead_sampling.py`, `LiDAR_sampling.py`), and past resolved bugs/fixes.
- Always maintain and update `README_CONTEXT.md` with newly resolved issues or codebase updates.

## Code Integrity & Safe Modification Rules (Strict - Never Violate)
- **Zero Accidental Deletions**: When editing, replacing, or refactoring code or notebooks, **NEVER delete, truncate, or omit surrounding lines** (such as variable/object initializations like `parser = argparse.ArgumentParser()`, imports, working directory switches like `%cd`, or environment configs).
- **Minimal Diff Only**: Apply surgical, minimal modifications strictly to the targeted lines. Do not rewrite surrounding blocks or entire functions/cells when only modifying specific parameters or lines.
- **Mandatory Diff Verification Before Commit**: Always run `git diff` on modified files before staging/committing to verify that no essential lines were unintentionally dropped or wiped out.
- **Python Syntax & Import Sanity Check**: After modifying Python scripts, perform a quick verification (`python -c "import ..."` or syntax check) to prevent runtime `NameError`, missing dependencies, or broken entrypoints.
