# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is a fresh scaffold with no source code yet — only `.gitignore` and `requirements.txt` exist. There is no README, package structure, or architecture to document at this point. When source code is added, this file should be updated with the real module layout and workflows.

## Environment

- Python 3.11 (`.venv/`, not committed — see `.gitignore`)
- Dependencies are pinned in `requirements.txt` and include: numpy, pandas, scipy, pydantic, networkx, pytest, mypy, ruff.

Note: `requirements.txt` is currently saved as UTF-16LE with CRLF line endings, not the usual UTF-8. Tools that assume UTF-8 (including `pip install -r`) may fail to parse it — re-save as UTF-8 if you hit encoding errors installing from it.

Activate the environment:
```
.venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Tooling

The pinned dependencies indicate the intended toolchain, though no config files (e.g. `pyproject.toml`, `pytest.ini`, `ruff.toml`) exist yet to customize them:
- Tests: `pytest`
- Lint: `ruff check .`
- Type checking: `mypy .`
