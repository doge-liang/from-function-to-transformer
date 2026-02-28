# AGENTS.md

This file guides agentic coding agents working in this repository.

## Project Overview

**From Function to Transformer** - A learning project explaining deep learning architectures through structured documentation and interactive notebooks.

**Type**: Documentation/Learning project | **Languages**: Python, Markdown | **Purpose**: Educational content and diagram generation

## Commands

### Diagram Generation
```bash
npm run tikz                    # Generate all TikZ diagrams
npm run tikz:watch              # Watch mode
npm run tikz:pdf2svg            # Alternative using pdf2svg
npm run mermaid                 # Generate all Mermaid diagrams
npm run mermaid:watch           # Watch mode
```

### Python Scripts
```bash
python scripts/generate-tikz.py                 # Generate all
python scripts/generate-tikz.py --watch         # Watch mode
python scripts/generate-tikz.py attention.tex   # Single file
```

### Notebooks & Linting
```bash
uv pip install torch matplotlib numpy jupyter
jupyter notebook notebooks/from-function-to-transformer.ipynb
npm run lint:md            # Check all markdown
npm run lint:md:fix        # Auto-fix markdown issues
npx markdownlint-cli2 docs/01-*.md  # Check single file
```

**Testing**: No formal test suite. Verify visual outputs manually.
**Single File Tests**: Generate specific diagram: `python scripts/generate-tikz.py attention.tex`

## Code Style Guidelines

### Python Scripts

- Shebang: `#!/usr/bin/env python3` with module docstring
- Imports: stdlib grouped (third-party/local not present)
- Type hints required: `def func(x: int) -> bool` (Python 3.10+ union syntax available if needed)
- Constants: `UPPER_CASE` | Functions: `snake_case` | Classes: `PascalCase`

```python
import argparse
import subprocess
from pathlib import Path

# Error handling pattern
try:
    result = subprocess.run([...], capture_output=True, timeout=60)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
except subprocess.TimeoutExpired:
    print("Timeout")
    return False
except FileNotFoundError as e:
    print(f"Tool not found: {e}")
    return False
```

- Use `pathlib.Path` for file operations (not `os.path`)
- `Path(__file__).resolve().parent.parent` for project root
- Use `subprocess.run()` with `capture_output=True` and `timeout`
- CLI args: `argparse.ArgumentParser()`, `nargs='*'`, `action='store_true'`
- Logging: `print()` with prefixes `[OK]`, `[FAIL]`, `[ERROR]`, `[TIMEOUT]`
- Return `bool` for success/failure in utility functions

### Markdown Documentation

- Simple flowcharts: Mermaid in `charts/mermaid/*.mmd`
- Complex architectures: TikZ in `charts/tikz/*.tex`
- Reference: `![](docs/assets/{filename}.svg)`
- Math: Inline `$E = mc^2$`, Display `$$E = mc^2$$`

### File Organization
```plaintext
scripts/         # Python utilities (generate-tikz.py)
charts/          # TikZ (.tex) and Mermaid (.mmd) sources
docs/            # Markdown docs + generated assets/*.svg
notebooks/       # Jupyter notebooks (.ipynb)
```

## Dependencies

**Python**: Standard library only (scripts) | Optional: `torch matplotlib numpy jupyter`
**Node.js**: `@mermaid-js/mermaid-cli`, `markdownlint-cli2`
**System**: `lualatex` + `dvisvgm` (or `pdf2svg`) for TikZ, `npx`

## Tooling

**Linting**: `markdownlint-cli2` configured (Python: none)
**Testing**: No test framework
**Formatting**: None
**CI/CD**: None

**Markdown Linting Rules** (.markdownlint-cli2.cjs):
- MD013/MD024/MD029/MD033/MD036/MD040/MD041/MD045 disabled (flexible formatting)
- Math formulas, code examples, tables allow long lines
- Duplicate headers allowed across different sections
- Code blocks without language spec allowed (ASCII art, diagrams)

**Verification**: Run scripts, check console output, inspect SVGs visually, review Markdown rendering

## Naming Conventions

Files: `kebab-case.py`/`kebab-case.md` | Functions: `snake_case` | Variables: `snake_case`
Constants: `UPPER_CASE` | Classes: `PascalCase` | Diagrams: `lowercase.svg`

## Common Patterns

```python
# Project root
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

# Command wrapper
def run_command(cmd: list[str], timeout: int = 60) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0, result.stderr.decode()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
```

## Adding New Content

**New Diagram**: Create source → Run generation → Verify in `docs/assets/` → Reference in Markdown
**New Documentation**: Create `.md` → Follow numbered structure → Use consistent headings → Add diagrams → Test rendering

## Before Committing

1. Run generation commands for modified diagrams
2. `npm run lint:md` (fix with `npm run lint:md:fix`)
3. Verify SVGs display correctly
4. Check Markdown rendering
5. Ensure all new SVGs in `docs/assets/`
6. No temporary LaTeX files (in .gitignore)

7. **删除 `nul` 文件**: `find . -name "nul" -type f -delete` (Windows 保留设备名，无法提交)

## Pre-commit Checks (Additional)

Windows 用户注意：如果系统生成 `nul` 文件（Windows 保留设备名），请务必删除：

```bash
find . -name "nul" -type f -delete 2>/dev/null || true
```