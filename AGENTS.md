# AGENTS.md

This file guides agentic coding agents working in this repository.

## Project Overview

**From Function to Transformer** - A learning project explaining deep learning architectures through structured documentation and interactive notebooks.

**Type**: Documentation/Learning project
**Languages**: Python (scripts), Markdown (docs)
**Purpose**: Educational content creation and diagram generation

## Commands

### Diagram Generation

```bash
# Generate TikZ diagrams (LaTeX-based)
npm run tikz                    # Generate all TikZ diagrams
npm run tikz:watch              # Watch mode for auto-recompilation
npm run tikz:pdf2svg            # Alternative using pdf2svg

# Generate Mermaid diagrams
npm run mermaid                 # Generate all Mermaid diagrams
npm run mermaid:watch           # Watch mode for auto-recompilation
```

### Python Scripts

```bash
# Run diagram generation scripts directly
python scripts/generate-tikz.py
python scripts/generate-tikz.py --watch
python scripts/generate-tikz.py attention.tex  # Single file
```

### Notebooks

```bash
# Install dependencies (when using notebooks)
uv pip install torch matplotlib numpy jupyter

# Start Jupyter
jupyter notebook notebooks/from-function-to-transformer.ipynb
```

**Testing**: This project has no formal test suite. Verify visual outputs manually.

## Code Style Guidelines

### Python Scripts

**File Structure**:
- Shebang line: `#!/usr/bin/env python3`
- Module docstring with usage examples
- Imports grouped: stdlib → third-party → local
- Type hints required (Python 3.10+ `|` syntax)
- Constants in `UPPER_CASE`
- Functions use `snake_case`
- Classes use `PascalCase`

**Imports**:
```python
import argparse
import subprocess
import sys
from pathlib import Path
```

**Type Hints**:
```python
def render_to_svg(formula: str, output_path: Path, display_mode: bool, width: int | None, height: int | None) -> bool:
```

**Error Handling**:
```python
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

**Path Operations**:
- Use `pathlib.Path` for all file operations
- `Path(__file__).resolve().parent.parent` to get project root
- `path.mkdir(parents=True, exist_ok=True)` for directory creation
- `path.write_text(content, encoding='utf-8')` for file writing

**CLI Arguments**:
```python
import argparse

parser = argparse.ArgumentParser(description='...')
parser.add_argument('files', nargs='*', metavar='FILE', help='...')
parser.add_argument('--watch', '-w', action='store_true', help='...')
args = parser.parse_args()
```

**Subprocess Execution**:
- Use `subprocess.run()` with `capture_output=True` and `timeout`
- Check `returncode` and decode stderr/stdout
- Provide clear error messages

**Logging**:
- Use `print()` for user-facing output (not logging module)
- Format: `[OK] file -> output` or `[FAIL] file: error message`
- Progress indicators for batch operations

### Markdown Documentation

**Diagrams**:
- Simple flowcharts: Mermaid in `charts/mermaid/*.mmd`
- Complex architectures: TikZ in `charts/tikz/*.tex`
- Reference: `![](docs/assets/{filename}.svg)`

**Math Notation**:
- Inline: `$E = mc^2$`
- Display: `$$E = mc^2$$`

### File Organization

```
scripts/              # Python utility scripts
  generate-tikz.py    # TikZ diagram generation

charts/               # Diagram source files
  tikz/*.tex         # LaTeX/TikZ diagrams
  mermaid/*.mmd      # Mermaid diagrams

docs/                 # Documentation
  assets/*.svg       # Generated diagrams
  *.md              # Markdown docs

notebooks/            # Jupyter notebooks
  *.ipynb
```

## Dependencies

**Python**:
- Standard library only (scripts are self-contained)
- Optional: `torch matplotlib numpy` for notebooks

**Node.js**:
- `@mermaid-js/mermaid-cli` - Mermaid diagrams

**System Tools**:
- `lualatex` and `dvisvgm` (or `pdf2svg`) for TikZ
- `npx` for Node package execution

## No Formal Tooling

**Linting**: No linters configured (ruff, pylint, black, etc.)
**Testing**: No test framework (pytest, unittest)
**Formatting**: No auto-formatting tools
**CI/CD**: No GitHub Actions or similar

**Manual Verification**:
- Run scripts and check console output
- Inspect generated SVGs visually
- Review Markdown rendering in VS Code/Typora/GitHub

## Naming Conventions

- **Files**: `kebab-case.py` (scripts), `kebab-case.md` (docs)
- **Functions**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Classes**: `PascalCase`
- **Diagram files**: Descriptive names in lowercase (e.g., `attention.svg`, `neural-network.svg`)

## Common Patterns

**Project Root Resolution**:
```python
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
```

**Command Execution Wrapper**:
```python
def run_command(cmd: list[str], timeout: int = 60) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0, result.stderr.decode()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
```

**Output Formatting**:
- Clear status prefixes: `[OK]`, `[FAIL]`, `[ERROR]`, `[TIMEOUT]`
- Include source and destination in success messages
- Truncate error output to console spam

## Adding New Content

**New Diagram**:
1. Create source file in `charts/tikz/` or `charts/mermaid/`
2. Run generation command
3. Verify output in `docs/assets/`
4. Reference in Markdown using appropriate format

**New Documentation**:
1. Create `.md` file in `docs/`
2. Follow existing structure with numbered chapters
3. Use consistent heading levels (##, ###)
4. Add diagrams where helpful
5. Test rendering in Markdown preview

## Before Committing

1. Run generation commands for all modified diagrams
2. Verify generated SVGs display correctly
3. Check Markdown rendering
4. Ensure all new SVGs are in `docs/assets/`
5. No temporary LaTeX files (should be in .gitignore)
