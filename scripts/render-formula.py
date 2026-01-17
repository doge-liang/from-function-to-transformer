#!/usr/bin/env python3
"""
Render LaTeX formulas to SVG files for Mermaid diagrams using KaTeX.

Usage:
    python scripts/render-formula.py "\\mathbf{W}_1"
    python scripts/render-formula.py "\\frac{1}{2}" --display -o custom.svg
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# Paths and defaults
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / 'docs' / 'assets'
KATEX_CSS_PATH = ROOT_DIR / 'node_modules' / 'katex' / 'dist' / 'katex.min.css'
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT_INLINE = 96
DEFAULT_HEIGHT_DISPLAY = 140
SVG_PADDING = 8


def compute_hash(formula: str) -> str:
    """Generate a short hash from the formula string."""
    return hashlib.md5(formula.encode('utf-8')).hexdigest()[:8]


def ensure_svg_extension(path: str) -> str:
    """Ensure the file path has a .svg extension."""
    if not path.endswith('.svg'):
        path = path.rsplit('.', 1)[0] + '.svg'
    return path


def load_katex_css() -> str:
    """Load KaTeX CSS so the SVG contains all required styles."""
    if KATEX_CSS_PATH.exists():
        return KATEX_CSS_PATH.read_text(encoding='utf-8')

    print(f"Warning: KaTeX CSS not found at {KATEX_CSS_PATH}")
    return ""


def estimate_size(formula: str, display_mode: bool) -> tuple[int, int]:
    """Heuristically estimate SVG canvas size based on formula length."""
    width = max(DEFAULT_WIDTH, min(1600, len(formula) * (14 if display_mode else 12) + 80))
    height = DEFAULT_HEIGHT_DISPLAY if display_mode else DEFAULT_HEIGHT_INLINE
    return width, height


def wrap_svg(html: str, css: str, width: int, height: int, padding: int = SVG_PADDING) -> str:
    """Wrap KaTeX HTML markup inside an SVG foreignObject."""
    return dedent(f"""\
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <style>{css}</style>
      <foreignObject x="0" y="0" width="100%" height="100%">
        <div xmlns="http://www.w3.org/1999/xhtml" style="display:inline-block;padding:{padding}px;">
          {html}
        </div>
      </foreignObject>
    </svg>
    """)


def render_with_katex(formula: str, display_mode: bool) -> str | None:
    """Render LaTeX formula to KaTeX HTML via CLI."""
    try:
        result = subprocess.run(
            ['npx', 'katex', '--format', 'html', '--no-throw-on-error']
            + (['--display-mode'] if display_mode else []),
            input=formula,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT_DIR,
        )

        if result.returncode != 0:
            print(f"KaTeX error: {result.stderr}")
            return None

        html = result.stdout.strip()
        if not html:
            print("KaTeX CLI returned empty output")
            return None

        return html

    except subprocess.TimeoutExpired:
        print("Error: KaTeX rendering timed out")
        return None
    except FileNotFoundError:
        print("Error: npx/katex not found. Please install KaTeX CLI.")
        return None
    except Exception as e:
        print(f"Error rendering formula: {e}")
        return None


def create_fallback_svg(formula: str, output_path: Path, width: int, height: int) -> bool:
    """Create a fallback SVG with plain text when KaTeX is unavailable."""
    escaped_formula = formula.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <text x="10" y="{height // 2}" font-family="Arial" font-size="20" dominant-baseline="middle">
    {escaped_formula}
  </text>
</svg>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding='utf-8')

    print(f"Created fallback SVG: {output_path}")
    return True


def render_to_svg(formula: str, output_path: Path, display_mode: bool, width: int | None, height: int | None) -> bool:
    """Render LaTeX formula to SVG using KaTeX."""
    # Ensure output has .svg extension
    output_path = Path(ensure_svg_extension(str(output_path)))

    html = render_with_katex(formula, display_mode)
    auto_width, auto_height = estimate_size(formula, display_mode)
    width = width or auto_width
    height = height or auto_height

    if html:
        css = load_katex_css()
        svg_content = wrap_svg(html, css, width, height)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(svg_content, encoding='utf-8')
        return True

    print("KaTeX unavailable, using fallback SVG")
    return create_fallback_svg(formula, output_path, width, height)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Render LaTeX formulas to SVG files for Mermaid diagrams using KaTeX.'
    )
    parser.add_argument('formula', help='LaTeX formula (use quotes for special chars)')
    parser.add_argument('-o', '--output', help='Output file path (default: auto-generated)')
    parser.add_argument('-d', '--dir', help='Output directory (default: docs/assets)')
    parser.add_argument('--display', action='store_true', help='Use KaTeX display mode (larger math)')
    parser.add_argument('--width', type=int, help='SVG width in px (auto-estimated by default)')
    parser.add_argument('--height', type=int, help='SVG height in px (auto-estimated by default)')
    parser.add_argument('-f', '--force', action='store_true', help='Force re-render even if the file exists')

    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.dir) if args.dir else ASSETS_DIR

    # Generate hash for the formula
    formula_hash = compute_hash(args.formula)

    # Determine output path (use .svg extension)
    if args.output:
        output_path = output_dir / args.output
    else:
        output_path = output_dir / f'{formula_hash}.svg'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists (cache)
    if output_path.exists() and not args.force:
        print(f"Formula already rendered: {output_path}")
        print(f"Hash: {formula_hash}")
        return

    # Render the formula
    print(f"Rendering formula: {args.formula}")
    print(f"Output path: {output_path}")

    if render_to_svg(args.formula, output_path, args.display, args.width, args.height):
        print(f"Success: {output_path}")
        print(f"Hash: {formula_hash}")
    else:
        print("Failed to render formula")
        sys.exit(1)


if __name__ == '__main__':
    main()
