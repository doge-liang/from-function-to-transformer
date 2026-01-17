#!/usr/bin/env python3
"""
TikZ 图表生成脚本

Usage:
    python scripts/generate-tikz.py                    # 生成所有
    python scripts/generate-tikz.py --watch            # 监听模式
    python scripts/generate-tikz.py attention.tex      # 生成单个文件
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent


def compile_latex(source: Path, output: Path) -> bool:
    """用 lualatex 编译 TikZ 源码，生成 SVG"""
    assets_dir = output.parent

    try:
        # 第一步：LaTeX -> DVI
        result1 = subprocess.run(
            [
                'lualatex', '-interaction=nonstopmode',
                '-output-format=dvi',
                '-output-directory', str(assets_dir),
                '-jobname', source.stem,
                str(source)
            ],
            capture_output=True,
            timeout=120
        )

        if result1.returncode != 0:
            print(f"  [FAIL] {source.name}: LaTeX 编译错误")
            print(result1.stderr.decode()[:500])
            return False

        # 第二步：DVI -> SVG (使用 dvisvgm)
        dvi_path = assets_dir / f"{source.stem}.dvi"

        # 在输出目录运行 dvisvgm，确保输出文件也在正确位置
        result2 = subprocess.run(
            ['dvisvgm', '--no-fonts', f"{source.stem}.dvi"],
            cwd=str(assets_dir),
            capture_output=True,
            timeout=60
        )

        if result2.returncode == 0:
            # dvisvgm creates {name}.svg in the output directory
            svg_path = assets_dir / f"{source.stem}.svg"
            if svg_path.exists():
                # 如果输出路径不同，重命名
                if svg_path != output:
                    if output.exists():
                        output.unlink()
                    svg_path.rename(output)
                print(f"  [OK] {source.name} -> {output.name}")
                return True
            else:
                print(f"  [FAIL] {source.name}: dvisvgm 未生成 SVG")
                return False
        else:
            print(f"  [FAIL] {source.name}: dvisvgm 错误")
            print(result2.stderr.decode()[:200])
            return False

    except FileNotFoundError as e:
        print(f"  [ERROR] 工具未找到: {e}")
        print("  请安装: lualatex 和 dvisvgm (TeX Live 或 MiKTeX)")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {source.name}: 编译超时")
        return False


def compile_pdf2svg(source: Path, output: Path) -> bool:
    """备选方案：使用 pdf2svg"""
    assets_dir = output.parent

    try:
        # 第一步：LaTeX -> PDF
        result1 = subprocess.run(
            [
                'lualatex', '-interaction=nonstopmode',
                '-output-directory', str(assets_dir),
                '-jobname', source.stem,
                str(source)
            ],
            capture_output=True,
            timeout=120
        )

        if result1.returncode != 0:
            print(f"  [FAIL] {source.name}: LaTeX 编译错误")
            return False

        # 第二步：PDF -> SVG
        pdf_path = assets_dir / f"{source.stem}.pdf"
        if pdf_path.exists():
            result2 = subprocess.run(
                ['pdf2svg', str(pdf_path), str(output)],
                capture_output=True,
                timeout=60
            )
            if result2.returncode == 0:
                print(f"  [OK] {source.name} -> {output.name}")
                return True

        print(f"  [FAIL] {source.name}: PDF 未生成")
        return False

    except FileNotFoundError as e:
        print(f"  [ERROR] 工具未找到: {e}")
        print("  请安装: lualatex 和 pdf2svg")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {source.name}: 编译超时")
        return False


def generate_single(tex_path: Path, assets_dir: Path, use_pdf2svg: bool = False) -> bool:
    """生成单个 TikZ 文件"""
    output = assets_dir / f"{tex_path.stem}.svg"

    if use_pdf2svg:
        return compile_pdf2svg(tex_path, output)
    else:
        return compile_latex(tex_path, output)


def main():
    parser = argparse.ArgumentParser(description='TikZ 图表生成工具')
    parser.add_argument(
        'files', nargs='*', metavar='FILE',
        help='指定要编译的 .tex 文件（默认编译所有）'
    )
    parser.add_argument(
        '--watch', '-w', action='store_true',
        help='监听模式：自动检测文件变化并重新编译'
    )
    parser.add_argument(
        '--pdf2svg', action='store_true',
        help='使用 pdf2svg 替代 dvisvgm（备选方案）'
    )
    args = parser.parse_args()

    root = get_project_root()
    tikz_dir = root / 'charts' / 'tikz'
    assets_dir = root / 'docs' / 'assets'

    # 确保输出目录存在
    assets_dir.mkdir(parents=True, exist_ok=True)

    if args.watch:
        print("监听模式已启动 (Ctrl+C 停止)...")
        print(f"监控目录: {tikz_dir}")
        print("-" * 50)

        try:
            import time
            import hashlib
            last_hashes = {}

            while True:
                tex_files = list(tikz_dir.glob('*.tex'))
                for tex in tex_files:
                    content = tex.read_bytes()
                    h = hashlib.md5(content).hexdigest()
                    if last_hashes.get(str(tex)) != h:
                        last_hashes[str(tex)] = h
                        print(f"\n检测到变化: {tex.name}")
                        generate_single(tex, assets_dir, args.pdf2svg)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n\n已停止。")
    else:
        print("TikZ 图表生成器")
        print("=" * 50)

        if args.files:
            # 编译指定文件
            for filename in args.files:
                tex_path = tikz_dir / filename
                if tex_path.exists():
                    generate_single(tex_path, assets_dir, args.pdf2svg)
                else:
                    print(f"  [SKIP] {filename} 不存在")
        else:
            # 编译所有 .tex 文件
            tex_files = list(tikz_dir.glob('*.tex'))
            if not tex_files:
                print("  未找到 .tex 文件")
                print(f"  请在 {tikz_dir} 目录下创建 TikZ 源文件")
            else:
                print(f"编译 {len(tex_files)} 个文件...")
                for tex in tex_files:
                    generate_single(tex, assets_dir, args.pdf2svg)

        print("=" * 50)
        print(f"输出目录: {assets_dir}")


if __name__ == '__main__':
    main()
