# 常用命令

## 图表生成
```bash
# TikZ 图表
npm run tikz                    # 生成所有
npm run tikz:watch              # 监听模式
npm run tikz:pdf2svg            # 使用 pdf2svg 备选方案

# 单个 TikZ 文件
python scripts/generate-tikz.py attention.tex
python scripts/generate-tikz.py --pdf2svg attention.tex

# Mermaid 图表
npm run mermaid                 # 生成所有
npm run mermaid:watch           # 监听模式
```

## Markdown Lint
```bash
npm run lint:md                 # 检查所有
npm run lint:md:fix             # 自动修复
npx markdownlint-cli2 docs/01-*.md  # 检查单个文件
```

## Jupyter Notebook
```bash
uv pip install torch matplotlib numpy jupyter
jupyter notebook notebooks/from-function-to-transformer.ipynb
```

## Git 命令
```bash
git status
git add .
git commit -m "message"
git push
```

## 系统工具 (Linux)
```bash
ls -la                          # 列出文件
grep -r "pattern" docs/         # 搜索内容
find . -name "*.tex"            # 查找文件
```
