# 代码库结构

```
from-function-to-transformer/
├── scripts/                    # Python 工具脚本
│   └── generate-tikz.py        # TikZ 图表生成器
│
├── charts/                     # 图表源文件
│   ├── tikz/                   # LaTeX/TikZ (复杂架构图)
│   │   ├── attention.tex
│   │   ├── transformer.tex
│   │   ├── forward-prapagation.tex
│   │   ├── forward-two-layer-demo.tex
│   │   └── single-neural-unit.tex
│   └── mermaid/                # Mermaid (简单流程图)
│       ├── neural-network.mmd
│       └── training-loop.mmd
│
├── docs/                       # 文档
│   ├── 01-from-function-to-neural-network.md
│   ├── 02-deep-neural-networks.md
│   ├── ... (共 17 章)
│   ├── 17-summary-and-next-steps.md
│   ├── archive/                # 旧章节归档
│   └── assets/                 # 生成的 SVG (勿编辑)
│       ├── attention.svg
│       ├── transformer.svg
│       └── ...
│
├── notebooks/                  # Jupyter notebooks
│   └── from-function-to-transformer.ipynb
│
├── AGENTS.md                   # AI Agent 指南
├── CLAUDE.md                   # Claude 使用指南
├── README.md                   # 项目说明
├── package.json                # npm 脚本
└── .markdownlint-cli2.cjs      # Markdown lint 配置
```

## 重要目录说明

| 目录 | 用途 | 可编辑 |
|------|------|--------|
| scripts/ | Python 工具 | ✅ |
| charts/tikz/ | TikZ 源文件 | ✅ |
| charts/mermaid/ | Mermaid 源文件 | ✅ |
| docs/*.md | 文档章节 | ✅ |
| docs/assets/ | 生成的 SVG | ❌ 自动生成 |
| docs/archive/ | 归档文档 | ⚠️ 谨慎修改 |
