# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a learning project under the AI Learning project hub.

## Project Type

- Language: Detect from project files
- Purpose: Learning/Practice

## Commands

No specific commands configured yet.

## TikZ 图表处理

### 目录结构

```plaintext
charts/tikz/*.tex          # TikZ 源文件
docs/assets/*.svg          # 生成的 SVG 文件
```

### 命令

```bash
# 生成所有 TikZ 图表
npm run tikz

# 监听模式（自动重新编译）
npm run tikz:watch

# 使用 pdf2svg 备选方案
npm run tikz:pdf2svg
```

### 前置依赖

- **LaTeX 发行版** (lualatex + dvisvgm)
  - Windows: MiKTeX (<https://miktex.org/>)
  - Mac: TeX Live via Homebrew (`brew install --cask mactex`)
  - Linux: `sudo apt install texlive-latex-extra lualatex dvisvgm`

### 添加新图表

1. 在 `charts/tikz/` 创建 `.tex` 文件
2. 运行 `npm run tikz` 生成 SVG
3. 在文档中引用：`![](docs/assets/{图表名}.svg)`

## Mermaid 图表处理

### 工具定位

```
┌─────────────────────────────────────────────────────────┐
│  简单流程图 → Mermaid（代码驱动，易维护）                │
│  复杂架构图 → TikZ（公式完美集成，学术标准）             │
│  数据图表   → Matplotlib（数据驱动）                     │
└─────────────────────────────────────────────────────────┘
```

### 目录结构

```plaintext
charts/mermaid/*.mmd      # Mermaid 源文件
docs/assets/*.svg         # 生成的 SVG 文件
```

### 命令

```bash
# 生成所有 Mermaid 图表
npm run mermaid

# 监听模式（自动重新编译）
npm run mermaid:watch
```

### 前置依赖

- **Mermaid CLI**: `npm install --save-dev @mermaid-js/mermaid-cli`

### 添加新图表

1. 在 `charts/mermaid/` 创建 `.mmd` 文件
2. 运行 `npm run mermaid` 生成 SVG
3. 在文档中引用：`![](docs/assets/{图表名}.svg)`
