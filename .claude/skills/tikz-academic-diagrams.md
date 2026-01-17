# TikZ 学术图表绘制规范

## 何时使用 TikZ

当图表满足以下任一条件时，使用 TikZ 而非 Mermaid：
1. 包含数学公式（LaTeX 公式）
2. 需要精确控制元素位置和对齐
3. 需要学术级别的图表质量
4. 需要复杂的矩阵/向量布局
5. 元素之间容易交叉重叠

## TikZ 图表模板

```latex
% 图表名称
% Generated with TikZ

\documentclass[tikz,border=8pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,fit,backgrounds}
\usepackage{amsmath}

\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm,
    font=\small,
    % 节点样式定义
    input/.style={draw, rounded corners=2pt, minimum width=1.2cm, minimum height=1cm, fill=gray!10},
    op/.style={draw, rounded corners=4pt, minimum width=1.8cm, minimum height=1.1cm, fill=blue!8},
    output/.style={draw, rounded corners=4pt, minimum width=2.2cm, minimum height=1.1cm, fill=green!10},
    arrow/.style={->, thick, draw=black!70},
]

    % 节点定义
    \node[input] (node1) {$X$};
    \node[op, right=of node1] (op1) {$W_1$};
    \node[output, right=of op1] (out1) {$H$};

    % 连接线
    \draw[arrow] (node1) -- (op1);
    \draw[arrow] (op1) -- (out1);

\end{tikzpicture}
\end{document}
```

## 美观原则

### 1. 元素排列
- **避免交叉**：使用 `node distance` 控制间距，必要时手动调整
- **水平/垂直对齐**：使用 `above`, `below`, `left`, `right` 或 `=of` 定位
- **矩阵布局**：复杂结构使用 `\matrix[column sep=...]` 排列多个节点

### 2. 样式规范
- 节点圆角：`rounded corners=2-4pt`
- 填充色：`fill=color!10`（淡色背景）
- 箭头粗细：`thick` 或 `semithick`
- 字体大小：`\small` 或 `\footnotesize`

### 3. 公式处理
- 简短公式直接放在节点内：`{$W$}`
- 复杂公式使用 `align=center` 多行显示
- 大型公式在节点外单独标注

### 4. 层次结构
```latex
% 推荐：使用背景层分组
\begin{pgfonlayer}{background}
    \node[draw, rounded corners=6pt, fill=blue!4, fit=(node1)(node2), inner sep=5pt] {};
\end{pgfonlayer}
```

## 常用定位模式

| 模式 | 示例 | 说明 |
|------|------|------|
| 水平排列 | `\node[op, right=1.5cm of A]` | 右侧 1.5cm |
| 垂直排列 | `\node[op, below=1cm of A]` | 下方 1cm |
| 矩阵 | `\matrix[column sep=1cm]` | 多列等距 |
| 相对坐标 | `\draw (A.south) -- (B.north)` | 精确连接 |

## 生成命令

```bash
# 生成所有 TikZ 图表
npm run tikz

# 或手动编译
lualatex --output-format=dvi charts/tikz/xxx.tex
dvisvgm xxx.dvi --font-format=woff
```

## 文件命名

- 源文件：`charts/tikz/{图表名}.tex`
- 输出：`docs/assets/{图表名}.svg`

## 示例：自注意力机制

参见 `charts/tikz/attention.tex`，展示：
- 矩阵布局（Q, K, V 并排）
- 公式嵌入（$\frac{QK^T}{\sqrt{d_k}}$）
- 分组标注（Score & scale, Mask, Normalize）
- 层次化背景框
