# from-function-to-transformer

从函数到 Transformer：一步步理解现代深度学习架构

## 项目结构

```
from-function-to-transformer/
├── docs/                       # Markdown 文档（系统学习）
│   ├── 01-basics.md            # 基础概念
│   ├── 02-neural-networks.md   # 神经网络结构
│   ├── 03-1-training-basics.md # 训练过程（基础篇）
│   ├── 03-2-optimizers.md      # 优化器
│   ├── 03-3-model-evaluation.md # 模型评估与调优
│   ├── 03-4-initialization.md  # 参数初始化
│   └── 04-next-steps.md        # 下一步与总结
├── notebooks/                  # Jupyter notebooks（交互式学习）
│   └── from-function-to-transformer.ipynb
└── README.md
```

## 学习路线

```
docs/01-basics.md              → 基础概念（函数、两种思维范式、激活函数）
     ↓
docs/02-neural-networks.md     → 神经网络结构（层次结构、前向传播）
     ↓
docs/03-1-training-basics.md   → 训练过程（基础篇）：损失函数、梯度下降、反向传播
     ↓
docs/03-2-optimizers.md        → 优化器：SGD, Adam, AdamW, LAMB（含论文引用）
     ↓
docs/03-3-model-evaluation.md  → 模型评估：过拟合、正则化、Batch Size、超参数
     ↓
docs/03-4-initialization.md    → 参数初始化：Xavier、He、PyTorch 实现
     ↓
docs/04-next-steps.md          → Transformer 前置知识与总结
```

## 文档内容

| 文件 | 章节 | 内容 |
|------|------|------|
| 01-basics.md | 1 | 用函数描述世界、符号主义 vs 连接主义、激活函数 |
| 02-neural-networks.md | 2 | 网络层次结构、前向传播、代码实现 |
| 03-1-training-basics.md | 3.1-3.4 | 损失函数、梯度下降、反向传播、完整训练流程 |
| 03-2-optimizers.md | 3.2-3.3 | SGD, Momentum, NAG, Adam, AdamW, LAMB（含论文引用） |
| 03-3-model-evaluation.md | 3.4-3.5 | 过拟合、正则化、Batch Size、超参数 |
| 03-4-initialization.md | 3.6 | Xavier、He 初始化、PyTorch 实现 |
| 04-next-steps.md | 4 | Transformer 前置知识、学习路线图 |

## 使用方法

### 文档学习

直接阅读 `docs/` 目录下的 Markdown 文件，支持：
- VS Code + Markdown Preview
- Typora
- GitHub/GitLab 在线预览

### 交互式学习

```bash
# 安装依赖
uv pip install torch matplotlib numpy

# 打开 Jupyter Notebook
jupyter notebook notebooks/from-function-to-transformer.ipynb
```

## 参考资源

- 《深度学习》（Deep Learning）- Ian Goodfellow
- 《神经网络与深度学习》- Michael Nielsen
- "Attention Is All You Need"（Transformer 原始论文）
- PyTorch 官方教程
