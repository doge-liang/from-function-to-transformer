# from-function-to-transformer

从函数到 Transformer：一步步理解现代深度学习架构

## 项目结构

```
from-function-to-transformer/
├── docs/                       # Markdown 文档（系统学习）
│   ├── 01-basics.md            # 基础概念（函数、两种思维范式、概率论基础）
│   ├── 02-neural-networks.md   # 神经网络结构
│   ├── 02-embeddings.md        # 词嵌入与表示学习
│   ├── 03-cnn.md               # 卷积神经网络
│   ├── 04-rnn.md               # 循环神经网络
│   ├── 05-generative-models.md # 生成模型（VAE, GAN, Diffusion, Flow）
│   ├── 05-1-training-basics.md # 训练过程（基础篇）
│   ├── 05-2-optimizers.md      # 优化器
│   ├── 05-3-initialization.md  # 参数初始化
│   ├── 05-4-model-evaluation.md # 模型评估与调优
│   ├── 06-next-steps.md        # Transformer 前置知识与总结
│   ├── 07-chain-of-thought.md  # Chain-of-Thought 思维链推理
│   ├── 07-reinforcement-learning.md # 强化学习
│   ├── 08-reasoning-evolution.md # 推理增强技术演进
│   └── 09-plangen.md           # PlanGEN 多智能体框架
├── ref/                        # 参考资料（深入数学原理）
│   └── conv.md                 # 卷积的数学原理
├── notebooks/                  # Jupyter notebooks（交互式学习）
│   └── from-function-to-transformer.ipynb
└── README.md
```

## 学习路线

```
docs/01-basics.md              → 基础概念（函数、两种思维范式、概率论基础）
     ↓
docs/02-neural-networks.md     → 神经网络结构（层次结构、前向传播）
     ↓
docs/02-embeddings.md          → 词嵌入与表示学习
     ↓
docs/03-cnn.md                 → 卷积神经网络（CNN）
     ↓
docs/04-rnn.md                 → 循环神经网络（RNN）
     ↓
docs/05-generative-models.md   → 生成模型（VAE, GAN, Diffusion, Flow）
     ↓
docs/05-1-training-basics.md   → 训练过程（基础篇）：损失函数、梯度下降、反向传播
     ↓
docs/05-2-optimizers.md        → 优化器：SGD, Adam, AdamW, LAMB（含论文引用）
     ↓
docs/05-3-initialization.md    → 参数初始化：Xavier、He、PyTorch 实现
     ↓
docs/05-4-model-evaluation.md  → 模型评估：过拟合、正则化、Batch Size、超参数
     ↓
docs/07-reinforcement-learning.md → 强化学习：MDP, DQN, PPO, RLHF
     ↓
docs/06-next-steps.md          → Transformer 前置知识与总结
     ↓
docs/07-chain-of-thought.md    → Chain-of-Thought：思维链推理基础
     ↓
docs/08-reasoning-evolution.md → 推理增强技术演进：Self-Consistency、ToT、GoT
     ↓
docs/09-plangen.md             → PlanGEN：多智能体推理框架
```

## 文档内容

| 文件 | 章节 | 内容 |
|------|------|------|
| 01-basics.md | 1 | 用函数描述世界、符号主义 vs 连接主义、激活函数、Softmax、N-gram、概率论基础 |
| 02-neural-networks.md | 2 | 网络层次结构、前向传播 |
| 02-embeddings.md | 2.x | 词嵌入、Word2Vec、潜空间 |
| 03-cnn.md | 3.x | 卷积神经网络、卷积核、池化 |
| 04-rnn.md | 4.x | 循环神经网络、LSTM |
| 05-generative-models.md | 5.x | VAE、GAN、Diffusion、Flow |
| 05-1-training-basics.md | 5.1 | 损失函数、梯度下降、反向传播、完整训练流程 |
| 05-2-optimizers.md | 5.2 | SGD, Momentum, NAG, Adam, AdamW, LAMB（含论文引用） |
| 05-3-initialization.md | 5.3 | Xavier、He 初始化、PyTorch 实现 |
| 05-4-model-evaluation.md | 5.4 | 过拟合、正则化、Batch Size、超参数 |
| 06-next-steps.md | 6 | Transformer 前置知识、学习路线图 |
| 07-chain-of-thought.md | 7 | Chain-of-Thought：Zero-shot/Few-shot CoT、工作原理、实践指南 |
| 07-reinforcement-learning.md | 7.x | MDP, Q-Learning, DQN, PPO, RLHF |
| 08-reasoning-evolution.md | 8 | Self-Consistency、Tree-of-Thought、Graph-of-Thought、o1-like模型 |
| 09-plangen.md | 9 | PlanGEN框架：约束/验证/选择智能体、增强算法 |
| ref/conv.md | 参考 | 卷积的数学原理详解 |

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
