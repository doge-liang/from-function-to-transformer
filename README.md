# 从函数到 Transformer

> 一步步理解现代深度学习架构

## 项目结构

```
from-function-to-transformer/
├── docs/                       # Markdown 文档（系统学习）
│   ├── 01-from-function-to-neural-network.md
│   ├── 02-deep-neural-networks.md
│   ├── 03-training-deep-networks.md
│   ├── 04-practice-mnist.md
│   ├── 05-convolutional-networks.md
│   ├── 06-recurrent-networks.md
│   ├── 07-attention-mechanisms.md
│   ├── 08-transformer-architecture.md
│   ├── 09-embeddings-and-language-models.md
│   ├── 10-generative-models-vae-gan.md
│   ├── 11-generative-models-diffusion-flow.md
│   ├── 12-large-language-models.md
│   ├── 13-reinforcement-learning.md
│   ├── 14-reinforcement-learning-advanced.md
│   ├── 15-reasoning-enhancement.md
│   ├── 16-multi-agent-systems.md
│   ├── 17-summary-and-next-steps.md
│   ├── archive/              # 旧章节归档
│   │   ├── 01-basics.md
│   │   ├── 02-neural-networks.md
│   │   ├── 02-embeddings.md
│   │   ├── 03-cnn.md
│   │   ├── 04-rnn.md
│   │   ├── 05-generative-models.md
│   │   ├── 05-1-training-basics.md
│   │   ├── 05-2-optimizers.md
│   │   ├── 05-3-initialization.md
│   │   ├── 05-4-model-evaluation.md
│   │   ├── 06-next-steps.md
│   │   ├── 07-reinforcement-learning.md
│   │   ├── 07-chain-of-thought.md
│   │   ├── 08-reasoning-evolution.md
│   │   └── 09-plangen.md
│   └── assets/                # 生成的图表
├── ref/                      # 参考资料（深入数学原理）
│   └── conv.md                 # 卷积的数学原理
├── notebooks/                # Jupyter notebooks（交互式学习）
│   └── from-function-to-transformer.ipynb
├── scripts/                   # Python 工具脚本
│   ├── generate-tikz.py     # TikZ 图表生成
│   └── render-formula.py    # LaTeX 公式渲染
├── charts/                    # 图表源文件
│   ├── tikz/*.tex            # LaTeX/TikZ 图表
│   └── mermaid/*.mmd          # Mermaid 图表
├── CLAUDE.md                  # Claude 使用指南
├── AGENTS.md                  # AI Agent 指南
└── README.md
```

## 学习路线

### 快速入门（1-2周）

```
01-04: 深度学习基础
├─ 函数思维到神经网络
├─ 网络结构与前向传播
├─ 训练方法（损失、优化、初始化）
└─ MNIST 实战项目
```

### 系统学习（8-12周）

```
Part I: 深度学习基础（第1-4周）
├─ 01-from-function-to-neural-network.md
├─ 02-deep-neural-networks.md
├─ 03-training-deep-networks.md
└─ 04-practice-mnist.md

Part II: 模型架构与算法（第5-8周）
├─ 05-convolutional-networks.md
├─ 06-recurrent-networks.md
├─ 07-attention-mechanisms.md
└─ 08-transformer-architecture.md

Part III: 生成式模型与应用（第9-12周）
├─ 09-embeddings-and-language-models.md
├─ 10-generative-models-vae-gan.md
├─ 11-generative-models-diffusion-flow.md
└─ 12-large-language-models.md

Part IV: 前沿与强化学习（第13-17周）
├─ 13-reinforcement-learning.md
├─ 14-reinforcement-learning-advanced.md
├─ 15-reasoning-enhancement.md
├─ 16-multi-agent-systems.md
└─ 17-summary-and-next-steps.md
```

### 计算机视觉方向（6-8周）

```
01-04: 深度学习基础
05-convolutional-networks.md
06-recurrent-networks.md（可选）
10-generative-models-vae-gan.md
11-generative-models-diffusion-flow.md（可选）
```

### 自然语言处理方向（6-8周）

```
01-04: 深度学习基础
06-recurrent-networks.md
07-attention-mechanisms.md
08-transformer-architecture.md
09-embeddings-and-language-models.md
12-large-language-models.md
15-reasoning-enhancement.md
```

### 深度研究（3个月+）

```
完整学习 + 源码阅读 + 论文研读
重点关注: 08-transformer-architecture.md, 12-large-language-models.md
实践项目: 从0到1实现一个LLM
```

## 文档内容

| 章节 | 内容 |
|------|------|
| 01-from-function-to-neural-network.md | 函数思维、线性回归、激活函数 |
| 02-deep-neural-networks.md | 网络结构、前向传播、反向传播 |
| 03-training-deep-networks.md | 损失函数、优化算法、参数初始化、正则化 |
| 04-practice-mnist.md | MNIST 完整实战项目（代码实现） |
| 05-convolutional-networks.md | CNN、卷积核、池化、经典架构 |
| 06-recurrent-networks.md | RNN、LSTM、GRU、双向RNN |
| 07-attention-mechanisms.md | 注意力机制、Self-Attention、多头注意力 |
| 08-transformer-architecture.md | Transformer 完整架构、位置编码、训练技巧 |
| 09-embeddings-and-language-models.md | Word2Vec、GloVe、BERT、GPT |
| 10-generative-models-vae-gan.md | VAE、GAN、训练与生成 |
| 11-generative-models-diffusion-flow.md | Diffusion、Normalizing Flows |
| 12-large-language-models.md | LLM、指令微调、RLHF |
| 13-reinforcement-learning.md | MDP、Q-Learning、DQN、PPO |
| 14-reinforcement-learning-advanced.md | RLHF、奖励模型、PPO 微调 |
| 15-reasoning-enhancement.md | CoT、Self-Consistency、ToT、GoT |
| 16-multi-agent-systems.md | PlanGEN、多智能体协作、任务分解 |
| 17-summary-and-next-steps.md | 全书总结、进阶学习、职业发展 |

## 使用方法

### 文档学习

直接阅读 `docs/` 目录下的 Markdown 文件，支持：

- VS Code + Markdown Preview
- Typora
- GitHub/GitLab 在线预览

### 交互式学习

```bash
# 安装依赖（当使用 notebooks 时）
uv pip install torch matplotlib numpy jupyter

# 启动 Jupyter
jupyter notebook notebooks/from-function-to-transformer.ipynb
```

### 图表生成

```bash
# 生成 TikZ 图表
npm run tikz

# 生成 Mermaid 图表
npm run mermaid

# 渲染 LaTeX 公式为 SVG（用于 Mermaid 图表）
python scripts/render-formula.py "\\mathbf{W}_1"
```

### 训练脚本

```bash
# 生成所有 TikZ 图表
python scripts/generate-tikz.py

# 监听模式（自动重新编译）
python scripts/generate-tikz.py --watch

# 生成单个文件
python scripts/generate-tikz.py attention.tex
```

## 前置依赖

### Python

- Python 3.10+
- 可选：`torch matplotlib numpy jupyter`（用于 notebooks）

### Node.js

```bash
# 安装依赖
npm install

# 关键依赖
- katex
- @mermaid-js/mermaid-cli
```

### 系统工具

- `lualatex` 和 `dvisvgm`（或 `pdf2svg`）用于 TikZ 图表
- `npx` 用于 Node 包执行

## 参考资源

- 《深度学习》（Deep Learning）- Ian Goodfellow
- 《神经网络与深度学习》- Michael Nielsen
- "Attention Is All You Need"（Transformer 原始论文）
- PyTorch 官方教程
- Hugging Face 文档

## 重构说明

本项目已完成重构，主要改进：

1. **知识依赖修复**：训练内容提前到第3章，解决了旧结构中Word2Vec在第2章但梯度下降在第10章的问题
2. **精简文档**：每章控制在 300-500 行，提升阅读体验
3. **清晰分层**：17章节分为4大部分，层次结构清晰
4. **实战项目**：第4章新增 MNIST 完整实战，第5-11章包含实战代码
5. **前沿内容**：覆盖 LLM、RLHF、推理增强、多智能体等最新技术

旧章节已归档到 `docs/archive/` 目录，方便对比查阅。

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎贡献！请参考 CLAUDE.md 和 AGENTS.md 了解项目规范。

---

开始你的深度学习之旅吧！🚀
