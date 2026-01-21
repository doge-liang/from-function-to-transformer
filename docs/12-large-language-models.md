# 第十二章：大规模语言模型（LLM）


> 从Transformer到现代AI的核心

---

## 12.1 LLM概述

### 12.1.1 什么是LLM？

**大规模语言模型（LLM）**：在超大规模语料上预训练的Transformer模型

| 模型 | 发布 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-3 | 2020 | 175B | 少样本学习 |
| GPT-4 | 2023 | ~1.8T | 多模态 |
| Claude | 2023 | 100B+ | 长上下文 |
| LLaMA | 2023 | 7B-65B | 开源 |

### 12.1.2 LLM能力

- 文本生成与续写
- 问答与对话
- 代码生成
- 数学推理
- 多语言翻译

---

## 12.2 BERT vs GPT

### 12.2.1 架构对比

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder-only | Decoder-only |
| 注意力 | 双向 | 单向（因果）|
| 训练目标 | MLM + NSP | 自回归 |
| 适用任务 | 理解类任务 | 生成类任务 |

### 12.2.2 使用示例

```python
from transformers import BertModel, GPT2LMHeadModel

# BERT（理解）
bert = BertModel.from_pretrained('bert-base-chinese')
output = bert(**inputs)  # [batch, seq_len, hidden]

# GPT（生成）
gpt = GPT2LMHeadModel.from_pretrained('gpt2')
output = gpt.generate(input_ids, max_length=50)
```


---

## 12.3 预训练

### 12.3.1 数据

- WebText（GPT）
- CommonCrawl（GPT-3）
- The Pile（LLaMA）
- 大规模代码（CodeLLaMA）

### 12.3.2 目标

**Decoder-only**：自回归语言建模

$$\mathcal{L} = -\sum_{i=1}^{T} \log P(x_i \mid x_{<i})$$

**规模效应**：模型越大，效果越好

$$Loss \approx N^{-\alpha}$$

---

## 12.4 指令微调（Instruction Tuning）

### 12.4.1 什么是指令微调？

**指令微调**：用指令-响应对数据微调预训练模型

```

指令："翻译成英文：我喜欢学习"
响应："I like learning"

```

### 12.4.2 Alpaca数据集

```

instruction: "写一个Python的Hello World程序"
input: ""
output: "print('Hello, World!')"

```

### 12.4.3 训练

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data
)

trainer.train()
```


---

## 12.5 对齐

### 12.5.1 RLHF（Reinforcement Learning from Human Feedback）

**三阶段**：

1. 预训练
2. 有监督微调（SFT）
3. RLHF

### 12.5.2 奖励模型（Reward Model）

```python
class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, prompt, response):
        inputs = prompt + response
        outputs = self.model(inputs)
        return outputs.logits  # 奖励分数
```


### 12.5.3 PPO微调

```python
from transformers import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
    train_dataset=dataset
)

ppo_trainer.train()
```


---

## 本章小结

**核心概念**：

1. ✅ LLM：大规模预训练语言模型
2. ✅ BERT：理解任务（Encoder）
3. ✅ GPT：生成任务（Decoder）
4. ✅ 指令微调：遵循人类指令
5. ✅ RLHF：对齐人类价值观

---

## 思考题

1. 为什么LLM需要大规模预训练？
2. BERT和GPT分别适合什么任务？
3. RLHF如何让模型对齐人类偏好？

---

## 下一步

下一章我们将学习**强化学习基础**：

- MDP与策略
- Q-Learning
- 深度Q网络（DQN）

准备好探索智能体的学习方式了吗？
