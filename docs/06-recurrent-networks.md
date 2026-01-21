# 第六章：循环神经网络（RNN）


> 处理序列数据的神经网络

---

## 6.1 为什么需要RNN？

### 6.1.1 序列数据的特点

**序列数据**：数据点之间有顺序依赖关系

```

文本：    "我" "爱" "学" "习"
          ↑    ↑    ↑    ↑
        依赖 依赖 依赖


时间序列：昨天的价格影响今天的价格
```


### 6.1.2 CNN的局限

CNN处理图像很出色，但处理序列数据有问题：

| 问题 | 原因 |
|------|------|
| 忽略顺序 | CNN不关心输入顺序 |
| 无法记忆 | 无法记住之前的输入 |
| 变长处理难 | CNN需要固定输入尺寸 |

### 6.1.3 RNN核心思想

**记忆之前的输入，影响当前输出**

```

文本处理：
"我 爱"  → RNN记住"爱" → 处理"学" → 理解是"爱学习"

```

**应用领域**：

- 机器翻译
- 文本生成
- 语音识别
- 时间序列预测

---

## 6.2 RNN结构

### 6.2.1 展开图

```mermaid
graph LR
    subgraph t=1
    x1[x1] --> h1[h1]
    h1 --> o1[o1]
    end

    subgraph t=2
    x2[x2] --> h2[h2]
    h2 --> o2[o2]
    h1 -.-> h2
    end

    subgraph t=3
    x3[x3] --> h3[h3]
    h3 --> o3[o3]
    h2 -.-> h3
    end

    style h1 fill:#c8e6c9
    style h2 fill:#c8e6c9
    style h3 fill:#c8e6c9
```


**关键**：隐藏状态 $h_t$ 传递到下一个时间步

### 6.2.2 数学公式

$$h_t = \text{tanh}(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

$$y_t = W_{hy} \cdot h_t$$

其中：

- $h_t$：时刻t的隐藏状态
- $x_t$：时刻t的输入
- $y_t$：时刻t的输出
- $W_{xh}$：输入到隐藏的权重
- $W_{hh}$：隐藏到隐藏的权重（记忆）

### 6.2.3 PyTorch实现

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True  # 输入形状: [batch, seq_len, features]
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # output: [batch, seq_len, hidden_dim]
        # hidden: [num_layers, batch, hidden_dim]
        output, hidden = self.rnn(embedded)

        # 取最后一个时间步的输出
        last_output = output[:, -1, :]  # [batch, hidden_dim]
        return self.fc(last_output)

# 使用
model = SimpleRNN(vocab_size=10000, embed_dim=300,
                  hidden_dim=128, num_classes=2)
x = torch.randint(0, 10000, (2, 10))  # [batch=2, seq_len=10]
output = model(x)
print(output.shape)  # [2, 2]
```


---

## 6.3 RNN的问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **梯度消失** | 长序列信息丢失 | 无法学习长期依赖 |
| **梯度爆炸** | 梯度指数增长 | 训练不稳定，NaN |
| **并行化差** | 必须按顺序计算 | 训练慢 |

**为什么梯度消失？**

$$\frac{\partial h_T}{\partial h_t} = \prod_{i=t}^{T-1} \frac{\partial h_{i+1}}{\partial h_i}$$

多个小于1的梯度相乘，结果接近0。

---

## 6.4 LSTM

### 6.4.1 LSTM核心思想

**LSTM（Long Short-Term Memory）**：用门控机制控制信息流

```

RNN：  只有一种记忆
LSTM： 长期记忆（细胞状态）+ 短期记忆（隐藏状态）


三个门：
- 遗忘门：决定丢弃什么信息
- 输入门：决定记住什么信息
- 输出门：决定输出什么信息
```


### 6.4.2 LSTM结构

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$  (遗忘门）

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$  (输入门)

$$\tilde{C}_t = \text{tanh}(W_C \cdot [h_{t-1}, x_t] + b_C)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$  (细胞状态)

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$  (输出门)

$$h_t = o_t \odot \text{tanh}(C_t)$$  (隐藏状态)

### 6.4.3 LSTM解决梯度消失

**细胞状态 $C_t$**：梯度可以几乎无损地传递

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

如果遗忘门 $f_t \approx 1$，梯度可以长期保持。

### 6.4.4 PyTorch LSTM

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,      # 多层LSTM
            batch_first=True,
            dropout=0.2        # 防止过拟合
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)

        # output: [batch, seq_len, hidden_dim]
        # hidden: [num_layers, batch, hidden_dim] (h, c)
        output, (hidden, cell) = self.lstm(embedded)

        # 取最后一层、最后一个时间步的隐藏状态
        last_hidden = hidden[-1]  # [batch, hidden_dim]
        return self.fc(last_hidden)

# 使用
model = LSTMModel(vocab_size=10000, embed_dim=300,
                hidden_dim=128, num_classes=2)
```


---

## 6.5 GRU

### 6.5.1 GRU vs LSTM

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 细胞状态 | 有 | 无（只有隐藏状态）|
| 参数量 | 多 | 少 |
| 性能 | 略好 | 相近 |
| 训练速度 | 慢 | 快 |

### 6.5.2 GRU公式

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$  (更新门)

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$  (重置门)

$$\tilde{h}_t = \text{tanh}(W \cdot [r_t \odot h_{t-1}, x_t])$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 6.5.3 PyTorch GRU

```python
self.gru = nn.GRU(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

# 使用方式与LSTM相同
output, hidden = self.gru(embedded)
```


---

## 6.6 双向RNN

**问题**：标准RNN只能看到过去的上下文

**解决**：双向RNN同时处理过去和未来

```python
self.rnn = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    batch_first=True,
    bidirectional=True  # 双向LSTM
)

# 隐藏状态维度会翻倍
# hidden: [2 * num_layers, batch, hidden_dim]
```


**适用场景**：

- 机器翻译（需要完整句子）
- 文本分类
- 问答系统

---

## 6.7 RNN训练

### 6.7.1 训练循环

```python
import torch.optim as optim

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(vocab_size=10000, embed_dim=300,
                hidden_dim=128, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 训练
for epoch in range(20):
    model.train()
    train_loss = 0
    correct = 0

    for batch in train_loader:
        x, labels = batch
        x, labels = x.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%')

    scheduler.step()
```


### 6.7.2 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```


---

## 6.8 实战：文本分类

**任务**：IMDB电影评论情感分类（正面/负面）

```python
from torchtext import datasets, data

# 字段定义
TEXT = data.Field(lower=True, batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

# 加载数据
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 构建词表
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# 数据加载器
BATCH_SIZE = 64
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

# 模型
model = LSTMModel(
    vocab_size=len(TEXT.vocab),
    embed_dim=100,
    hidden_dim=256,
    num_classes=2
).to(device)

# 训练（目标准确率：>88%）
# 完整代码参考第4章MNIST示例
```


---

## 6.9 总结：RNN变体对比

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| RNN | 简单，但梯度消失 | 短序列 |
| LSTM | 门控，长期记忆 | 长序列，通用 |
| GRU | 参数少，训练快 | 长序列，需要速度 |
| Bi-RNN | 双向上下文 | 文本分类、翻译 |

---

## 本章小结

**核心概念**：

1. ✅ RNN通过隐藏状态传递记忆
2. ✅ 梯度消失导致无法处理长序列
3. ✅ LSTM用门控机制解决梯度消失
4. ✅ GRU是LSTM的简化版本
5. ✅ 双向RNN同时看到过去和未来

**关键公式**：

- RNN：$h_t = \text{tanh}(W_{xh}x_t + W_{hh}h_{t-1} + b)$
- LSTM细胞状态：$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

---

## 思考题

1. 为什么标准RNN无法处理长序列？
2. LSTM的细胞状态和隐藏状态有什么区别？
3. GRU相比LSTM有什么优势？
4. 什么情况下需要使用双向RNN？

---

## 下一步

下一章我们将学习**注意力机制**，这是现代NLP的基础：

- 为什么RNN仍然有局限？
- 注意力机制的核心思想
- Self-Attention如何工作
- 多头注意力的威力

准备好进入Transformer的世界了吗？
