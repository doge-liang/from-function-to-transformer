# 第二章：神经网络结构

> 理解前向传播和网络层次

---

## 2.1 网络层次结构

神经网络由三种层组成：**输入层**、**隐藏层**和**输出层**。

```mermaid
graph TD
    subgraph 输入层
    I1[x1]
    I2[x2]
    end

    subgraph 隐藏层1
    H1
    H2
    H3
    end

    subgraph 隐藏层2
    H4
    H5
    end

    subgraph 输出层
    O1[y_hat]
    end

    I1 --> H1
    I1 --> H2
    I1 --> H3
    I2 --> H1
    I2 --> H2
    I2 --> H3
    H1 --> H4
    H1 --> H5
    H2 --> H4
    H2 --> H5
    H3 --> H4
    H3 --> H5
    H4 --> O1
    H5 --> O1

    style 输入层 fill:#e1f5fe
    style 输出层 fill:#fff3e0
    style 隐藏层1 fill:#f3e5f5
    style 隐藏层2 fill:#f3e5f5
```

### 术语解释

| 层次 | 英文 | 说明 |
|------|------|------|
| 输入层 | Input Layer | 接收原始数据 |
| 隐藏层 | Hidden Layer | 中间的计算层，不直接与外界交互 |
| 输出层 | Output Layer | 产生最终预测结果 |
| 神经元 | Neuron | 每个圆圈代表一个神经元 |

### 重要概念

- **层数**：计算隐藏层的数量（不包含输入层和输出层）
- **神经元数量**：每层中神经元的个数
- **全连接**：每个神经元与前一层的所有神经元相连

---

## 2.2 前向传播

**前向传播**（Forward Propagation）：数据从输入层到输出层的计算过程。

### 2.2.1 数学表示

对于第 $l$ 层：

$$z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}$$

$$a^{(l)} = \text{激活函数}(z^{(l)})$$

其中：
- $W^{(l)}$：第 $l$ 层的权重矩阵
- $b^{(l)}$：第 $l$ 层的偏置向量
- $a^{(l-1)}$：第 $l-1$ 层的输出

### 2.2.2 前向传播流程

```mermaid
flowchart LR
    X[输入 x] -->|"W1"| Z1[z1 = W1*x + b1]
    Z1 --> A1[a1 = sigma_z1]
    A1 -->|"W2"| Z2[z2 = W2*a1 + b2]
    Z2 --> A2[a2 = sigma_z2]
    A2 --> Y[输出 y_hat]

    style X fill:#e1f5fe
    style Y fill:#fff3e0
```

### 2.2.3 单个神经元的计算

一个神经元接收多个输入 $x_1, x_2, ..., x_n$，计算：

$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

$$a = \sigma(z)$$

```mermaid
graph LR
    x1((x1)) -->|"w1"| sum1((+))
    x2((x2)) -->|"w2"| sum1
    x3((x3)) -->|"w3"| sum1
    b((b)) --> sum1
    sum1 --> sigma((sigma))
    sigma --> a((a))

    style x1 fill:#e1f5fe
    style x2 fill:#e1f5fe
    style x3 fill:#e1f5fe
    style a fill:#fff3e0
```

---

## 2.3 完整的前向传播示例

假设有一个简单的两层神经网络：
- 输入层：2 个神经元
- 隐藏层：3 个神经元（ReLU 激活）
- 输出层：1 个神经元（Sigmoid 激活）

```mermaid
flowchart LR
    subgraph 输入["输入层 (2)"]
    x1[x1]
    x2[x2]
    end

    subgraph 隐藏["隐藏层 (3)"]
    h1[h1]
    h2[h2]
    h3[h3]
    end

    subgraph 输出["输出层 (1)"]
    o[y_hat]
    end

    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> o
    h2 --> o
    h3 --> o

    style 输入 fill:#e1f5fe
    style 隐藏 fill:#f3e5f5
    style 输出 fill:#fff3e0
```

---

## 代码实现

```python
import numpy as np

def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

# 初始化网络参数
np.random.seed(42)

# 输入层 -> 隐藏层 (2 -> 3)
W1 = np.random.randn(2, 3)   # 权重矩阵
b1 = np.random.randn(3)      # 偏置

# 隐藏层 -> 输出层 (3 -> 1)
W2 = np.random.randn(3, 1)
b2 = np.random.randn(1)

def forward(X):
    """前向传播"""
    # 输入 -> 隐藏层
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    # 隐藏层 -> 输出层
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    return a2

# 测试
X_test = np.array([[0.5, 1.0]])
output = forward(X_test)
print(f"输入: {X_test}")
print(f"输出: {output[0][0]:.4f}")
```

---

## 思考题

1. 如果增加隐藏层的层数，会发生什么？
2. 为什么需要非线性激活函数？如果全部用线性函数会怎样？
3. 神经网络的"深度"指的是什么？

---

## 下一步

下一章我们将讨论如何训练神经网络：损失函数、梯度下降和反向传播。
