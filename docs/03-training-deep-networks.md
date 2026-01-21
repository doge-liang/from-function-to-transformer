# 第三章：训练深度神经网络

> 损失函数、优化算法、参数初始化与正则化

---

## 3.1 损失函数

### 3.1.1 什么是损失函数？

**损失函数**（Loss Function）：衡量预测值与真实值之间差距的函数。

$$\text{Loss} = f(\hat{y}, y)$$

**目标**：找到使损失函数最小化的参数 $W$ 和 $b$。

$$\min_{W, b} \text{Loss}(W, b)$$

### 3.1.2 回归损失

**均方误差（MSE）**：
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**特点**：

- 惩罚大误差（平方放大）
- 可导，便于优化
- 对异常值敏感

**均绝对误差（MAE）**：
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**特点**：

- 对异常值鲁棒
- 在零点不可导（需用次梯度）

```python
import torch.nn as nn

# MSE
mse_loss = nn.MSELoss()

# MAE
mae_loss = nn.L1Loss()

# Huber Loss (结合MSE和MAE的优点)
huber_loss = nn.SmoothL1Loss()
```

### 3.1.3 分类损失

**交叉熵损失（Cross-Entropy）**：

二分类：
$$L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

多分类：
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

其中 $C$ 是类别数。

**PyTorch 实现**：

```python
import torch.nn as nn

# 二分类
criterion = nn.BCELoss()  # 需要手动sigmoid
# 或
criterion = nn.BCEWithLogitsLoss()  # 内部包含sigmoid

# 多分类
criterion = nn.CrossEntropyLoss()  # 内部包含softmax
```

### 3.1.4 损失函数选择指南

| 任务类型 | 推荐损失 | PyTorch |
|----------|----------|---------|
| 回归 | MSE | `nn.MSELoss()` |
| 回归（鲁棒） | MAE | `nn.L1Loss()` |
| 二分类 | Binary Cross-Entropy | `nn.BCEWithLogitsLoss()` |
| 多分类 | Cross-Entropy | `nn.CrossEntropyLoss()` |

---

## 3.2 梯度下降与优化算法

### 3.2.1 基础梯度下降

$$W_t = W_{t-1} - \eta \cdot \nabla L(W_{t-1})$$

其中 $\eta$ 是学习率（Learning Rate）。

### 3.2.2 Mini-Batch SGD

**三种梯度下降方式**：

| 类型 | 每批样本数 | 更新次数/epoch | 特点 |
|------|------------|----------------|------|
| Batch GD | 全部数据 | 1 | 梯度准确，计算慢 |
| SGD | 1个 | N | 噪声大，震荡 |
| **Mini-Batch** | **32/64/128** | **N/batch** | **实际常用** |

**代码实现**：

```python
import torch
import torch.optim as optim

# 模型定义
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        # 前向传播
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()
```

### 3.2.3 Momentum

**问题**：SGD在平坦区域收敛慢，容易陷入局部最优。

**解决方案**：Momentum（动量）

$$v_t = \beta v_{t-1} + (1-\beta)\nabla L(W_{t-1})$$

$$W_t = W_{t-1} - \eta v_t$$

**效果**：

- 加速收敛
- 跳出局部最优

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 3.2.4 Nesterov Accelerated Gradient (NAG)

**论文**：《Nesterov: A method of solving a convex programming problem with convergence rate O(1/k²)》

**思想**：先看"如果动量更新后的位置"，再计算梯度。

$$v_t = \beta v_{t-1} + \nabla L(W_{t-1} - \eta \beta v_{t-1})$$

$$W_t = W_{t-1} - \eta v_t$$

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

### 3.2.5 AdaGrad

**论文**：《Duchi et al. (2011): Adaptive Subgradient Methods for Online Learning and Stochastic Optimization》

**核心**：自适应学习率，频繁更新的参数学习率小。

$$G_t = G_{t-1} + (\nabla L)^2$$

$$W_t = W_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} \nabla L$$

**特点**：

- 适合稀疏数据
- 学习率递减可能过早停止

```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

### 3.2.6 RMSprop

**论文**：《Hinton et al. (2012): Neural Networks for Machine Learning》

**改进AdaGrad**：使用指数移动平均，学习率不递减。

$$G_t = \beta G_{t-1} + (1-\beta)(\nabla L)^2$$

$$W_t = W_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} \nabla L$$

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

### 3.2.7 Adam（Adaptive Moment Estimation）

**论文**：《Kingma & Ba (2015): Adam: A Method for Stochastic Optimization》

**核心**：结合Momentum和RMSprop的优势。

**一阶矩估计（动量）**：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L$$

**二阶矩估计（方差）**：
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$$

**偏差修正**：
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**参数更新**：
$$W_t = W_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**默认超参数**：

- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3.2.8 AdamW

**论文**：《Loshchilov & Hutter (2019): Decoupled Weight Decay Regularization》

**改进**：将权重衰减（L2正则）从梯度更新中解耦。

$$W_t = W_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda W_{t-1}$$

**优势**：更适合大规模预训练。

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 3.2.9 优化器对比

| 优化器 | 优点 | 缺点 | 推荐场景 |
|--------|------|------|----------|
| SGD | 简单、泛化好 | 收敛慢、震荡 | 学术研究、最终调优 |
| SGD+Momentum | 收敛快 | 需调参 | CNN训练 |
| Adam | 自适应、快 | 可能泛化差 | NLP、Transformer |
| AdamW | 解耦权重衰减 | 需调参 | 大模型预训练 |

---

## 3.3 参数初始化

### 3.3.1 为什么初始化很重要？

**问题**：

- **梯度消失**：初始化值太小，反向传播时梯度指数级衰减
- **梯度爆炸**：初始化值太大，反向传播时梯度指数级增长
- **对称性问题**：所有神经元初始相同，学习相同特征

### 3.3.2 全零初始化的问题

如果所有权重初始化为0：

- 所有神经元输出相同
- 无法学习不同特征
- 对称性无法打破

**偏置可以初始化为0**（不会导致对称问题）

```python
# 错误：权重全零初始化
W = torch.zeros(10, 10)

# 正确：偏置全零初始化
bias = torch.zeros(10)
```

### 3.3.3 Xavier初始化

**论文**：《Glorot & Bengio (2010): Understanding the difficulty of training deep feedforward neural networks》

**核心**：保持各层激活值和梯度的方差一致。

**Xavier Uniform**：
$$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$

**Xavier Normal**：
$$W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

**适用**：Sigmoid、Tanh激活函数

```python
# PyTorch
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

### 3.3.4 He初始化

**论文**：《He et al. (2015): Delving Deep into Rectifiers》

**核心**：针对ReLU激活函数优化。

**He Uniform**：
$$W \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$$

**He Normal**：
$$W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**适用**：ReLU、Leaky ReLU激活函数

```python
# PyTorch
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### 3.3.5 LeCun初始化

**适用**：SELU激活函数

$$W \sim N\left(0, \frac{1}{n_{in}}\right)$$

```python
nn.init.normal_(layer.weight, mean=0, std=np.sqrt(1/n_in))
```

### 3.3.6 PyTorch默认初始化

| 层类型 | 默认初始化 |
|--------|------------|
| nn.Linear | Xavier Uniform |
| nn.Conv2d | Kaiming Uniform |
| nn.RNN | Xavier Uniform |

---

## 3.4 正则化技术

### 3.4.1 什么是正则化？

**正则化（Regularization）**：在损失函数中添加惩罚项，限制模型复杂度。

**目的**：

- 防止过拟合
- 提高泛化性

### 3.4.2 L1正则化（Lasso）

$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n}|w_i|$$

**特点**：

- 产生稀疏解（很多权重为0）
- 可用于特征选择
- 适合高维稀疏数据

```python
# L1正则化（手动实现）
lambda_l1 = 0.01
l1_reg = sum(torch.norm(param, 1) for param in model.parameters())
loss = criterion(predictions, y) + lambda_l1 * l1_reg
```

### 3.4.3 L2正则化（Ridge）

$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n}w_i^2$$

**特点**：

- 权重趋近于0但不为0
- 稳定训练
- **最常用**

```python
# L2正则化（使用weight_decay）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 3.4.4 Dropout

**论文**：《Srivastava et al. (2014): Dropout: A Simple Way to Prevent Neural Networks from Overfitting》

**核心**：训练时随机丢弃一部分神经元，测试时使用全部神经元。

**训练时**：
$$h = \text{dropout}(a) = a \cdot \text{mask}$$

其中 mask $\in \{0, 1\}$，$P(mask=1) = p$（保留概率）

**测试时**：
$$h = a \cdot p$$

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 训练时随机丢弃
        x = self.fc2(x)
        return x
```

### 3.4.5 Batch Normalization

**论文**：《Ioffe & Szegedy (2015): Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》

**核心**：对每层的激活值进行标准化。

**训练时**：
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i$$

$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

**优点**：

- 加速收敛
- 允许更大学习率
- 减少对初始化的敏感性

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # BatchNorm
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

### 3.4.6 Early Stopping

**核心**：监控验证集损失，当不再下降时停止训练。

```python
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch()

    val_loss = validate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
```

---

## 3.5 学习率调度

### 3.5.1 学习率调度策略

**Step LR**：固定间隔衰减

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**Exponential LR**：指数衰减

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

**Cosine Annealing**：余弦退火

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

**ReduceLROnPlateau**：监控指标，不改善时降低学习率

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)
```

### 3.5.2 学习率预热（Warmup）

**场景**：Transformer训练

**策略**：前N步线性增加学习率，之后使用调度器

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, d_model):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.d_model ** -0.5 * min(
            self.current_step ** -0.5,
            self.current_step * self.warmup_steps ** -1.5
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

---

## 3.6 完整训练代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 模型定义
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. 初始化
model = MLP()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 4. 训练循环
def train_one_epoch(epoch):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')

    return avg_loss

def validate():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Val: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')

    return avg_loss

# 5. 训练与验证
best_val_loss = float('inf')
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(epoch)
    val_loss = validate()
    scheduler.step()

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved!')
```

---

## 本章小结

**核心概念**：

1. ✅ 损失函数：MSE、Cross-Entropy等
2. ✅ 优化算法：SGD、Momentum、Adam、AdamW
3. ✅ 参数初始化：Xavier、He初始化
4. ✅ 正则化：L1/L2、Dropout、BatchNorm
5. ✅ 学习率调度：Step、Cosine、Warmup

**关键公式**：

- MSE：$\frac{1}{n}\sum(y_i - \hat{y}_i)^2$
- Adam：$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
- He初始化：$W \sim N(0, \sqrt{2/n_{in}})$

---

## 思考题

1. 为什么Adam有时不如SGD泛化好？
2. Dropout如何防止过拟合？为什么测试时不需要Dropout？
3. BatchNorm在训练和测试时的区别是什么？
4. 如何选择合适的优化器和学习率？

---

## 练习题

**选择题**：

1. 哪种正则化方法会产生稀疏权重？
   A. L2正则  B. L1正则  C. Dropout  D. BatchNorm

2. AdamW相比Adam的改进是什么？
   A. 更快收敛  B. 解耦权重衰减  C. 更省内存  D. 更稳定的梯度

**答案**：

1. B. L1正则
2. B. 解耦权重衰减

---

## 下一步

下一章我们将通过**MNIST实战项目**构建第一个完整的神经网络，包括：

- 数据加载与预处理
- 模型构建与训练
- 模型评估与优化
- 常见错误排查

准备好开始第一个深度学习项目了吗？
