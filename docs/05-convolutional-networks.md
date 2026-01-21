# 第五章：卷积神经网络（CNN）

> 用卷积操作提取图像局部特征

---

## 5.1 为什么需要CNN？

### 5.1.1 全连接层的问题

处理224×224×3的RGB图像，如果第一个隐藏层有1000个神经元：

```

参数数量 = 224×224×3 × 1000 ≈ 1.5亿

```

**问题**：

- 参数爆炸
- 过拟合风险
- 丢失空间信息
- 计算量大

### 5.1.2 图像的两个特性

**1. 局部性**：检测特征只需看局部像素
**2. 可重复性**：学到的特征应该在所有位置有效

### 5.1.3 CNN核心思想

| 技术 | 作用 |
|------|------|
| 局部连接 | 每个神经元只连接局部区域 |
| 权重共享 | 同一卷积核在整个图像上复用 |

---

## 5.2 卷积操作

### 5.2.1 卷积核

**卷积核（Kernel）**：一个小窗口，在图像上滑动提取特征。

```

比喻：用放大镜（卷积核）在图片上滑动

- 放大镜看到局部区域
- 不同的放大镜关注不同特征
- 移动放大镜覆盖整个图片

```

### 5.2.2 卷积计算

**核心操作**：对应位置相乘，然后求和

**一维示例**：

```

输入：    [1, 2, 3, 4, 5]
卷积核：  [0.5, 1, 0.5]


输出：
位置0：1×0.5 + 2×1 + 3×0.5 = 4.0
位置1：2×0.5 + 3×1 + 4×0.5 = 6.0
位置2：3×0.5 + 4×1 + 5×0.5 = 8.0

输出：[4.0, 6.0, 8.0]
```

**二维示例**：

```

输入区域：        卷积核：
┌ 1  0  1 ┐      ┌ 1  0 -1 ┐
│ 2  1  0 │      │ 0  1  0 │
└ 1  0  1 ┘      └ -1 0  1 ┘


计算：
(1×1 + 0×0 + 1×-1) +
(2×0 + 1×1 + 0×0) +
(1×-1 + 0×0 + 1×1) = 1
```

### 5.2.3 卷积参数

```python
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size)
conv = nn.Conv2d(
    in_channels=3,      # 输入通道数（RGB=3）
    out_channels=64,    # 输出通道数（卷积核数量）
    kernel_size=3,      # 卷积核大小
    stride=1,          # 步长
    padding=1           # 填充
)

# 输入：[batch, 3, 28, 28]
# 输出：[batch, 64, 28, 28]
```

| 参数 | 含义 | 常用值 |
|------|------|--------|
| in_channels | 输入通道数 | 3(RGB)或1(灰度）|
| out_channels | 输出通道数 | 32, 64, 128... |
| kernel_size | 卷积核大小 | 3×3, 5×5 |
| stride | 步长 | 1, 2 |
| padding | 填充 | 0, 1, 2 |

### 5.2.4 不同卷积核的效果

| 卷积核 | 检测效果 |
|--------|----------|
| `[[-1,0,1], [-2,0,2], [-1,0,1]]` | 垂直边缘 |
| `[[1,2,1], [0,0,0], [-1,-2,-1]]` | 水平边缘 |
| `[[1/9,...],...,]` | 模糊 |
| `[[0,-1,0], [-1,5,-1], [0,-1,0]]` | 锐化 |

**重要**：CNN的卷积核值是**从数据中学习**的，不是手工设计。

---

## 5.3 池化层

### 5.3.1 什么是池化？

**池化（Pooling）**：对局部区域进行下采样，减小特征图尺寸。

### 5.3.2 Max Pooling vs Average Pooling

```

输入2×2区域：
┌ 1  3 ┐
│ 2  4 │
└──────┘


Max Pooling：   max(1,3,2,4) = 4
Avg Pooling：    avg(1,3,2,4) = 2.5
```

| 类型 | 计算 | 特点 |
|------|------|------|
| Max Pooling | 取最大值 | 保留最显著特征 |
| Average Pooling | 取平均值 | 保留整体信息 |

### 5.3.3 池化的作用

1. **减少计算量**：特征图变小
2. **防止过拟合**：减少参数
3. **平移不变性**：物体移动，池化结果不变

```python
# Max Pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average Pooling
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
```

---

## 5.4 CNN架构

### 5.4.1 典型CNN结构

```

输入图像
    │
    ▼
[卷积 + 激活 + 池化] × N
    │
    ▼
[全连接层]
    │
    ▼
输出

```

### 5.4.2 PyTorch实现

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 28×28 → 14×14
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14×14 → 7×7
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# 使用
model = CNN(num_classes=10)
x = torch.randn(1, 3, 28, 28)  # [batch, channels, height, width]
output = model(x)
print(output.shape)  # [1, 10]
```

---

## 5.5 经典CNN架构

### 5.5.1 LeNet-5（1998）

**结构**：`输入 → 卷积 → 池化 → 卷积 → 池化 → 全连接`

**用途**：手写数字识别（MNIST）

### 5.5.2 AlexNet（2012）

**突破**：首次使用深度CNN在ImageNet取得重大突破

**特点**：

- 使用ReLU激活
- Dropout正则化
- 数据增强

### 5.5.3 VGG（2014）

**特点**：

- 全部使用3×3卷积核
- 非常深的网络（16/19层）

### 5.5.4 ResNet（2015）

**创新**：残差连接（Residual Connection）

```python
# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return torch.relu(out)
```

**优势**：

- 解决梯度消失
- 可训练超深网络（100+层）

---

## 5.6 CNN训练

### 5.6.1 训练流程

CNN的训练与普通神经网络相同：

```python
import torch.optim as optim

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 训练循环
for epoch in range(20):
    model.train()
    train_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%')
```

### 5.6.2 数据增强

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ColorJitter(0.2, 0.2, 0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## 5.7 实战：CIFAR-10分类

**数据集**：10类32×32彩色图像

```python
from torchvision import datasets, transforms

# 数据准备
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 训练（使用上面的模型）
# 目标准确率：>85%
```

---

## 本章小结

**核心概念**：

1. ✅ CNN通过局部连接和权重共享减少参数
2. ✅ 卷积操作：滑动窗口，对应位置相乘求和
3. ✅ 池化层：下采样，减少计算量，增加不变性
4. ✅ 经典架构：LeNet、AlexNet、VGG、ResNet
5. ✅ CNN训练流程与MLP相同

**关键公式**：

- 卷积：$(f * g)(i,j) = \sum_m \sum_n f(i+m,j+n) \cdot g(m,n)$
- 输出尺寸：$\lfloor \frac{W-K+2P}{S} \rfloor + 1$

---

## 思考题

1. 为什么卷积核的值是学习的而不是固定的？
2. Max Pooling和Average Pooling分别适合什么场景？
3. ResNet的残差连接解决了什么问题？
4. 为什么CNN相比全连接网络更适合图像任务？

---

## 下一步

下一章我们将学习**循环神经网络（RNN）**，用于处理序列数据：

- 为什么CNN不适合处理序列？
- RNN的基本结构
- LSTM如何解决梯度消失
- 序列建模实战

准备好探索时间序列和自然语言处理的模型了吗？
