# 第四章：实践 - MNIST手写数字识别

> 构建你的第一个深度学习项目

---

## 4.1 项目概述

### 4.1.1 MNIST数据集

**简介**：MNIST是深度学习领域的"Hello World"数据集。

**内容**：

- 60,000张训练图像
- 10,000张测试图像
- 10类手写数字（0-9）
- 每张图像：28×28像素，灰度图

**下载**：会自动从互联网下载

```python
from torchvision import datasets, transforms

# 自动下载MNIST数据集
train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
```

### 4.1.2 项目目标

| 指标 | 目标值 |
|------|--------|
| 测试集准确率 | >98% |
| 训练时间 | <5分钟（GPU）或 <15分钟（CPU） |
| 模型大小 | <1MB |

---

## 4.2 环境准备

### 4.2.1 安装依赖

```bash
# 安装PyTorch（选择适合你系统的版本）
pip install torch torchvision

# 可选：Jupyter Notebook用于交互式编程
pip install jupyter matplotlib
```

### 4.2.2 验证安装

```python
import torch
import torchvision

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 加载MNIST测试
mnist = datasets.MNIST('./data', download=True)
print(f"MNIST下载成功！")
```

---

## 4.3 数据准备

### 4.3.1 数据预处理

**为什么需要预处理？**

| 原始数据 | 问题 | 解决方案 |
|----------|------|----------|
| 0-255整数 | 数值大，梯度不稳定 | 归一化到[0, 1] |
| 灰度值 | 单通道 | 转换为Tensor |
| 28×28矩阵 | 需要展平或CNN | 保持形状或展平 |

```python
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为[0, 1]的Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（使用MNIST的均值和标准差）
])

# 加载数据集
train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    './data', train=False, download=True, transform=transform
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
```

### 4.3.2 数据加载器

```python
from torch.utils.data import DataLoader

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64,      # 每批64个样本
    shuffle=True,       # 训练时打乱数据
    num_workers=2       # 多进程加载数据
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,    # 测试时可以使用更大的batch
    shuffle=False       # 测试时不需要打乱
)

# 查看一个batch
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"图像形状: {images.shape}")  # [64, 1, 28, 28]
print(f"标签形状: {labels.shape}")  # [64]
print(f"标签内容: {labels[:10]}")
```

### 4.3.3 数据可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels, n_rows=4, n_cols=8):
    """显示图像和标签"""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 显示前32张训练图像
show_images(images, labels)
```

---

## 4.4 模型构建

### 4.4.1 简单多层感知机（MLP）

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 展平图像: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(x.size(0), -1)

        # 第一层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)
        return x

# 创建模型
model = MLP()
print(model)

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数量: {total_params:,}")
```

### 4.4.2 更深的网络（可选）

```python
class DeepMLP(nn.Module):
    """更深的网络"""
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = DeepMLP()
```

---

## 4.5 训练配置

### 4.5.1 设备选择

```python
import torch

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将模型移动到设备
model = model.to(device)
```

### 4.5.2 损失函数和优化器

```python
import torch.optim as optim

# 损失函数：交叉熵（自动包含Softmax）
criterion = nn.CrossEntropyLoss()

# 优化器：Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器（可选）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

---

## 4.6 训练循环

### 4.6.1 训练函数

```python
def train(model, device, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 移动数据到设备
        data, target = data.to(device), target.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每100个batch打印一次
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

    return avg_loss, accuracy
```

### 4.6.2 测试函数

```python
def test(model, device, test_loader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return avg_loss, accuracy
```

### 4.6.3 完整训练循环

```python
def train_model(model, device, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20):
    """完整训练流程"""
    best_accuracy = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # 训练
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)

        # 测试
        test_loss, test_acc = test(model, device, test_loader, criterion)

        # 学习率调度
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print(f"✓ 最佳模型已保存，准确率: {test_acc:.2f}%")

    return history

# 训练模型
history = train_model(model, device, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20)
```

---

## 4.7 结果分析

### 4.7.1 训练曲线可视化

```python
def plot_training_history(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['test_acc'], label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(history)
```

### 4.7.2 预测结果可视化

```python
def visualize_predictions(model, device, test_loader, num_samples=16):
    """可视化预测结果"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        pred_label = predicted[i].item()
        true_label = labels[i].item()
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'Pred: {pred_label}, True: {true_label}', color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(model, device, test_loader)
```

---

## 4.8 模型评估

### 4.8.1 混淆矩阵

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, device, test_loader):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(model, device, test_loader)
```

### 4.8.2 各类别准确率

```python
def calculate_per_class_accuracy(model, device, test_loader):
    """计算每个类别的准确率"""
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("各类别准确率:")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"数字 {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")

calculate_per_class_accuracy(model, device, test_loader)
```

---

## 4.9 常见问题与调试

### 4.9.1 模型不收敛

**症状**：训练损失不下降或震荡

**可能原因**：

1. 学习率太大
2. 模型太复杂，过拟合
3. 数据预处理错误
4. 梯度爆炸

**解决方案**：

```python
# 降低学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4.9.2 损失为NaN

**症状**：训练过程中损失变为NaN

**可能原因**：

1. 学习率太大
2. 初始化不当
3. 数值不稳定

**解决方案**：

```python
# 降低学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 使用更稳定的初始化
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### 4.9.3 训练慢

**症状**：训练时间过长

**优化方法**：

```python
# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 增加batch size（如果显存足够）
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 使用更快的优化器
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 减少数据加载进程
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
```

### 4.9.4 准确率不提升

**症状**：训练集准确率高，测试集准确率低

**诊断**：过拟合

**解决方案**：

```python
# 增加Dropout
self.dropout = nn.Dropout(0.5)  # 从0.2增加到0.5

# 添加L2正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# 使用Batch Normalization
self.bn = nn.BatchNorm1d(hidden_size)

# 数据增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),  # 随机旋转
    transforms.Normalize((0.1307,), (0.3081,))
])
```

---

## 4.10 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 模型定义
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 3. 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 4. 训练
def train_one_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0

    for data, target in train_loader:
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
    return avg_loss, accuracy

def test(model, device, test_loader, criterion):
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
    return avg_loss, accuracy

# 训练循环
best_acc = 0
for epoch in range(1, 21):
    train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    scheduler.step()

    print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
          f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%')

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_mnist_model.pth')

print(f'\n最佳准确率: {best_acc:.2f}%')
```

---

## 本章小结

**项目完成目标**：

1. ✅ 加载和预处理MNIST数据集
2. ✅ 构建多层感知机模型
3. ✅ 训练模型达到98%+准确率
4. ✅ 可视化训练过程和预测结果
5. ✅ 诊断和解决常见问题

**关键技能**：

- 数据加载与预处理
- 模型设计与实现
- 训练循环编写
- 结果分析与可视化
- 调试与优化技巧

---

## 练习与挑战

**基础练习**：

1. 尝试不同的网络架构（层数、神经元数）
2. 调整超参数（学习率、batch size、dropout比例）
3. 使用不同的优化器（SGD、Adam、AdamW）

**进阶挑战**：

1. 实现早停（Early Stopping）
2. 添加学习率预热（Warmup）
3. 实现数据增强（随机旋转、平移）
4. 使用卷积神经网络（CNN）替代MLP

**终极挑战**：

- 尝试达到99%+的准确率
- 对比不同模型的性能
- 分析模型对哪些数字最容易混淆

---

## 下一步

恭喜你完成了第一个深度学习项目！

下一章我们将学习**卷积神经网络（CNN）**，包括：

- 卷积操作与卷积核
- 池化层的作用
- 经典CNN架构（LeNet、AlexNet、VGG、ResNet）
- CIFAR-10图像分类实战

准备好探索图像识别的更强大模型了吗？
