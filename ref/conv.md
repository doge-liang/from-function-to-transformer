# 卷积的数学原理

> 深入理解卷积操作的数学本质

---

## 1. 卷积的数学定义

### 1.1 一维卷积

**连续形式**：

$$(f * g)(t) = \int_{-\infty}^{+\infty} f(\tau) \cdot g(t - \tau) \, d\tau$$

**离散形式**（本文档主要讨论）：

$$(f * g)[n] = \sum_{m=-\infty}^{+\infty} f[m] \cdot g[n - m]$$

其中：
- $f$ 是输入信号
- $g$ 是卷积核（kernel）或滤波器（filter）
- $n$ 是输出位置索引
- $m$ 是求和变量

### 1.2 二维卷积（图像）

对于图像处理，通常使用**二维离散卷积**：

$$(I * K)[i, j] = \sum_{m=-\infty}^{+\infty} \sum_{n=-\infty}^{+\infty} I[m, n] \cdot K[i - m, j - n]$$

在实际应用中，卷积核大小是有限的（通常为 3×3 或 5×5），因此简化为：

$$(I * K)[i, j] = \sum_{m=0}^{H_k-1} \sum_{n=0}^{W_k-1} I[i + m, j + n] \cdot K[m, n]$$

其中：
- $I$ 是输入图像，尺寸为 $H_I \times W_I$
- $K$ 是卷积核，尺寸为 $H_k \times W_k$
- $[i, j]$ 是输出位置

**注意**：在深度学习中，我们通常使用"互相关"（cross-correlation）而非真正的卷积，区别在于是否翻转卷积核。但为简化，本文使用深度学习领域的习惯用法。

---

## 2. 卷积的通俗理解

### 2.1 核心操作："滑动窗口 + 元素级乘积求和"

```
卷积 = 滑动窗口 × 对应位置相乘，然后求和
```

### 2.2 一维示例

假设输入信号和卷积核如下：

```
输入信号 f = [1, 2, 3, 4, 5]
卷积核 g   = [0.5, 1, 0.5]    （核大小 K=3）
```

**计算过程**：

| 输出位置 | 计算式 | 结果 |
|----------|--------|------|
| n=0 | 1×0.5 + 2×1 + 3×0.5 | 4.0 |
| n=1 | 2×0.5 + 3×1 + 4×0.5 | 6.0 |
| n=2 | 3×0.5 + 4×1 + 5×0.5 | 8.0 |

**Python 实现**：

```python
import numpy as np

f = np.array([1, 2, 3, 4, 5])
g = np.array([0.5, 1, 0.5])

# 手动实现一维卷积
def conv1d(input, kernel):
    k_len = len(kernel)
    out_len = len(input) - k_len + 1
    output = np.zeros(out_len)

    for i in range(out_len):
        output[i] = np.sum(input[i:i+k_len] * kernel)

    return output

result = conv1d(f, g)
# result = [4.0, 6.0, 8.0]
```

### 2.3 二维示例（图像卷积）

对于 5×5 图像和 3×3 卷积核：

```
输入图像 I (5×5)：          卷积核 K (3×3)：
┌ 1  0  1  2  0 ┐          ┌ 1  0 -1 ┐
│ 2  1  0  1  1 │          │ 0  1  0 │
│ 1  0  1  2  0 │    *     │ -1 0  1 │
│ 0  1  0  1  1 │          └         ┘
└ 1  0  1  0  0 ┘

计算输出 O[0,0]（左上角第一个值）：
O[0,0] = Σ(I[m,n] × K[m,n]) for m,n in 0..2

     I[0,0]×K[0,0] + I[0,1]×K[0,1] + I[0,2]×K[0,2]
   + I[1,0]×K[1,0] + I[1,1]×K[1,1] + I[1,2]×K[1,2]
   + I[2,0]×K[2,0] + I[2,1]×K[2,1] + I[2,2]×K[2,2]

   = 1×1 + 0×0 + 1×(-1)
   + 2×0 + 1×1 + 0×0
   + 1×(-1) + 0×0 + 1×1

   = 1 - 1 + 1 - 1 + 1
   = 1
```

**Python 实现**：

```python
import numpy as np

# 输入图像 (5×5)
I = np.array([
    [1, 0, 1, 2, 0],
    [2, 1, 0, 1, 1],
    [1, 0, 1, 2, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0]
])

# 卷积核 (3×3)
K = np.array([
    [1, 0, -1],
    [0, 1, 0],
    [-1, 0, 1]
])

def conv2d(image, kernel):
    """手动实现二维卷积"""
    H, W = image.shape
    kH, kW = kernel.shape
    outH = H - kH + 1
    outW = W - kW + 1

    output = np.zeros((outH, outW))

    for i in range(outH):
        for j in range(outW):
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)

    return output

result = conv2d(I, K)
print("卷积结果 (3×3):")
print(result)
```

---

## 3. 卷积的关键参数

### 3.1 步长（Stride）

**定义**：卷积核在输入上滑动的间隔。

```
Stride = 1:  卷积核每次移动 1 个像素
Stride = 2:  卷积核每次移动 2 个像素
```

**输出尺寸计算**：

$$\text{output\_size} = \left\lfloor \frac{\text{input\_size} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1$$

| 输入尺寸 | 卷积核 | 步长 | 输出尺寸 |
|----------|--------|------|----------|
| 28 | 3 | 1 | 26 |
| 28 | 3 | 2 | 13 |
| 28 | 5 | 1 | 24 |
| 28 | 5 | 2 | 12 |

### 3.2 填充（Padding）

**定义**：在输入边缘添加额外的像素值（通常为 0）。

```
无填充 (padding=0):
输入:  ┌─────────┐ 5×5
输出:  └───┘ 3×3

有填充 (padding=1):
输入:  ┌───────────┐ 7×7 (原5×5 + 1圈0)
输出:  └─────────┘ 5×5
```

**输出尺寸计算**：

$$\text{output\_size} = \left\lfloor \frac{\text{input\_size} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1$$

### 3.3 膨胀（Dilation）

**定义**：卷积核内部像素之间的间隔。

```
Dilation = 1 (标准卷积):    Dilation = 2 (空洞卷积):
┌───┬───┬───┐              ┌─────┬─────┬─────┐
│ 1 │ 2 │ 3 │              │  1  │     │  2  │
├───┼───┼───┤              ├─────┼─────┼─────┤
│ 4 │ 5 │ 6 │       →      │     │  5  │     │
├───┼───┼───┤              ├─────┼─────┼─────┤
│ 7 │ 8 │ 9 │              │  7  │     │  8  │
└───┴───┴───┘              └─────┴─────┴─────┘
有效感受野: 3×3            有效感受野: 5×5
```

**感受野计算**：

$$\text{effective\_kernel\_size} = \text{kernel\_size} + (\text{dilation} - 1) \times (\text{kernel\_size} - 1)$$

### 3.4 参数汇总

| 参数 | 符号 | 作用 |
|------|------|------|
| 步长 | $s$ | 控制输出尺寸，减小分辨率 |
| 填充 | $p$ | 保持空间尺寸，控制信息边界 |
| 膨胀 | $d$ | 扩大感受野，捕获多尺度特征 |

---

## 4. 卷积的数学性质

### 4.1 交换律

$$f * g = g * f$$

### 4.2 结合律

$$(f * g) * h = f * (g * h)$$

### 4.3 分配律

$$f * (g + h) = f * g + f * h$$

### 4.4 与恒等核的卷积

$$f * \delta = f$$

其中 $\delta$ 是单位冲激函数（delta function）：

$$\delta[n] = \begin{cases} 1 & n = 0 \\ 0 & n \neq 0 \end{cases}$$

---

## 5. PyTorch 中的卷积

### 5.1 nn.Conv2d 参数说明

```python
torch.nn.Conv2d(
    in_channels,    # 输入通道数（如 RGB 图像 = 3）
    out_channels,   # 输出通道数（即卷积核数量）
    kernel_size,    # 卷积核大小（如 3 或 (3, 3)）
    stride=1,       # 步长
    padding=0,      # 填充
    dilation=1,     # 膨胀
    bias=True       # 是否使用偏置
)
```

### 5.2 输出尺寸计算公式

$$\text{out\_size} = \left\lfloor \frac{\text{in\_size} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel\_size} - 1) - 1}{\text{stride}} \right\rfloor + 1$$

### 5.3 代码示例

```python
import torch
import torch.nn as nn

# 创建卷积层：3 通道输入，64 通道输出，3×3 卷积核
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1
)

# 输入：batch=1, 通道=3, 高=224, 宽=224
x = torch.randn(1, 3, 224, 224)

# 前向传播
y = conv(x)
print(y.shape)  # torch.Size([1, 64, 224, 224])
```

### 5.4 参数数量计算

对于一个卷积层：

$$\text{参数数量} = (\text{kernel\_size} \times \text{kernel\_size} \times \text{in\_channels} + 1) \times \text{out\_channels}$$

其中 $+1$ 是偏置项（如果启用）。

**示例**：
```
3×3 卷积，3 输入通道，64 输出通道
参数 = (3×3×3 + 1) × 64 = (27 + 1) × 64 = 28 × 64 = 1,792
```

---

## 6. 转置卷积（反卷积）

### 6.1 什么是转置卷积？

转置卷积（Transposed Convolution）是卷积的逆操作，用于**上采样**。

```
卷积:           转置卷积:
5×5 ─────→ 3×3       3×3 ─────→ 5×5
(下采样)              (上采样)
```

### 6.2 PyTorch 实现

```python
import torch
import torch.nn as nn

# 转置卷积层
conv_transpose = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=2,           # 2 倍上采样
    padding=1,
    output_padding=1    # 补充输出尺寸
)

x = torch.randn(1, 64, 32, 32)
y = conv_transpose(x)
print(y.shape)  # torch.Size([1, 3, 64, 64])
```

---

## 7. 常见卷积类型

### 7.1 标准卷积

```
输入:  C_in × H × W
输出:  C_out × H' × W'

参数:  C_in × C_out × K × K
```

### 7.2 深度可分离卷积（Depthwise Separable）

将标准卷积分解为两步：
1. **Depthwise**：每个通道独立卷积
2. **Pointwise**：1×1 卷积融合通道

```python
import torch.nn as nn

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=1,
            groups=in_channels  # 组数 = 输入通道数
        )

        # Pointwise
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

**参数对比**：

| 类型 | 参数数量 |
|------|----------|
| 标准 3×3 卷积 | $C_{in} \times C_{out} \times 9$ |
| 深度可分离卷积 | $C_{in} \times 9 + C_{in} \times C_{out}$ |

**MobileNet 使用深度可分离卷积，将参数减少约 8-9 倍**。

### 7.3 逐通道卷积（Depthwise Convolution）

每个输入通道独立进行卷积：

```python
groups = in_channels
```

### 7.4 逐点卷积（Pointwise Convolution）

1×1 卷积，用于改变通道数。

---

## 8. 卷积神经网络中的卷积 vs 信号处理中的卷积

| 特性 | 深度学习中的卷积 | 信号处理中的卷积 |
|------|------------------|------------------|
| 核翻转 | 不翻转 | 翻转 $(g[-m, -n])$ |
| 互相关 | 是 | 否 |
| 命名 | 常称"卷积" | 严格区分卷积和互相关 |

**深度学习中的"卷积"实际上是互相关**：

$$\text{corr}(f, g)[i, j] = \sum_{m} \sum_{n} f[i+m, j+n] \cdot g[m, n]$$

由于卷积核是**可学习的**，是否翻转不影响网络的学习能力，因此深度学习框架通常直接实现互相关。

---

## 9. 代码实现：手动实现卷积层

```python
import torch
import torch.nn as nn

class ManualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # 初始化卷积核和偏置
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        batch_size, C_in, H_in, W_in = x.shape

        # 计算输出尺寸
        H_out = (H_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # 填充输入
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = torch.nn.functional.pad(x, (self.padding[1], self.padding[1],
                                            self.padding[0], self.padding[0]))

        # 手动实现卷积
        output = torch.zeros(batch_size, self.out_channels, H_out, W_out)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(0, H_in + 2*self.padding[0] - self.kernel_size[0] + 1, self.stride[0]):
                        for j in range(0, W_in + 2*self.padding[1] - self.kernel_size[1] + 1, self.stride[1]):
                            # 提取输入 patch
                            patch = x[b, ic, i:i+self.kernel_size[0], j:j+self.kernel_size[1]]
                            # 逐元素乘积求和
                            output[b, oc, i//self.stride[0], j//self.stride[1]] += torch.sum(patch * self.weight[oc, ic])

                # 添加偏置
                output[b, oc] += self.bias[oc]

        return output
```

---

## 10. 参考资料

1. **LeCun, Y., et al. (1998)** - "Gradient-based learning applied to document recognition"
2. **Goodfellow, I., et al. (2016)** - "Deep Learning" - Chapter 9
3. **PyTorch Documentation** - `torch.nn.Conv2d`
