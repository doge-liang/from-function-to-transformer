# 第十一章：生成式模型（二）Diffusion与Flow


> 现代生成模型的前沿技术

---

## 11.1 扩散模型（Diffusion Models）

### 11.1.1 核心思想

**扩散模型**：逐步加噪破坏数据，逐步去噪生成新样本

```

正向过程（加噪）：
x0 (原始图像) → x1 → x2 → ... → xT (纯噪声)


反向过程（去噪）：
xT (纯噪声) → xT-1 → ... → x0 (生成图像)
```


### 11.1.2 前向扩散过程

$$q(x_t \mid x_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

其中$\beta_t$是每步的噪声方差。

### 11.1.3 反向去噪过程

$$p_\theta(x_{t-1} \mid x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

训练神经网络预测每步的去噪量。

### 11.1.4 DDPM训练

```python
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            # ... 更多卷积层
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # 预测噪声
        noise_pred = self.model(x)
        return noise_pred

# 简化的训练循环
def train_diffusion(model, dataloader, T=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for images, _ in dataloader:
            # 随机采样时间步
            t = torch.randint(0, T, (images.size(0),))

            # 添加噪声
            noise = torch.randn_like(images)
            noisy_images = forward_diffusion(images, noise, t)

            # 预测噪声
            noise_pred = model(noisy_images, t)

            # MSE损失
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```


### 11.1.5 Stable Diffusion

**组成**：

- VAE：文本条件编码
- U-Net：去噪网络
- CLIP：文本编码器

---

## 11.2 归一化流（Normalizing Flows）

### 11.2.1 核心思想

**归一化流**：通过可逆变换将复杂分布映射到简单分布

```

复杂分布 p(x) ──→ 可逆变换 f ──→ 简单分布 p(z) = N(0,I)


生成：z ~ N(0,I) → x = f^-1(z)
```


### 11.2.2 变换公式

$$p(x) = p(z) \left|\det \frac{\partial z}{\partial x}\right|$$

其中 $\left|\det \frac{\partial z}{\partial x}\right|$ 是雅可比行列式。

### 11.2.3 RealNVP

```python
class CouplingLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net_s = nn.Conv2d(in_channels//2, in_channels, 3, padding=1)
        self.net_t = nn.Conv2d(in_channels//2, in_channels, 3, padding=1)

    def forward(self, x, reverse=False):
        # 分割
        x1, x2 = x.chunk(2, dim=1)

        s = torch.tanh(self.net_s(x2))
        t = self.net_t(x2)

        if reverse:
            y1 = (x1 - t) * torch.exp(-s)
            y2 = x2
        else:
            y1 = x1 * torch.exp(s) + t
            y2 = x2

        y = torch.cat([y1, y2], dim=1)
        logdet = torch.sum(s, dim=[1,2,3])

        return y, logdet
```


---

## 11.3 对比

| 特性 | Diffusion | Flow |
|------|-----------|------|
| 训练稳定性 | 稳定 | 稳定 |
| 生成质量 | 高 | 高 |
| 采样步数 | 多（1000+）| 少（几十）|
| 潜在空间 | 噪声空间 | 明确映射 |
| 应用 | 文本到图像 | 图像到图像 |

---

## 本章小结

**核心概念**：

1. ✅ 扩散模型：加噪→去噪
2. ✅ 归一化流：可逆变换
3. ✅ Stable Diffusion：文本到图像生成
4. ✅ RealNVP：仿射耦合层

---

## 思考题

1. 扩散模型和VAE的核心区别？
2. 为什么扩散模型采样步数多？
3. 归一化流中雅可比行列式的作用？

---

## 下一步

下一章我们将学习**大规模语言模型（LLM）**：

- BERT vs GPT
- 指令微调
- 对齐方法
- LLM的应用与局限

准备好进入现代AI的核心了吗？
