# 第十三章：强化学习基础

> 智能体如何通过交互学习最优策略

---

## 13.1 强化学习核心

### 13.1.1 关键要素

```
智能体(Agent) ——动作(Action)——>环境(Environment)
    ↑                                 |
    |---状态(State)—奖励(Reward)---|

```

| 要素 | 含义 |
|------|------|
| 智能体 | 学习决策的系统 |
| 环境 | 交互的对象 |
| 状态 | 当前情况 |
| 动作 | 智能体的行为 |
| 奖励 | 反馈信号 |

### 13.1.2 与其他学习范式

| 类型 | 数据 | 目标 |
|------|------|------|
| 监督学习 | $(x, y)$ | 拟合$P(y\|x)$ |
| 强化学习 | $(s, a, r, s')$ | 最大化累积奖励 |

---

## 13.2 马尔可夫决策过程（MDP）

### 13.2.1 MDP定义

$$MDP = (S, A, P, R, \gamma)$$

### 13.2.2 回报与价值函数

**回报**：
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**价值函数**：
$$V(s) = E[G_t \mid s_t = s]$$

**Q函数**：
$$Q(s, a) = E[G_t \mid s_t = s, a_t = a]$$

### 13.2.3 贝尔曼方程

$$V(s) = \max_a [R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s')]$$

---

## 13.3 Q-Learning

### 13.3.1 Q-Learning更新

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### 13.3.2 代码实现

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
```

---

## 13.4 深度Q网络（DQN）

### 13.4.1 DQN架构

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)
```

### 13.4.2 经验回放

```python
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
```

---

## 本章小结

**核心概念**：

1. ✅ 强化学习：智能体与环境交互
2. ✅ MDP：状态、动作、奖励、转移
3. ✅ Q-Learning：值函数迭代
4. ✅ DQN：深度神经网络近似Q函数
5. ✅ 经验回放：打破数据相关性

---

## 思考题

1. 马尔可夫性质对强化学习为什么重要？
2. 探索与利用的权衡如何处理？
3. 为什么DQN需要经验回放？

---

## 下一步

下一章我们将学习**RLHF**，将强化学习应用于大语言模型：

- 奖励模型训练
- PPO微调LLM
- 对齐人类价值观

准备好探索智能体的对齐方法了吗？
