# VAE Loss 从零推导

## 第 0 步：最基本的概念

### 什么是"概率分布"？

一个分布告诉你"某个事件出现的可能性有多大"。

例子：身高的高斯分布 $\mathcal{N}(\mu=170, \sigma=10)$
- 身高 170 的人很多（概率密度高）
- 身高 220 的人很少（概率密度低）

写法：$p(x) = \mathcal{N}(x; \mu, \sigma^2)$，意思是"x 服从均值 $\mu$、方差 $\sigma^2$ 的高斯"。

具体公式：

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

---

### 什么是"似然"（likelihood）？

**给定一个分布，看一个具体的样本"发生这件事的概率密度"是多少**。

例子：分布是 $\mathcal{N}(170, 10^2)$，看到一个人身高 175。
- 似然 = $p(175) = \frac{1}{\sqrt{2\pi \cdot 100}}\exp(-\frac{(175-170)^2}{200})$ ≈ 一个具体数字

如果你看到 100 个独立的人，似然就是 100 个 $p(x_i)$ 相乘：

$$\text{Likelihood} = \prod_{i=1}^{100} p(x_i)$$

---

### 什么是"最大化似然"？

我们手头有一堆数据（比如 100 个人的身高），但**不知道分布的参数 $\mu, \sigma$**。

策略：找一组 $\mu, \sigma$，让"看到这堆数据的可能性"**最大**。

也就是 maximize：

$$\prod_i p(x_i; \mu, \sigma)$$

直觉：选那组参数，使得"刚好看到这些数据"是最自然的事情。

---

### 为什么取对数？

连乘很多很小的数 → 数值下溢（下溢到 0）。

取 log 把连乘变成连加，方便计算且不会下溢：

$$\log \prod_i p(x_i) = \sum_i \log p(x_i)$$

最大化对数似然 ≡ 最大化似然（log 单调）。

---

### 为什么取负？

机器学习里习惯**最小化** loss，所以把"最大化对数似然"变成"最小化负对数似然"：

$$\text{NLL} = -\sum_i \log p(x_i)$$

**这就是机器学习里大量出现的"loss = -log p(x)"的来源**。

---

## 第 1 步：套用到 VAE 的 Decoder

### Decoder 的设定

Decoder 输入 latent $z$，输出图像 $x$。

VAE 把这一步建模成**条件高斯分布**：

$$p_\theta(x|z) = \mathcal{N}(x;\ \mu_\theta(z),\ \sigma_\theta^2 I)$$

意思是：给定 $z$，$x$ 是个高斯分布，均值是 $\mu_\theta(z)$（神经网络 decoder 的输出），方差是 $\sigma_\theta^2$。

- $\mu_\theta(z)$ 在代码里就是 `x_mean`
- $\sigma_\theta^2 = e^{\text{x\_logvar}}$（学的是 logvar）

---

### 我们想最大化什么？

希望"给定 $z$，模型预测的分布把真实图像 $x$ 当作高概率事件"。

最大化似然 → 最小化负对数似然：

$$\mathcal{L}_{\text{recon}} = -\log p_\theta(x|z)$$

---

### 代入高斯公式

$d$ 维高斯（$d$ = 像素总数）的对数：

$$\log p_\theta(x|z) = -\frac{d}{2}\log(2\pi) - \frac{d}{2}\log\sigma_\theta^2 - \frac{\|x - \mu_\theta(z)\|^2}{2\sigma_\theta^2}$$

取负，去掉常数 $\frac{d}{2}\log(2\pi)$（不依赖参数）：

$$\mathcal{L}_{\text{recon}} = \underbrace{\frac{1}{2\sigma_\theta^2}\|x - \mu_\theta(z)\|^2}_{\text{重建误差}} + \underbrace{\frac{d}{2}\log\sigma_\theta^2}_{\text{decoder confidence}}$$

---

### 这两项的直觉

**第一项**：$x$ 和重建 $\mu_\theta(z)$ 的差距，**除以方差**。

- 方差小 → 第一项大 → 重建误差被严厉惩罚（"我很自信，所以你必须准"）
- 方差大 → 第一项被稀释 → 重建误差不太重要

**第二项**：$\log\sigma_\theta^2$，方差越大这项越大。

- 防止 decoder 偷懒：把 $\sigma^2$ 调到无穷大让第一项趋于 0
- 第二项把它"拽回来"

两项一起平衡："要多自信"和"要多准"。

---

## 第 2 步：Encoder 部分（KL 散度）

### Encoder 的设定

Encoder 输入图像 $x$，输出**一个分布**（不是单个 $z$）：

$$q_\phi(z|x) = \mathcal{N}(z;\ \mu_\phi(x),\ \sigma_\phi^2 I)$$

- $\mu_\phi(x)$ = `z_mean`
- $\sigma_\phi^2 = e^{\text{z\_logvar}}$

---




### 为什么 encoder 输出分布而不是单个值？

普通 autoencoder（AE）：encoder → 单个 $z$ → decoder → $\hat x$
- 问题：$z$ 是某个固定点，潜在空间没结构。$z$ 之间的"中间区域"解出来可能是垃圾。

VAE：encoder → $z$ 的分布 → 采样 $z$ → decoder
- 训练时 $z$ 在某个邻域内随机采，decoder 必须**对邻近 $z$ 都给出合理输出** → 潜在空间**连续 + 平滑**。
- 训练完了之后从 $\mathcal{N}(0, I)$ 采样就能生成新图。

---

### 但有个问题：随便编码的话，潜在空间还是乱的

如果 encoder 想怎么编就怎么编，可能：
- 数字 0 编到 $z = 100$ 附近
- 数字 1 编到 $z = -50$ 附近
- 中间区域没有任何意义

我们希望**所有图像编码出来的 $z$ 集中在 $\mathcal{N}(0, I)$ 附近**——这样从 $\mathcal{N}(0, I)$ 采样就能生成新图。

---

### 怎么衡量"两个分布有多像"？

**KL 散度（KL divergence）**：

$$D_{\text{KL}}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx$$

性质：

- $\geq 0$
- $= 0$ 当且仅当 $q = p$（完全相同）
- 越大越不像

**用 KL 散度作为"距离"：让 encoder 的分布 $q_\phi(z|x)$ 接近 $\mathcal{N}(0, I)$**。

---

### 两个高斯之间的 KL（PDF Display 80）

设 $q = \mathcal{N}(\mu_q, \sigma_q^2)$，$p = \mathcal{N}(\mu_p, \sigma_p^2)$（一维），KL 公式可以推出来：

$$D_{\text{KL}}(q\|p) = \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2} - \frac{1}{2}$$

---

### 取 $p = \mathcal{N}(0, 1)$（标准高斯）

$\mu_p = 0, \sigma_p^2 = 1$ 代入：

$$D_{\text{KL}}(q\|\mathcal{N}(0,1)) = -\frac{1}{2}\log\sigma_q^2 + \frac{\sigma_q^2 + \mu_q^2}{2} - \frac{1}{2}$$

整理：

$$D_{\text{KL}} = \frac{1}{2}\left( \mu_q^2 + \sigma_q^2 - \log\sigma_q^2 - 1 \right)$$

多维（每维独立）就求和：

$$\mathcal{L}_{\text{KL}} = \frac{1}{2} \sum_j \left( \mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1 \right)$$

---

### 直觉

每一项都是在惩罚某个偏离：

- $\mu_j^2$：让均值 $\mu_j \to 0$
- $\sigma_j^2 - \log\sigma_j^2 - 1$：这个函数最小值在 $\sigma_j^2 = 1$。展开看：
  - $\sigma^2 = 1$ 时：$1 - 0 - 1 = 0$ ✓
  - $\sigma^2 = 0.1$ 时：$0.1 - (-2.3) - 1 = 1.4$（被惩罚）
  - $\sigma^2 = 10$ 时：$10 - 2.3 - 1 = 6.7$（被惩罚）

合起来：让 $q_\phi(z|x) \to \mathcal{N}(0, I)$。

---

## 第 3 步：合起来 = VAE Loss

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

具体：

$$\mathcal{L}_{\text{VAE}} = \underbrace{\frac{1}{2\sigma_\theta^2}\|x - \mu_\theta(z)\|^2 + \frac{d}{2}\log\sigma_\theta^2}_{\text{重建}} + \beta \cdot \underbrace{\frac{1}{2}\sum\left( \mu_\phi^2 + \sigma_\phi^2 - \log\sigma_\phi^2 - 1 \right)}_{\text{KL}}$$

$\beta$ 平衡两个目标：
- $\beta$ 大 → 更强迫 latent 接近高斯（生成质量好，但重建可能差）
- $\beta$ 小 → 更看重重建，可能 latent 不够规整

Lab 里 $\beta = 10$。

---

## 第 4 步：但还差一个问题——梯度怎么传？

### 问题

Loss 里有 $z \sim q_\phi(z|x)$ 的**采样**操作。采样是随机的，**不可导**。

如果 $z = \text{sample}(q_\phi)$，那 $\nabla_\phi z$ 没有定义，无法反向传播。

---

### 重参数化技巧（Reparameterization Trick）

把"随机采样"改写成"确定计算 + 噪声"：

$$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

数学上：$\mu + \sigma \epsilon$ 服从 $\mathcal{N}(\mu, \sigma^2)$，和原来等价。

**关键差别**：随机性现在只在 $\epsilon$ 里，$\epsilon$ 与参数 $\phi$ 无关。

$\mu_\phi$ 和 $\sigma_\phi$ 是**确定函数**（神经网络的输出），可导。

---

### Lab 代码里的对应

```python
def forward(self, x):
    z_mean, z_logvar = self.encode(x)
    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
    #     ↑              ↑                    ↑
    #     μ_φ            σ_φ = exp(0.5·logσ²) ε ~ N(0, I)
    x_mean, x_logvar = self.decode(z)
    return z_mean, z_logvar, x_mean, x_logvar
```

这样 $z$ 关于 $\phi$ 可导，整条计算图可微，可以正常反向传播。

---

## 第 5 步：为什么学 $\log\sigma^2$ 而不是 $\sigma$？

### 1. 数值稳定

$\sigma^2 > 0$ 必须保证。神经网络 Linear 层输出可正可负，没有"大于 0"的天然约束。

- 直接学 $\sigma^2$：要加 softplus 或 exp 强制为正 → 多一层非线性，可能数值问题
- **学 $\log\sigma^2$**：值域是整个实数 → 自由
- 用的时候 $\sigma^2 = e^{\log\sigma^2}$ → 自动 > 0

### 2. 公式更简洁

KL 公式里本来就有 $\log\sigma^2$ 这一项，直接当变量更省事。

---

## 第 6 步：写代码（字符级对应）

Lab 里 `z_logvar` 和 `x_logvar` 都是**标量**（所有维度共享一个）。

---

### KL Loss 逐部分对应

#### 数学公式

$$\mathcal{L}_{\text{KL}} = \frac{1}{2}\sum\left(\mu_\phi^2 + \sigma_\phi^2 - \log\sigma_\phi^2 - 1\right)$$

#### 代码

```python
kl_loss = 0.5 * (z_mean**2 + z_logvar.exp() - z_logvar - 1).sum() / batch_size
```

#### 字符级对应表

| 数学符号 | 代码 | 解释 |
|---------|------|------|
| $\frac{1}{2}$ | `0.5 *` | 公式前面的 1/2 |
| $\mu_\phi^2$ | `z_mean**2` | $\mu_\phi$ 就是 `z_mean`，平方 |
| $\sigma_\phi^2$ | `z_logvar.exp()` | $\sigma^2 = e^{\log\sigma^2}$，所以是 `exp(z_logvar)` |
| $-\log\sigma_\phi^2$ | `- z_logvar` | $\log\sigma^2$ 就是 `z_logvar` 本身 |
| $-1$ | `- 1` | 常数 |
| $\sum$ | `.sum()` | 对所有维度求和 |
| 取平均 | `/ batch_size` | 公式是单样本的，需要除以 batch_size |

---

### Recon Loss 逐部分对应

#### 数学公式

$$\mathcal{L}_{\text{recon}} = \frac{1}{2\sigma_\theta^2}\|x - \mu_\theta\|^2 + \frac{d}{2}\log\sigma_\theta^2$$

其中：
- $\sigma_\theta^2$ = decoder 的方差 = `x_logvar.exp()`
- $\mu_\theta$ = decoder 的均值 = `x_mean`
- $x$ = 真实图像 = `x_true`
- $d$ = 单个样本的像素数（lab 里 1×32×32 = 1024）

---

#### 第一项：$\frac{1}{2\sigma_\theta^2}\|x - \mu_\theta\|^2$

| 数学符号 | 代码 | 解释 |
|---------|------|------|
| $\|x - \mu_\theta\|^2$ | `(x_true - x_mean).pow(2).sum()` | 所有像素差的平方求和 |
| $\frac{1}{2\sigma_\theta^2}$ | `0.5 / x_logvar.exp()` | $\sigma^2 = e^{\log\sigma^2}$ |

合起来：

```python
term1 = 0.5 * (x_true - x_mean).pow(2).sum() / x_logvar.exp()
```

---

#### 第二项：$\frac{d}{2}\log\sigma_\theta^2$

##### `d` 是怎么来的？

要理解 `d` 从哪里冒出来，先回顾几个基础。

---

**基础 1：什么是"维度"**

一张 32×32 灰度图，本质上是 $32 \times 32 = 1024$ 个数字。

把它当成一个 **1024 维的向量** $\mathbf{x} = (x_1, x_2, \dots, x_{1024})$，每个 $x_i$ 是一个像素的值。

所以 $d$ 是**向量维度数 = 像素总数**。

---

**基础 2：1 维高斯密度公式**

单个像素 $x_i$ 服从 1 维高斯（之前讲过）：

$$p(x_i) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x_i-\mu_i)^2}{2\sigma^2}\right)$$

---

**基础 3：多个独立变量的联合密度 = 各自密度的乘积**

VAE 的假设：**decoder 输出的每个像素是独立高斯**（这就是"各向同性高斯"的意思）。

如果像素之间互相独立：

$$p(x_1, x_2, \dots, x_d) = p(x_1) \cdot p(x_2) \cdots p(x_d) = \prod_{i=1}^{d} p(x_i)$$

这是概率论的基本规则——独立事件的联合概率 = 各自概率相乘。

---

**基础 4：把单像素公式套进来**

每个 $p(x_i)$ 用基础 2 的公式：

$$p(\mathbf{x}) = \prod_{i=1}^{d}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x_i-\mu_i)^2}{2\sigma^2}\right)$$

---

**化简连乘**

分两部分：

**(a) 系数部分**：$d$ 个 $\frac{1}{\sqrt{2\pi\sigma^2}}$ 相乘：

$$\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^d = \frac{1}{(2\pi\sigma^2)^{d/2}}$$

**(b) 指数部分**：$\exp$ 相乘 = 指数相加：

$$\exp\left(-\sum_{i=1}^d \frac{(x_i - \mu_i)^2}{2\sigma^2}\right) = \exp\left(-\frac{\|\mathbf{x}-\boldsymbol{\mu}\|^2}{2\sigma^2}\right)$$

（因为 $\sum_i (x_i - \mu_i)^2 = \|\mathbf{x} - \boldsymbol{\mu}\|^2$，向量差的平方范数。）

合起来：

$$p(\mathbf{x}) = \frac{1}{(2\pi\sigma^2)^{d/2}}\exp\!\left(-\frac{\|\mathbf{x}-\boldsymbol{\mu}\|^2}{2\sigma^2}\right)$$

---

**取对数**

$$\log p(\mathbf{x}) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{\|\mathbf{x}-\boldsymbol{\mu}\|^2}{2\sigma^2}$$

把 $\log(2\pi\sigma^2)$ 拆成 $\log(2\pi) + \log\sigma^2$：

$$\log p(\mathbf{x}) = -\frac{d}{2}\log(2\pi) - \frac{d}{2}\log\sigma^2 - \frac{\|\mathbf{x}-\boldsymbol{\mu}\|^2}{2\sigma^2}$$

**`d` 就从这里来**——因为有 $d$ 个独立维度相乘，log 之后就出现了 $\frac{d}{2}\log\sigma^2$。

---

**取负去掉常数**

$\frac{d}{2}\log(2\pi)$ 不依赖任何参数，是常数，求梯度时为 0，可以忽略：

$$-\log p(\mathbf{x}) = \underbrace{\frac{d}{2}\log\sigma^2}_{\text{第二项}} + \underbrace{\frac{\|\mathbf{x}-\boldsymbol{\mu}\|^2}{2\sigma^2}}_{\text{第一项}}$$

---

##### Lab 里 $d$ 等于多少？

每张图是 `1 × 32 × 32 = 1024` 个像素。每个像素被建模成一维高斯，**总共 1024 个独立维度**。

所以 $d = 1024$。

代码 `x_true[0].numel()`：
- `x_true.shape = (batch_size, 1, 32, 32)`
- `x_true[0]` 取第一个样本，形状 `(1, 32, 32)`
- `.numel()` 算元素总数 = 1 × 32 × 32 = **1024**

---

##### 字符级对应

| 数学符号 | 代码 | 解释 |
|---------|------|------|
| $d$ | `x_true[0].numel()` | 单样本像素数（如 1024）|
| $\frac{d}{2}$ | `0.5 * d` | $d/2$ |
| $\log\sigma_\theta^2$ | `x_logvar` | logvar 本身就是 log σ² |

---

##### 为什么 batch_size 也乘进来？

公式 $\frac{d}{2}\log\sigma^2$ 是**一个样本**的 loss。

batch 里有 `batch_size` 个样本，每个都有这一项：

$$\text{总和} = \sum_{i=1}^{\text{batch\_size}} \frac{d}{2}\log\sigma_\theta^2 = \text{batch\_size} \cdot \frac{d}{2}\log\sigma_\theta^2$$

因为 $\log\sigma^2$ 是标量（每个样本共享），所以这就是简单乘法：

```python
term2 = batch_size * 0.5 * d * x_logvar
```

最后整体再除以 `batch_size` 取平均：

```python
recon_loss = (term1 + term2) / batch_size
```

`term2 / batch_size = 0.5 * d * x_logvar` —— 刚好就是单样本的 $\frac{d}{2}\log\sigma^2$。

---

#### 合起来再除以 batch_size

```python
recon_loss = (term1 + term2) / batch_size
```

---

### 完整 compute_loss

```python
def compute_loss(self, z_mean, z_logvar, x_mean, x_logvar, x_true):
    batch_size = x_true.shape[0]
    d = x_true[0].numel()  # 单样本像素数（1×32×32 = 1024）

    # KL loss: 让 latent 接近 N(0, I)
    kl_loss = 0.5 * (
        z_mean**2 + z_logvar.exp() - z_logvar - 1
    ).sum() / batch_size

    # Reconstruction loss: -log p(x|z)
    recon_loss = (
        0.5 * (x_true - x_mean).pow(2).sum() / x_logvar.exp()    # 第一项
        + batch_size * 0.5 * d * x_logvar                         # 第二项
    ) / batch_size

    return self.beta * kl_loss + recon_loss
```

注意是 `self.beta * kl_loss`（β 在 KL 上），不是 `kl_loss + recon_loss`。

---

### 验证 shape

```
z_mean:   (b, c, h, w)   # 比如 (64, 128, 4, 4)
z_logvar: ()             # 标量
x_mean:   (b, 1, 32, 32)
x_logvar: ()             # 标量
x_true:   (b, 1, 32, 32)
```

- `(z_mean**2).sum()` → 标量 ✓
- `z_logvar.exp()` → 标量，可以广播
- `(x_true - x_mean).pow(2).sum()` → 标量 ✓
- 最终 `kl_loss`, `recon_loss` 都是标量 ✓

---

## 一句话总结整个推导

1. **似然**：分布在某个数据点的概率密度
2. **最大化似然**：找参数让数据"最自然"
3. **取 -log**：变成最小化 NLL（机器学习习惯）
4. **VAE 的两个分布**：decoder $p_\theta(x|z)$ 和 encoder $q_\phi(z|x)$
5. **重建 loss** = -log p_θ(x|z) 展开后两项
6. **KL loss** = 让 q_φ 接近标准高斯（用 KL 散度衡量）
7. **重参数化** = 让采样可导
8. **总 loss** = recon + β · KL

每一步都有数学动机，不是凭空冒出来的公式。
