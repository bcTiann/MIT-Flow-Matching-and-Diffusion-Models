# Part 4: 训练 VAE 的引言段落解释

## 原文整体在讲什么

第 4 部分要训练一个 **VAE（变分自编码器）**，第 5 部分会在 VAE 学到的"潜在空间"里训练 DiT——这就是 **Stable Diffusion** 的核心套路。

---

## 逐句解释

### 1. "In this section, we'll train a variational autoencoder (VAE) for MNIST."

> 这一节我们要训练一个 **VAE** 用于 MNIST。

**VAE 是什么**：一种神经网络，能把真实图像 $x$ 压缩到一个低维"潜在空间" $z$，再从 $z$ 重建出图像。

和 PCA 思路类似，但更强大、且是概率性的。

---

### 2. "In the next section, we'll then train a diffusion transformer inside of the learned latent space."

> 下一节，我们会在 VAE 学到的潜在空间里训练 diffusion transformer。

**关键概念：Latent Diffusion**

- Part 3 的 DiT：直接在像素空间（32×32 = 1024 维）做扩散
- Part 5 的 DiT：先用 VAE 把图像压到比如 4×4×128，再在这个小空间做扩散

**好处**：
- 维度低 → 计算快
- VAE 已经学过"什么是合理图像" → 扩散更聚焦

这就是 **Stable Diffusion** 的核心思想。

---

### 3. "Recall from the notes that the overall structure of the VAE consists of..."

> 回忆笔记里讲的，VAE 由两部分组成。

---

### 4. Encoder 部分

> an *encoder* $q_\phi(z|x)$, mapping the input `x` with shape `b 1 32 32`, to outputs `z_mean` with shape `b c h w` and learned scalar `z_logvar`.

**编码器** $q_\phi(z|x)$，把输入 $x$（形状 `b 1 32 32`）映射到：

- `z_mean`（形状 `b c h w`）—— 潜在变量的**均值**
- `z_logvar`（**标量**）—— 潜在变量的**对数方差**

**关键点**：

- $\phi$ 是 encoder 的参数
- $q_\phi(z|x)$ 是一个**分布**，不是单个值——它说"给定 $x$，$z$ 服从一个高斯分布"
- 这个高斯由 `z_mean` 和 `z_logvar` 描述：

$$z \sim \mathcal{N}(\text{z\_mean},\ e^{\text{z\_logvar}} I)$$

- `z_logvar` 是**单个数**（所有样本、所有维度共享），不是每像素一个

---

### 5. log-variance 参数化

> Note that for training stability reasons, ... we choose to indirectly parameterize the log-variance $\log \sigma_\phi(x)$.

> 出于训练稳定性考虑，**间接参数化 log-variance**。

**为什么不直接学 $\sigma$（标准差）**：

- $\sigma > 0$ 必须非负，但神经网络 Linear 输出可正可负
- 直接学 $\sigma$ 容易爆炸或变 0，训练不稳定
- **学 $\log \sigma$**：值域是整个实数，使用时 $\sigma = e^{\log \sigma}$ 自动非负

这是 VAE 实现的标准技巧。

---

### 6. Decoder 部分

> a *decoder* $p_\theta(x|z)$ which similarly maps the latent `z` to outputs `x_mean` with shape `b 1 32 32` and learned scalar `x_logvar`.

**解码器** $p_\theta(x|z)$，把潜在变量 $z$ 映射回：

- `x_mean`（形状 `b 1 32 32`）—— 重建图像的均值
- `x_logvar`（标量）—— 重建图像的对数方差

**对应**：

- $\theta$ 是 decoder 的参数
- $p_\theta(x|z)$ 也是分布——给定 $z$，$x$ 服从某个高斯：

$$x \sim \mathcal{N}(\text{x\_mean},\ e^{\text{x\_logvar}} I)$$

- 实际生成时通常直接用 `x_mean` 作为输出图像

---

### 7. 网络结构（block-based）

> we'll architect the encoder (resp. decoder) as a series of **blocks**, each of which contains several residual connections, followed by an attention layer, followed by a downsampling (resp. upsampling) layer with the exception of the final block.

每个 block 包含：

- 几个**残差连接**（ResidualBlock）
- 一层 **attention**（AttnBlock）
- 一层 **下采样**（encoder）或 **上采样**（decoder）—— 最后一个 block 没有

**结构图**：

```
Encoder block:  ResBlock → ResBlock → AttnBlock → Downsample
Decoder block:  ResBlock → ResBlock → AttnBlock → Upsample

(最后一个 block 没有 down/up sampling)
```

---

## 整体数据流直觉

```
原图 x (b, 1, 32, 32)
    ↓ Encoder（block1 → block2 → ... → blockN）
潜在 z (b, c, h, w)   ← 压缩 + 概率化（高斯）
    ↓ Decoder（block1 → block2 → ... → blockN）
重建 x' (b, 1, 32, 32)
```

**Encoder**：图像逐渐变小（H、W 减半），通道逐渐增多
- (b, 1, 32, 32) → (b, 16, 32, 32) → (b, 32, 16, 16) → (b, 64, 8, 8) → (b, 128, 4, 4)

**Decoder**：反过来——通道减少、空间变大

---

## VAE 的训练目标

两部分相加：

### 1. 重建误差（reconstruction loss）

希望 $x' \approx x$：

$$\mathcal{L}_{\text{recon}} = \mathbb{E}_{z \sim q_\phi(z|x)} \left[-\log p_\theta(x|z)\right]$$

直观就是"重建图像和原图差多少"。

### 2. KL 散度（regularization）

希望编码出的潜在分布 $q_\phi(z|x)$ 接近标准高斯 $\mathcal{N}(0, I)$：

$$\mathcal{L}_{\text{KL}} = \text{KL}\big(q_\phi(z|x)\ \|\ \mathcal{N}(0, I)\big)$$

**为什么要这个**：让潜在空间结构良好——之后从 $\mathcal{N}(0, I)$ 采样 $z$ 喂给 decoder 就能生成新图像。

### 总 Loss

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

$\beta$ 是平衡两项的超参数（lab 里设为 10.0）。

---

## 为什么 VAE 比普通 Autoencoder 好

普通 AE：

- $z$ 是确定的点
- 潜在空间没结构，"中间区域"可能解出乱七八糟的东西
- 不能用来生成新样本

VAE：

- $z$ 是分布（带不确定性）
- 训练时 $z$ 在邻域内采样 → 邻近 $z$ 的解码也合理 → 空间**连续 + 平滑**
- 从先验 $\mathcal{N}(0,I)$ 采样 → 解码就是生成

---

## 接下来要实现的组件

按 lab 顺序：

1. **ResidualBlock**：两次 LN + Conv + 激活 + 残差
2. **AttnBlock**：在图像 $(b, c, h, w)$ 上做自注意力
3. **EncoderBlock**：ResBlock + ResBlock + AttnBlock + (Downsample)
4. **Encoder**：把多个 EncoderBlock 串起来 + 输出 z_mean, z_logvar
5. **DecoderBlock**：ResBlock + ResBlock + AttnBlock + (Upsample)
6. **Decoder**：把多个 DecoderBlock 串起来 + 输出 x_mean, x_logvar
7. **VAE**：组装 encoder + decoder + 重参数化采样 + loss
