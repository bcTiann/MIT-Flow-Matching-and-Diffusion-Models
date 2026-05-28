# CFG 核心公式推导

$$b_t \nabla_x \log p_t(y|x) = u_t^{\text{target}}(x|y) - u_t^{\text{target}}(x)$$

---

## 第一步：贝叶斯定理作用于梯度

由贝叶斯定理：

$$p_t(x|y) = \frac{p_t(y|x)\, p_t(x)}{p_t(y)}$$

两边取对数：

$$\log p_t(x|y) = \log p_t(y|x) + \log p_t(x) - \log p_t(y)$$

对 $x$ 求梯度（$p_t(y)$ 与 $x$ 无关，梯度为零）：

$$\nabla_x \log p_t(x|y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y|x)$$

整理：

$$\nabla_x \log p_t(y|x) = \nabla_x \log p_t(x|y) - \nabla_x \log p_t(x)$$

---

## 第二步：高斯路径下向量场与 score 的关系

对于高斯条件概率路径 $p_t(x|z) = \mathcal{N}(\alpha_t z,\ \beta_t^2 I)$，条件向量场为：

$$u_t(x|z) = \dot{\alpha}_t z + \frac{\dot{\beta}_t}{\beta_t}(x - \alpha_t z)$$

对 $z$ 求期望得到目标向量场：

$$u_t^{\text{target}}(x) = \mathbb{E}[u_t(x|z) \mid x_t = x]$$

利用 **Tweedie 公式**（高斯分布特有的结论）：

$$\mathbb{E}[z \mid x_t = x] = \frac{x}{\alpha_t} + \frac{\beta_t^2}{\alpha_t} \nabla_x \log p_t(x)$$

代入后化简，得到：

$$u_t^{\text{target}}(x) = \frac{\dot{\alpha}_t}{\alpha_t} x + b_t \cdot \nabla_x \log p_t(x)$$

其中：

$$b_t = \frac{\beta_t(\dot{\alpha}_t \beta_t - \alpha_t \dot{\beta}_t)}{\alpha_t}$$

**有条件标签 $y$ 时**，推导完全一样，只是把 $p(z)$ 换成 $p(z|y)$：

$$u_t^{\text{target}}(x|y) = \frac{\dot{\alpha}_t}{\alpha_t} x + b_t \cdot \nabla_x \log p_t(x|y)$$

---

## 第三步：两式相减

$$u_t^{\text{target}}(x|y) - u_t^{\text{target}}(x)
= b_t \left[\nabla_x \log p_t(x|y) - \nabla_x \log p_t(x)\right]$$

代入第一步的结论：

$$\boxed{u_t^{\text{target}}(x|y) - u_t^{\text{target}}(x) = b_t \cdot \nabla_x \log p_t(y|x)}$$

---

## 直觉理解

- $\nabla_x \log p_t(y|x)$ 表示"朝着让标签 $y$ 更可能出现的方向"
- CFG 的做法是把这个方向放大：

$$u_t^{\text{CFG}}(x|y) = u_t(x) + w \cdot [u_t(x|y) - u_t(x)]$$

$w > 1$ 时就是在加强条件引导，$w = 0$ 时退化为无条件生成。
