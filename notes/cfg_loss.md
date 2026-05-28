# CFG Loss Function 解释

## Loss 长这样

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E} \left\| u_t^{\theta}(x|y) - u_t^{\text{ref}}(x|z) \right\|^2$$

其中期望是对这些随机变量取的：

$$z, y \sim p_{\text{data}}(z, y), \quad x \sim p_t(x|z), \quad t \sim \text{Uniform}(0,1)$$

---

## $u_t^{\text{ref}}(x|z)$ 是什么？

**它是我们已知的解析公式，不需要学习。**

对于高斯条件概率路径 $p_t(x|z) = \mathcal{N}(\alpha_t z,\ \beta_t^2 I)$，条件向量场有闭合解：

$$u_t^{\text{ref}}(x|z) = \dot{\alpha}_t z + \frac{\dot{\beta}_t}{\beta_t}(x - \alpha_t z)$$

- $z$ 是真实图像（从 MNIST 采样的）
- $x$ 是在时间 $t$ 的带噪图像（由 $p_t(x|z)$ 采样）
- $\dot{\alpha}_t, \dot{\beta}_t$ 是 $\alpha_t, \beta_t$ 对时间的导数

直觉上，这个公式告诉我们：在时间 $t$、位置 $x$、已知目标图像是 $z$ 时，
"流"应该朝哪个方向走。

---

## $u_t^{\theta}(x|y)$ 是什么？

**它是神经网络，需要学习。**

- 输入：带噪图像 $x$、时间 $t$、标签 $y$（比如数字"8"）
- 输出：预测的向量场方向
- 目标：逼近 $u_t^{\text{ref}}$，但不能用 $z$（测试时不知道真实图像是什么）

---

## 核心对比

| 符号 | 含义 | 已知/未知 |
|------|------|----------|
| $u_t^{\text{ref}}(x \| z)$ | 真实向量场，给定 $z$ 可精确计算 | 训练时已知 |
| $u_t^{\theta}(x \| y)$ | 神经网络预测的向量场，只用标签 $y$ | 需要学习 |

训练时，用 $z$（真实图像）来构造"答案" $u_t^{\text{ref}}$，然后让网络 $u_t^\theta$ 去拟合它。

测试时，只给标签 $y$（比如"生成数字 8"），网络直接输出向量场来引导生成。

---

## 一句话总结

$u_t^{\text{ref}}$ 是"标准答案"（训练时用真实图像算出来的），
$u_t^{\theta}$ 是"学生作答"（神经网络），
Loss 就是让学生答案尽量贴近标准答案。
