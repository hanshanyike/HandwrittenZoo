# Denoising Diffusion Probabilistic Model (DDPM) - 去噪扩散概率模型

## 算法简介

去噪扩散概率模型 (DDPM) 是 Ho 等人于 2020 年提出的生成模型，它通过模拟物理中的扩散现象来生成数据: 先将数据逐步加噪直至变为纯高斯噪声，再训练神经网络学习逆向去噪过程。DDPM 是 Stable Diffusion、DALL-E 2、Imagen 等当前顶尖图像生成系统的核心技术基础。

## 核心思想

DDPM 包含两个马尔可夫链过程:

1. **前向过程 (Forward / Diffusion Process)**: 预设的加噪过程，在 $T$ 个时间步内逐步向数据 $x_0$ 添加高斯噪声，最终 $x_T$ 近似于标准正态分布。该过程没有可学习参数，且利用累积分布的性质，可以直接从 $x_0$ 跳到任意 $x_t$。
2. **反向过程 (Reverse / Denoising Process)**: 用神经网络 $\epsilon_\theta(x_t, t)$ 预测每一步添加的噪声，逐步从 $x_T$ 去噪恢复出 $x_0$。训练目标非常简单: 最小化预测噪声与真实噪声的 MSE。

相比 GAN，DDPM 训练更稳定 (无需对抗训练)；相比 VAE，DDPM 生成质量更高且支持渐进式生成。

## 数学公式

### 1. 前向过程 (加噪)
前向过程是马尔可夫链，每步添加少量高斯噪声:
$$q(x_t | x_{t-1}) = N(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

其中 $\beta_t \in (0, 1)$ 为噪声调度 (noise schedule)，通常从 $10^{-4}$ 线性增加到 $0.02$。

令 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，利用重参数化可得直接从 $x_0$ 到 $x_t$ 的闭式解:
$$q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim N(0, I)$$

### 2. 反向过程 (去噪)
反向过程从 $p(x_T) = N(0, I)$ 开始，逐步去噪:
$$p_\theta(x_{t-1} | x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

其中均值和方差由神经网络参数化。在简化版 DDPM 中，方差固定为 $\Sigma_\theta = \beta_t I$ 或 $\tilde{\beta}_t I$，网络只学习均值 (等价于预测噪声)。

### 3. 训练目标
Ho 等人发现，直接预测噪声比预测均值效果更好。训练目标为:
$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon \sim N(0, I), t \sim [1, T]} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$。

### 4. 采样 (去噪迭代)
从 $x_T \sim N(0, I)$ 开始，对于 $t = T, T-1, \ldots, 1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} z$$

其中 $z \sim N(0, I)$ (当 $t > 1$)，$z = 0$ (当 $t = 1$)。

### 5. 后验方差
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

## 时间/空间复杂度

- **训练时间复杂度**: $O(T \cdot B \cdot D \cdot H)$，每轮训练随机采样一个时间步 $t$，单次前向传播。
- **采样时间复杂度**: $O(T \cdot B \cdot D \cdot H)$，需要迭代 $T$ 步 (通常 $T=1000$)，这是 DDPM 的主要瓶颈。
- **空间复杂度**: $O(B \cdot D + T)$，存储数据及噪声调度参数。
- **与 GAN 对比**: DDPM 训练稳定、生成多样性好，但采样慢 (需多步)；GAN 采样快 (单步) 但训练困难。
- **与 VAE 对比**: DDPM 生成质量更高，无编码器-解码器结构，直接建模数据分布。

## 面试高频考点

### 1. 扩散模型与 GAN、VAE 的核心区别是什么？
**答案**:
- **GAN**: 通过对抗训练学习生成器，采样快 (单步)，但训练不稳定、易模式崩溃。
- **VAE**: 通过编码器-解码器结构学习潜在空间，训练稳定，但生成样本较模糊。
- **Diffusion**: 通过逐步去噪生成样本，训练最稳定 (回归损失)，生成质量最高，但采样需多步迭代 (慢)。

### 2. 为什么前向过程可以直接从 x_0 跳到 x_t？
**答案**:
因为每一步加噪都是高斯分布的线性组合，高斯分布的叠加仍为高斯分布。累积后:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
其中 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。这使得训练时可以随机采样任意时间步 $t$，无需逐步迭代加噪，极大提升训练效率。

### 3. 为什么训练目标是预测噪声，而不是直接预测 x_{t-1} 或均值？
**答案**:
Ho 等人在论文中对比了多种参数化方式，发现直接预测噪声 $\epsilon$ 效果最佳，原因包括:
- 噪声的分布是标准正态，均值为 0，方差为 1，学习难度较低。
- 预测噪声等价于预测均值，但数值稳定性更好。
- 与直接预测 $x_0$ 相比，预测噪声使网络在不同时间步的任务难度更均衡。

### 4. DDPM 采样慢的问题如何解决？
**答案**:
- **DDIM (Denoising Diffusion Implicit Models)**: 将扩散过程改为非马尔可夫链，支持跳步采样，可将步数从 1000 降到 50 甚至 10 步。
- **Progressive Distillation**: 训练学生模型一步模拟教师模型多步的去噪效果。
- **Latent Diffusion (Stable Diffusion)**: 在压缩的潜在空间 (latent space) 而非像素空间进行扩散，降低计算维度。
- **Consistency Models**: 学习将任意时刻的加噪数据直接映射到干净数据，实现单步生成。

### 5. beta 调度 (noise schedule) 的作用是什么？常见的调度方式有哪些？
**答案**:
beta 调度控制每一步添加噪声的量，直接影响训练稳定性和生成质量:
- **线性调度 (Linear)**: $\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$，DDPM 原始论文使用。
- **余弦调度 (Cosine)**: 使信号保留比例 $\bar{\alpha}_t$ 按余弦曲线下降， Nichol & Dhariwal 提出，生成质量通常更好。
- **Sigmoid 调度**: 在两端平滑过渡，中间变化较快。
合适的调度应保证早期步骤保留足够信号 (便于网络学习)，后期步骤充分加噪 (接近标准正态)。

## 代码解析

### 正弦时间步嵌入
```python
def timestep_embedding(timesteps, dim, max_period=10000):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```
将离散时间步 $t$ 映射到连续向量，使网络感知当前去噪阶段。不同频率的正弦/余弦组合使模型能够区分不同时间尺度。

### 前向加噪 (重参数化)
```python
sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None]
sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None]
x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
```
利用闭式解直接计算 $x_t$，无需逐步迭代。`register_buffer` 确保这些预计算参数随模型移动到 GPU，但不参与梯度更新。

### 训练损失
```python
t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
noise_pred = self.model(x_t, t)
loss = F.mse_loss(noise_pred, noise)
```
随机采样时间步 $t$，让网络预测添加的噪声。MSE 损失简单稳定，是 DDPM 训练成功的关键之一。

### 反向采样
```python
model_mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha_cumprod * noise_pred)
x_t = model_mean + torch.sqrt(posterior_variance) * noise
```
根据网络预测的噪声，计算去噪后的均值，并添加适当方差的高斯噪声 (最后一步不加)。迭代 $T$ 次后得到生成样本。

## 参考资料

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [DDIM: Denoising Diffusion Implicit Models (Song et al., 2021)](https://arxiv.org/abs/2010.02502)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)
- [PyTorch-DDPM 实现参考](https://github.com/LinXueyuanStdio/PyTorch-DDPM)
- [labml.ai DDPM 详解](https://nn.labml.ai/diffusion/ddpm/index.html)
