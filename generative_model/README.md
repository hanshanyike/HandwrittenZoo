# Generative Models - 生成模型

## 类别概述

生成模型 (Generative Models) 是机器学习领域中一类学习数据分布并能够生成新样本的模型。与判别模型 (Discriminative Models) 只关注 $p(y|x)$ 不同，生成模型建模联合分布 $p(x, y)$ 或边缘分布 $p(x)$，从而支持采样、插值、异常检测等任务。

本目录涵盖了三类最具代表性的深度生成模型，它们分别代表了生成模型发展的三个重要阶段:

| 模型 | 核心思想 | 训练方式 | 生成质量 | 采样速度 | 训练稳定性 |
|------|---------|---------|---------|---------|-----------|
| **VAE** | 概率编码 + 重参数化 + KL 约束 | 最大化 ELBO | 中等 (较模糊) | 快 (单步) | 高 |
| **GAN** | 生成器 vs 判别器对抗博弈 | Minimax 对抗训练 | 高 (锐利) | 快 (单步) | 低 |
| **Diffusion** | 前向加噪 + 反向逐步去噪 | 噪声预测 (MSE) | 极高 | 慢 (多步) | 极高 |

## 文件结构

```
generative_model/
├── vae.py          # 变分自编码器实现
├── vae.md          # VAE 算法详解与面试考点
├── gan.py          # 生成对抗网络实现
├── gan.md          # GAN 算法详解与面试考点
├── diffusion.py    # 去噪扩散概率模型实现
├── diffusion.md    # Diffusion 算法详解与面试考点
└── README.md       # 本文件
```

## 面试聚焦

生成模型是算法工程师面试中的**高频核心考点**，尤其在大模型和 AIGC 方向。以下是本类别覆盖的重点面试主题:

### 1. 模型原理与对比
- VAE 的重参数化技巧为什么有效？编码器为什么输出 log_var？
- GAN 的 minimax 目标函数如何推导？最优判别器形式是什么？
- Diffusion 的前向过程为什么可以直接从 $x_0$ 跳到 $x_t$？
- 三类模型的优缺点对比: 何时选 VAE / GAN / Diffusion？

### 2. 训练技巧与问题诊断
- GAN 训练不稳定的原因是什么？模式崩溃如何解决？
- 为什么 GAN 的生成器损失常用 $-\log D(G(z))$ 而非 $\log(1-D(G(z)))$？
- WGAN 相比原始 GAN 的核心改进是什么？
- DDPM 采样慢的问题有哪些解决方案 (DDIM、Latent Diffusion、Consistency Models)？

### 3. 数学推导
- VAE 的 ELBO 推导与 KL 散度闭式解。
- GAN 最优判别器 $D^*(x) = \frac{p_{data}}{p_{data} + p_g}$ 的推导。
- Diffusion 前向过程累积分布 $q(x_t|x_0)$ 的推导。

### 4. 工程实践
- 时间步嵌入 (Timestep Embedding) 的设计动机与实现。
- 噪声调度 (Noise Schedule) 的选择对 Diffusion 的影响。
- 如何从潜在空间插值生成平滑过渡的样本？

## 学习建议

1. **先理解 VAE**: VAE 是概率生成模型的入门，重参数化技巧和 KL 散度是后续许多模型 (如 VQ-VAE、Diffusion) 的基础。
2. **再攻克 GAN**: GAN 的对抗思想深刻影响了生成模型的发展，理解其训练动力学和模式崩溃对面试至关重要。
3. **最后掌握 Diffusion**: Diffusion 是当前工业界和学术界的主流方向，Stable Diffusion、Sora、DALL-E 3 均基于此。掌握其前向/反向过程、噪声预测目标和加速采样方法是面试加分项。

## 参考资料

- [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)
