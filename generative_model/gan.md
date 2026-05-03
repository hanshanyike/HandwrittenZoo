# Generative Adversarial Network (GAN) - 生成对抗网络

## 算法简介

生成对抗网络 (GAN) 由 Ian Goodfellow 等人于 2014 年提出，是深度学习领域最具影响力的生成模型之一。GAN 由生成器 (Generator) 和判别器 (Discriminator) 两个神经网络组成，通过对抗训练的方式学习数据分布，能够生成高度逼真的图像、音频和文本等样本。

## 核心思想

GAN 的设计灵感来源于博弈论中的零和博弈:

1. **生成器 (Generator, G)**: 接收随机噪声 $z \sim p_z(z)$ 作为输入，学习映射函数 $G(z; \theta_g)$，目标是生成与真实数据分布 $p_{data}(x)$ 难以区分的样本。
2. **判别器 (Discriminator, D)**: 接收真实样本 $x$ 或生成样本 $G(z)$，输出 $D(x; \theta_d) \in [0, 1]$ 表示其为真实样本的概率，目标是尽可能准确地区分真假。
3. **对抗训练**: G 和 D 形成 minimax 博弈，G 试图最大化 D 犯错的可能性，D 试图最小化分类错误。当达到纳什均衡时，G 生成的样本分布与真实分布一致，D 无法区分真假 (输出恒为 0.5)。

## 数学公式

### 1. Minimax 目标函数
GAN 的原始优化目标为:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- D 最大化该值 (正确识别真假)。
- G 最小化该值 (使 $D(G(z))$ 接近 1，即 $\log(1 - D(G(z)))$ 最小)。

### 2. 判别器损失
对于判别器，固定 G，最大化:
$$\mathcal{L}_D = -\left[ \mathbb{E}_{x \sim p_{data}} \log D(x) + \mathbb{E}_{z \sim p_z} \log(1 - D(G(z))) \right]$$
等价于最小化上述负值，即标准的二分类交叉熵损失。

### 3. 生成器损失
对于生成器，固定 D，最小化:
$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z} \log D(G(z))$$
实践中也常使用 $\mathcal{L}_G = \mathbb{E}_{z \sim p_z} \log(1 - D(G(z)))$，但前者梯度更稳定 (避免早期梯度消失)。

### 4. 最优判别器
给定固定的 G，最优判别器为:
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$
其中 $p_g(x)$ 为生成样本的分布。

### 5. 全局最优
当 $p_g = p_{data}$ 时，$D^*(x) = 0.5$，达到纳什均衡，此时 C = $-\log 4$ 为全局最小值。

## 时间/空间复杂度

- **时间复杂度**: $O(B \cdot D \cdot H)$，每轮迭代需分别前向/反向传播 G 和 D 两个网络。
- **空间复杂度**: $O(|\theta_G| + |\theta_D|)$，需同时存储两套网络参数及各自的优化器状态。
- **与 VAE 对比**: GAN 生成样本更锐利逼真，但训练不稳定；VAE 训练稳定且有显式似然，但样本较模糊。
- **与 Diffusion 对比**: GAN 采样速度快 (单步前向)，但训练难度高；Diffusion 训练稳定、生成质量高，但采样需多步迭代。

## 面试高频考点

### 1. GAN 训练不稳定的原因是什么？
**答案**: 
- **梯度消失**: 早期生成器质量差，判别器很容易区分真假，导致 $D(G(z)) \approx 0$，生成器梯度 $\nabla \log(1-D(G(z)))$ 接近 0，G 无法更新。
- **模式崩溃 (Mode Collapse)**: 生成器发现某类样本能欺骗判别器后，持续生成相似样本，丧失多样性。
- **纳什均衡难达**: 高维非凸优化中，G 和 D 的参数更新互相干扰，难以收敛到理论均衡点。

### 2. 什么是模式崩溃 (Mode Collapse)？如何解决？
**答案**:
模式崩溃指生成器只覆盖真实数据分布的部分模式 (modes)，输出单一或高度相似的样本。
解决方法:
- **WGAN**: 使用 Wasserstein 距离替代 JS 散度，提供更平滑的梯度。
- **WGAN-GP**: 引入梯度惩罚 (Gradient Penalty) 强制 Lipschitz 约束。
- ** minibatch Discrimination**: 让判别器同时判断一批样本，鼓励多样性。
- **TTUR (Two Time-Scale Update Rule)**: 为 G 和 D 设置不同学习率。
- **Unrolled GAN**: 更新 G 时考虑 D 的多步响应。

### 3. 为什么生成器损失常用 $-\log D(G(z))$ 而非 $\log(1-D(G(z)))$？
**答案**:
早期训练时 D 很容易识别假样本，$D(G(z)) \approx 0$。若使用 $\log(1-D(G(z)))$，其梯度为 $\frac{1}{1-D(G(z))} \cdot \nabla D(G(z))$，当 $D(G(z)) \to 0$ 时梯度也趋于 0，导致 G 无法学习。而 $-\log D(G(z))$ 的梯度为 $\frac{-1}{D(G(z))} \cdot \nabla D(G(z))$，在 $D(G(z)) \to 0$ 时梯度很大，能提供更强烈的更新信号。

### 4. WGAN 相比原始 GAN 的核心改进是什么？
**答案**:
- **损失函数**: 用 Earth Mover's Distance (Wasserstein-1) 替代 Jensen-Shannon 散度，即使两个分布无重叠也能提供有意义梯度。
- **判别器变 Critic**: 去掉输出层 Sigmoid，直接输出标量分数，并施加 Lipschitz 约束 (权重裁剪或梯度惩罚)。
- **训练稳定性**: 解决了原始 GAN 的梯度消失和训练震荡问题，loss 值与生成质量正相关。

### 5. GAN 中的纳什均衡是什么意思？
**答案**:
在 GAN 框架下，纳什均衡指 G 和 D 都达到最优且无法通过单方面改变策略获得更好结果的状态。此时:
- 生成分布 $p_g$ 等于真实分布 $p_{data}$。
- 最优判别器 $D^*(x) = 0.5$，即完全无法区分真假。
- 理论上目标函数达到全局最小值 $-\log 4$。

## 代码解析

### Generator 生成器
```python
self.fc1 = nn.Linear(latent_dim, hidden_dim)
self.activation = nn.LeakyReLU(0.2)
x_gen = torch.tanh(self.fc3(h))
```
使用 LeakyReLU 避免神经元死亡，输出层 Tanh 将值域限制在 [-1, 1]，适合已归一化到该范围的数据 (如 MNIST 预处理)。

### Discriminator 判别器
```python
self.dropout = nn.Dropout(0.3)
validity = torch.sigmoid(self.fc3(h))
```
Dropout 防止判别器过强导致梯度消失，Sigmoid 将输出压缩为概率值。

### 对抗损失计算
```python
fake_data = self.generator(z).detach()
```
训练判别器时，使用 `.detach()` 阻断生成器的梯度流，确保只更新判别器参数。这是 GAN 交替训练的关键实现细节。

### 生成器损失
```python
fake_labels = torch.ones(batch_size, 1).to(device)
g_loss = self.adversarial_loss(d_pred, fake_labels)
```
生成器试图欺骗判别器，因此标签设为 1 (真实)。使用 BCELoss 衡量 $D(G(z))$ 与 1 的差距。

## 参考资料

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [DCGAN: Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)
- [WGAN-GP: Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
