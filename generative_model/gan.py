"""
Generative Adversarial Network (GAN) - 生成对抗网络

算法简介:
    GAN 由 Goodfellow 于 2014 年提出，包含生成器 (Generator) 和判别器 (Discriminator) 两个网络。
    生成器学习从噪声分布映射到真实数据分布，判别器学习区分真实样本与生成样本。
    两者通过零和博弈交替训练，最终达到纳什均衡。

核心思想:
    1. 生成器 G: 输入随机噪声 z，输出生成样本 G(z)，目标是欺骗判别器。
    2. 判别器 D: 输入真实样本 x 或生成样本 G(z)，输出其为真实样本的概率 D(x)。
    3. 对抗训练: D 最大化区分真实与伪造的能力，G 最小化 D 的识别能力，形成 minimax 博弈。

时间复杂度: O(B * D * H)，训练涉及两个网络的前向与反向传播。
空间复杂度: O(B * D + B * latent_dim)，需存储两套网络参数。

面试频率: 高 (生成模型经典必考，模式崩溃与训练不稳定是重点)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    生成器: 将低维噪声向量 z 映射到与真实数据同维度的生成样本。
    使用多层全连接网络配合 ReLU 激活，输出层使用 Tanh 或 Sigmoid。
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Generator, self).__init__()
        # 将噪声向量映射到隐藏层
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        # 中间层进一步扩展特征表达能力
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层，维度与真实数据一致
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # LeakyReLU 可避免神经元死亡，在生成器中常用
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        # 输出层使用 tanh，将值域限制在 [-1, 1]，适合归一化到该范围的数据
        x_gen = torch.tanh(self.fc3(h))
        return x_gen


class Discriminator(nn.Module):
    """
    判别器: 判断输入样本是真实数据还是生成器伪造的数据。
    输出单个标量，经 Sigmoid 压缩为概率值。
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层只有一个神经元，表示为真实样本的概率
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.LeakyReLU(0.2)
        # Dropout 可防止判别器过强，有助于训练稳定性
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.activation(self.fc1(x)))
        h = self.dropout(self.activation(self.fc2(h)))
        # Sigmoid 将输出压缩到 [0, 1]，表示真实概率
        validity = torch.sigmoid(self.fc3(h))
        return validity


class GAN(nn.Module):
    """
    GAN 整体封装: 包含生成器 G 和判别器 D，提供训练用的损失计算接口。
    """

    def __init__(self, input_dim: int, latent_dim: int = 100, hidden_dim: int = 256):
        super(GAN, self).__init__()
        self.generator = Generator(latent_dim, hidden_dim, input_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)
        self.latent_dim = latent_dim
        # 二元交叉熵损失，用于判别器的分类任务
        self.adversarial_loss = nn.BCELoss()

    def generator_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成器损失:
        采样噪声 z，生成样本 G(z)，并试图让判别器将其判断为真实样本 (label=1)。
        损失为 D(G(z)) 与全 1 标签的 BCE。
        """
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_data = self.generator(z)
        # 生成器希望判别器对伪造样本输出接近 1
        fake_labels = torch.ones(batch_size, 1).to(device)
        d_pred = self.discriminator(fake_data)
        g_loss = self.adversarial_loss(d_pred, fake_labels)
        return g_loss

    def discriminator_loss(
        self, real_data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        判别器损失:
        对真实样本 x 希望输出 1，对生成样本 G(z) 希望输出 0。
        损失为两部分 BCE 之和。
        """
        batch_size = real_data.size(0)
        # 真实样本标签为 1
        real_labels = torch.ones(batch_size, 1).to(device)
        real_pred = self.discriminator(real_data)
        real_loss = self.adversarial_loss(real_pred, real_labels)

        # 生成伪造样本，标签为 0
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_data = self.generator(z).detach()  # 阻断生成器梯度，只更新判别器
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_pred = self.discriminator(fake_data)
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)

        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        从噪声先验中采样并生成新数据。
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.generator(z)
        self.generator.train()
        return samples


if __name__ == "__main__":
    # 自测试块: 验证 GAN 的生成器、判别器前向传播和损失计算
    batch_size = 4
    input_dim = 784  # 模拟 MNIST 展平图像
    latent_dim = 100
    hidden_dim = 256
    device = torch.device("cpu")

    model = GAN(input_dim, latent_dim, hidden_dim).to(device)

    # 模拟真实数据 (已归一化到 [-1, 1])
    real_data = torch.rand(batch_size, input_dim) * 2 - 1

    # 测试判别器损失
    d_loss = model.discriminator_loss(real_data, device)
    print("判别器损失:", d_loss.item())

    # 测试生成器损失
    g_loss = model.generator_loss(batch_size, device)
    print("生成器损失:", g_loss.item())

    # 测试采样
    samples = model.sample(2, device)
    print("采样输出形状:", samples.shape)

    # 验证梯度回传
    d_loss.backward()
    g_loss.backward()
    print("梯度回传成功，GAN 基础结构验证通过。")
