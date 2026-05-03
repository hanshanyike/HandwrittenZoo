"""
Variational Autoencoder (VAE) - 变分自编码器

算法简介:
    VAE 是一种深度生成模型，通过编码器将数据映射到潜在空间的概率分布，
    再从该分布中采样并通过解码器重构数据，同时利用 KL 散度约束潜在空间结构。

核心思想:
    1. 编码器输出潜在变量的均值 mu 和对数方差 log_var，而非确定性向量。
    2. 重参数化技巧 (Reparameterization Trick): z = mu + sigma * epsilon，
       将随机性从计算图中分离，保证梯度可反向传播。
    3. 损失函数 = 重构损失 (MSE/BCE) + KL 散度正则项，使潜在空间平滑且连续。

时间复杂度: O(B * D * H)，其中 B 为 batch size，D 为输入维度，H 为隐藏层维度。
空间复杂度: O(B * D + B * latent_dim)。

面试频率: 高 (生成模型、概率图模型必考点)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    编码器: 将输入数据 x 映射到潜在空间的参数 (mu, log_var)。
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        # 第一层全连接，将输入映射到隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 第二层全连接，进一步提取特征
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出潜在分布的均值 mu
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # 输出潜在分布的对数方差 log_var，数值稳定性更好
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 使用 ReLU 激活引入非线性
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class Decoder(nn.Module):
    """
    解码器: 从潜在变量 z 重构原始数据 x。
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        # 将潜在向量映射到隐藏层
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        # 第二层全连接
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出重构数据，维度与输入一致
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        # 输出层使用 sigmoid，将像素值压缩到 [0, 1] 区间 (适用于归一化图像)
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon


class VAE(nn.Module):
    """
    变分自编码器整体封装: Encoder + Reparameterization + Decoder。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 400, latent_dim: int = 20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧:
        原始采样 z ~ N(mu, sigma^2) 不可导，无法反向传播。
        将其改写为 z = mu + sigma * epsilon，其中 epsilon ~ N(0, I)。
        这样随机性只来自与参数无关的 epsilon，mu 和 sigma 的路径是确定性的，梯度可以正常回传。
        """
        std = torch.exp(0.5 * log_var)  # 由对数方差计算标准差: sigma = exp(0.5 * log(sigma^2))
        epsilon = torch.randn_like(std)  # 从标准正态分布采样，形状与 std 相同
        z = mu + std * epsilon
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播:
        1. 编码得到 mu 和 log_var。
        2. 重参数化采样得到 z。
        3. 解码得到重构数据 x_recon。
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def loss_function(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        VAE 损失函数:
        1. 重构损失: 衡量输入与输出的差异，使用二元交叉熵 (BCE) 或 MSE。
        2. KL 散度: D_KL(N(mu, sigma^2) || N(0, I))，约束潜在分布接近标准正态分布。
           闭式解: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))。
        """
        # 重构损失: 将 x 展平后与 x_recon 计算 BCE
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
        # KL 散度，对 batch 内每个样本求和
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_divergence

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        从先验分布 N(0, I) 中采样潜在向量，并通过解码器生成新数据。
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples


if __name__ == "__main__":
    # 自测试块: 验证 VAE 前向传播、损失计算和采样功能
    batch_size = 4
    input_dim = 784  # 例如 28x28 的 MNIST 图像展平后
    hidden_dim = 400
    latent_dim = 20

    # 构造随机输入数据，模拟归一化后的图像
    x = torch.rand(batch_size, input_dim)

    model = VAE(input_dim, hidden_dim, latent_dim)
    x_recon, mu, log_var = model(x)

    print("输入形状:", x.shape)
    print("重构形状:", x_recon.shape)
    print("mu 形状:", mu.shape)
    print("log_var 形状:", log_var.shape)

    loss = model.loss_function(x_recon, x, mu, log_var)
    print("VAE 损失值:", loss.item())

    # 测试采样功能
    samples = model.sample(2, torch.device("cpu"))
    print("采样输出形状:", samples.shape)

    # 验证梯度回传
    loss.backward()
    print("梯度回传成功，训练流程验证通过。")
