"""
Denoising Diffusion Probabilistic Model (DDPM) - 去噪扩散概率模型 (简化版)

算法简介:
    DDPM 是一类基于马尔可夫链的生成模型，通过两个过程实现数据生成:
    1. 前向过程 (Forward Process): 逐步向数据添加高斯噪声，直至变为纯噪声。
    2. 反向过程 (Reverse Process): 训练神经网络预测噪声，逐步去噪恢复原始数据。
    本实现提供简化但完整的 DDPM 核心逻辑，适用于教学与面试准备。

核心思想:
    1. 前向加噪是预设的马尔可夫链，无需学习参数，利用重参数化可在任意时刻 t 直接采样 x_t。
    2. 反向去噪用神经网络 epsilon_theta(x_t, t) 预测每一步添加的噪声，通过 MSE 损失训练。
    3. 采样时从纯噪声 x_T ~ N(0, I) 出发，迭代 T 步去噪得到生成样本 x_0。

时间复杂度:
    训练: O(T * B * D * H)，T 为时间步数，B 为 batch size，D 为数据维度，H 为网络隐藏维度。
    采样: O(T * B * D * H)，需迭代 T 步。
空间复杂度: O(B * D + T)，主要存储数据和时间步相关参数。

面试频率: 极高 (当前生成模型最热门方向，Stable Diffusion 的基础)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    正弦时间步嵌入 (Sinusoidal Timestep Embedding):
    将离散的时间步 t 映射到连续的 d 维向量，使网络感知当前去噪阶段。
    使用不同频率的正弦和余弦函数，类似于 Transformer 的位置编码。
    """
    half = dim // 2
    # 计算频率因子: 1 / (10000^(2i/d))
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    # 计算角度: t * freq
    args = timesteps[:, None].float() * freqs[None]
    # 拼接正弦和余弦分量
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # 若维度为奇数，补零对齐
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SimpleUNet(nn.Module):
    """
    简化版 U-Net 去噪网络:
    接收加噪图像 x_t 和时间嵌入 t_emb，预测当前步骤添加的噪声 epsilon。
    实际应用中通常使用带跳跃连接的卷积 U-Net，此处用全连接版本便于理解核心逻辑。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, time_emb_dim: int = 128):
        super(SimpleUNet, self).__init__()
        # 时间步嵌入投影层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 编码路径
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 解码路径，引入时间信息
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 生成时间步嵌入
        t_emb = timestep_embedding(t, 128)
        t_feat = self.time_mlp(t_emb)

        # 编码路径
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))

        # 将时间特征与隐藏特征拼接，注入时间信息
        h_t = torch.cat([h2, t_feat], dim=-1)
        h3 = self.activation(self.fc3(h_t))
        # 输出预测的噪声，与输入同维度
        noise_pred = self.fc4(h3)
        return noise_pred


class Diffusion(nn.Module):
    """
    简化版 DDPM 扩散模型:
    封装前向加噪、反向去噪训练和采样生成流程。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super(Diffusion, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.model = SimpleUNet(input_dim, hidden_dim)

        # 注册前向过程的噪声调度参数 (beta_t)，不参与梯度更新
        # beta_t 从 beta_start 线性增加到 beta_end
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        # alpha_bar_t = prod_{i=1}^t alpha_i，表示累积保留的信号比例
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        # 反向过程方差，使用 beta_t 简化
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程 (加噪):
        利用重参数化技巧，直接从 x_0 和 t 计算 x_t，无需迭代:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        # 根据时间步 t 取对应的累积信号比例
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        # 重参数化: 直接合成第 t 步的加噪数据
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        训练损失:
        随机采样时间步 t，生成加噪数据 x_t，让网络预测添加的噪声 epsilon。
        损失为预测噪声与真实噪声之间的均方误差 (MSE)。
        """
        batch_size = x_0.size(0)
        device = x_0.device
        # 随机采样时间步 t，范围 [0, num_timesteps-1]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        # 生成真实噪声
        noise = torch.randn_like(x_0)
        # 前向加噪
        x_t, _ = self.forward_diffusion(x_0, t, noise)
        # 网络预测噪声
        noise_pred = self.model(x_t, t)
        # MSE 损失
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        反向采样过程 (去噪):
        从纯噪声 x_T ~ N(0, I) 开始，迭代 T 步去噪生成样本。
        每一步: x_{t-1} = (x_t - beta_t / sqrt(1-alpha_bar_t) * epsilon_pred) / sqrt(alpha_t) + sigma_t * z
        """
        # 从标准正态分布采样初始噪声
        x_t = torch.randn(batch_size, self.input_dim).to(device)

        # 从 T-1 到 0 逐步去噪
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            # 预测当前噪声
            noise_pred = self.model(x_t, t)

            # 计算去噪均值
            sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None]
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None]
            beta = self.betas[t][:, None]

            # 均值预测公式
            model_mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha_cumprod * noise_pred)

            if t_idx == 0:
                # 最后一步不加额外噪声
                x_t = model_mean
            else:
                posterior_variance = self.posterior_variance[t][:, None]
                noise = torch.randn_like(x_t)
                x_t = model_mean + torch.sqrt(posterior_variance) * noise

        return x_t


if __name__ == "__main__":
    # 自测试块: 验证扩散模型的前向加噪、损失计算和反向采样
    batch_size = 4
    input_dim = 784  # 模拟 MNIST 展平图像
    num_timesteps = 100  # 测试时使用较小步数以加速

    # 构造随机输入数据，模拟归一化图像
    x_0 = torch.rand(batch_size, input_dim)

    model = Diffusion(input_dim, hidden_dim=256, num_timesteps=num_timesteps)

    # 测试前向加噪
    t = torch.randint(0, num_timesteps, (batch_size,))
    x_t, noise = model.forward_diffusion(x_0, t)
    print("原始数据形状:", x_0.shape)
    print("加噪后数据形状:", x_t.shape)
    print("噪声形状:", noise.shape)

    # 测试损失计算
    loss = model.compute_loss(x_0)
    print("训练损失:", loss.item())

    # 测试采样 (使用较小步数)
    samples = model.sample(2, torch.device("cpu"))
    print("采样输出形状:", samples.shape)

    # 验证梯度回传
    loss.backward()
    print("梯度回传成功，Diffusion 模型基础结构验证通过。")
