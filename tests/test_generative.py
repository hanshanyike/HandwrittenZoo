"""
生成模型模块单元测试
测试 VAE、GAN、Diffusion
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from generative_model.vae.vae import VAE
from generative_model.gan.gan import Generator, Discriminator, GAN
from generative_model.diffusion.diffusion import DiffusionModel, UNet


class TestVAE:
    """测试变分自编码器 (VAE)"""

    def test_encode_shape(self):
        input_dim = 784
        hidden_dim = 400
        latent_dim = 20
        vae = VAE(input_dim, hidden_dim, latent_dim)
        x = torch.randn(4, input_dim)
        mu, logvar = vae.encode(x)
        assert mu.shape == (4, latent_dim)
        assert logvar.shape == (4, latent_dim)

    def test_reparameterize_shape(self):
        input_dim = 784
        hidden_dim = 400
        latent_dim = 20
        vae = VAE(input_dim, hidden_dim, latent_dim)
        mu = torch.randn(4, latent_dim)
        logvar = torch.randn(4, latent_dim)
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (4, latent_dim)

    def test_decode_shape(self):
        input_dim = 784
        hidden_dim = 400
        latent_dim = 20
        vae = VAE(input_dim, hidden_dim, latent_dim)
        z = torch.randn(4, latent_dim)
        out = vae.decode(z)
        assert out.shape == (4, input_dim)

    def test_forward_shape(self):
        input_dim = 784
        hidden_dim = 400
        latent_dim = 20
        vae = VAE(input_dim, hidden_dim, latent_dim)
        x = torch.randn(4, input_dim)
        recon, mu, logvar = vae(x)
        assert recon.shape == (4, input_dim)
        assert mu.shape == (4, latent_dim)
        assert logvar.shape == (4, latent_dim)

    def test_kl_loss_finite(self):
        input_dim = 784
        hidden_dim = 400
        latent_dim = 20
        vae = VAE(input_dim, hidden_dim, latent_dim)
        x = torch.randn(4, input_dim)
        recon, mu, logvar = vae(x)
        loss = vae.loss_function(x, recon, mu, logvar)
        assert torch.isfinite(loss)


class TestGAN:
    """测试生成对抗网络 (GAN)"""

    def test_generator_output_shape(self):
        latent_dim = 100
        img_dim = 784
        gen = Generator(latent_dim, img_dim)
        z = torch.randn(4, latent_dim)
        out = gen(z)
        assert out.shape == (4, img_dim)

    def test_generator_output_range(self):
        latent_dim = 100
        img_dim = 784
        gen = Generator(latent_dim, img_dim)
        z = torch.randn(4, latent_dim)
        out = gen(z)
        # tanh 输出范围 [-1, 1]
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_discriminator_output_shape(self):
        img_dim = 784
        disc = Discriminator(img_dim)
        x = torch.randn(4, img_dim)
        out = disc(x)
        assert out.shape == (4, 1)

    def test_discriminator_probability_range(self):
        img_dim = 784
        disc = Discriminator(img_dim)
        x = torch.randn(4, img_dim)
        out = disc(x)
        # sigmoid 输出范围 [0, 1]
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_gan_forward(self):
        latent_dim = 100
        img_dim = 784
        gan = GAN(latent_dim, img_dim)
        z = torch.randn(4, latent_dim)
        fake_img = gan(z)
        assert fake_img.shape == (4, img_dim)


class TestDiffusion:
    """测试扩散模型 (Diffusion)"""

    def test_unet_output_shape(self):
        in_channels = 3
        out_channels = 3
        time_emb_dim = 256
        unet = UNet(in_channels, out_channels, time_emb_dim)
        x = torch.randn(2, in_channels, 32, 32)
        t = torch.randn(2, time_emb_dim)
        out = unet(x, t)
        assert out.shape == (2, out_channels, 32, 32)

    def test_forward_diffusion(self):
        img_size = 32
        in_channels = 3
        model = DiffusionModel(img_size, in_channels)
        x = torch.randn(2, in_channels, img_size, img_size)
        t = torch.randint(0, model.timesteps, (2,))
        noisy, noise = model.forward_diffusion(x, t)
        assert noisy.shape == x.shape
        assert noise.shape == x.shape

    def test_reverse_diffusion_shape(self):
        img_size = 32
        in_channels = 3
        model = DiffusionModel(img_size, in_channels)
        shape = (2, in_channels, img_size, img_size)
        samples = model.reverse_diffusion(shape)
        assert samples.shape == shape

    def test_sample_shape(self):
        img_size = 32
        in_channels = 3
        model = DiffusionModel(img_size, in_channels)
        samples = model.sample(4)
        assert samples.shape == (4, in_channels, img_size, img_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
