"""
CNN 模块单元测试
测试 ResNet、VGG
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from cnn.resnet.resnet import BasicBlock, Bottleneck, ResNet
from cnn.vgg.vgg import VGGBlock, VGG


class TestBasicBlock:
    """测试 ResNet BasicBlock"""

    def test_output_shape(self):
        in_channels = 64
        out_channels = 64
        block = BasicBlock(in_channels, out_channels)
        x = torch.randn(2, in_channels, 32, 32)
        out = block(x)
        assert out.shape == (2, out_channels, 32, 32)

    def test_downsample(self):
        in_channels = 64
        out_channels = 128
        block = BasicBlock(in_channels, out_channels, stride=2)
        x = torch.randn(2, in_channels, 32, 32)
        out = block(x)
        assert out.shape == (2, out_channels, 16, 16)


class TestBottleneck:
    """测试 ResNet Bottleneck"""

    def test_output_shape(self):
        in_channels = 64
        out_channels = 256
        block = Bottleneck(in_channels, out_channels)
        x = torch.randn(2, in_channels, 32, 32)
        out = block(x)
        assert out.shape == (2, out_channels, 32, 32)

    def test_downsample(self):
        in_channels = 64
        out_channels = 256
        block = Bottleneck(in_channels, out_channels, stride=2)
        x = torch.randn(2, in_channels, 32, 32)
        out = block(x)
        assert out.shape == (2, out_channels, 16, 16)


class TestResNet:
    """测试 ResNet 模型"""

    def test_resnet18(self):
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    def test_resnet50(self):
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1000)

    def test_feature_extraction(self):
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        features = model.features(x)
        assert features.shape[0] == 2
        assert features.dim() == 4


class TestVGGBlock:
    """测试 VGG Block"""

    def test_output_shape(self):
        in_channels = 3
        out_channels = 64
        block = VGGBlock(in_channels, out_channels)
        x = torch.randn(2, in_channels, 32, 32)
        out = block(x)
        assert out.shape == (2, out_channels, 32, 32)

    def test_pooling(self):
        in_channels = 64
        out_channels = 128
        block = VGGBlock(in_channels, out_channels, pool=True)
        x = torch.randn(2, in_channels, 32, 32)
        out = block(x)
        assert out.shape == (2, out_channels, 16, 16)


class TestVGG:
    """测试 VGG 模型"""

    def test_vgg16(self):
        model = VGG([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1000)

    def test_vgg11(self):
        model = VGG([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1000)

    def test_feature_extractor(self):
        model = VGG([64, 64, 'M', 128, 128, 'M'])
        x = torch.randn(2, 3, 224, 224)
        features = model.features(x)
        assert features.dim() == 4
        assert features.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
