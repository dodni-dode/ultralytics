import torch
import torch.nn as nn
import torchvision.models as models

__all__ = ['ResNetBlock']

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_bottleneck=True):
        super(ResNetBlock, self).__init__()
        if use_bottleneck:
            # Bottleneck architecture (used in ResNet50)
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4),
                nn.ReLU(inplace=True),
            )
        else:
            # Standard ResNet block
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)
