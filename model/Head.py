import numpy as np
import torch
import torch.nn as nn

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x

class HeadDepth(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=1, stride=1, padding=0),
            # nn.ReLU()
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.head(x)
        return x

class HeadSeg(nn.Module):
    def __init__(self, features, nclasses=2):
        super(HeadSeg, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        x = self.head(x)
        return x