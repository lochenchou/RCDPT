import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()
    
    def forward(self, pred_depth, image):
        # Normalize the depth with mean
        depth_mean = pred_depth.mean(2, True).mean(3, True)
        pred_depth_normalized = pred_depth / (depth_mean + 1e-7)

        # Compute the gradient of depth
        grad_depth_x = torch.abs(pred_depth_normalized[:, :, :, :-1] - pred_depth_normalized[:, :, :, 1:])
        grad_depth_y = torch.abs(pred_depth_normalized[:, :, :-1, :] - pred_depth_normalized[:, :, 1:, :])

        # Compute the gradient of the image
        grad_image_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        grad_image_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_image_x)
        grad_depth_y *= torch.exp(-grad_image_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
