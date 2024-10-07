import torch
import torch.nn as nn
import torch.nn.functional as F


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # Apply sigmoid activation to get probabilities
        inputs = F.sigmoid(inputs)

        # Compute BCE loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Compute DICE loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # total loss
        Dice_BCE = BCE + dice_loss

        return Dice_BCE