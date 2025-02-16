import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
  def __init__(self):
    super(DiceBCELoss, self).__init__()
    self.bce = nn.BCEWithLogitsLoss()

  def forward(self, preds, targets):
    preds = torch.sigmoid(preds)  # Apply sigmoid
    smooth = 1.0
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    bce = self.bce(preds, targets)
    return 0.5 * bce + 0.5 * (1 - dice)  # Combine BCE & Dice