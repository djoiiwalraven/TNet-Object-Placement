import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


# Composite loss function
def composite_loss(outputs, targets, bce_weight=0.5, dice_weight=0.5):
    # BCEWithLogitsLoss expects raw logits
    bce_loss = nn.BCEWithLogitsLoss()(outputs, targets)
    dice_loss = DiceLoss()(outputs, targets)
    total_loss = bce_weight * bce_loss + dice_weight * dice_loss
    return total_loss

