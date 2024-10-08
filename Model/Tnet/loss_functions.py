import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class segmentation.

        Args:
            alpha (Tensor, list, or None): Weighting factor for each class. Shape: (C,)
            gamma (float): Focusing parameter.
            reduction (str): Reduction method ('none', 'mean', 'sum').
        """
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (Tensor): Predicted logits, shape (N, C, H, W).
            targets (Tensor): Ground truth class indices, shape (N, H, W).

        Returns:
            Tensor: Computed focal loss.
        """
        N, C, H, W = inputs.shape
        # Apply log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)  # Shape: (N, C, H, W)
        probs = torch.exp(log_probs)  # Shape: (N, C, H, W)

        # Gather the probabilities corresponding to the target class
        #targets = targets.long()
        targets = targets.view(N, 1, H, W)
        probs_true = probs.gather(1, targets).squeeze(1)  # Shape: (N, H, W)
        log_probs_true = log_probs.gather(1, targets).squeeze(1)  # Shape: (N, H, W)

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha.gather(0, targets.squeeze(1))  # Shape: (N, H, W)
        else:
            alpha = 1.0

        loss = -alpha * ((1 - probs_true) ** self.gamma) * log_probs_true  # Shape: (N, H, W)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Shape: (N, H, W)

class GIoULoss(nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        """
        Generalized IoU Loss for multi-class segmentation.

        Args:
            num_classes (int): Number of classes.
            reduction (str): Reduction method ('none', 'mean', 'sum').
        """
        super(GIoULoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (Tensor): Predicted probabilities after softmax, shape (N, C, H, W).
            targets (Tensor): Ground truth one-hot encoded, shape (N, C, H, W).

        Returns:
            Tensor: Computed GIoU loss.
        """
        # Convert targets to one-hot encoding
        if targets.dim() == 3:
            targets = F.one_hot(targets, num_classes=self.num_classes)  # Shape: (N, H, W, C)
            targets = targets.permute(0, 3, 1, 2).float()  # Shape: (N, C, H, W)

        # Compute IoU per class
        intersection = (inputs * targets).sum(dim=(2, 3))  # Shape: (N, C)
        union = (inputs + targets - inputs * targets).sum(dim=(2, 3))  # Shape: (N, C)
        iou = intersection / (union + 1e-7)  # Shape: (N, C)

        # GIoU is similar to IoU but accounts for the smallest enclosing box
        # For segmentation maps, defining enclosing boxes is not straightforward.
        # Instead, we can use IoU-based loss directly.

        # Define loss as 1 - IoU
        loss = 1 - iou  # Shape: (N, C)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Shape: (N, C)


class GDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, reduction='mean'):
        """
        Dice Loss for multi-class segmentation.

        Args:
            num_classes (int): Number of classes.
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('none', 'mean', 'sum').
        """
        super(GDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (Tensor): Predicted probabilities after softmax, shape (N, C, H, W).
            targets (Tensor): Ground truth one-hot encoded, shape (N, C, H, W).

        Returns:
            Tensor: Computed Dice loss.
        """
        # Convert targets to one-hot encoding if necessary
        if targets.dim() == 3:
            targets = F.one_hot(targets, num_classes=self.num_classes)  # Shape: (N, H, W, C)
            targets = targets.permute(0, 3, 1, 2).float()  # Shape: (N, C, H, W)

        # Flatten tensors
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # Shape: (N, C, H*W)
        targets = targets.view(targets.size(0), targets.size(1), -1)  # Shape: (N, C, H*W)

        intersection = (inputs * targets).sum(dim=2)  # Shape: (N, C)
        union = inputs.sum(dim=2) + targets.sum(dim=2)  # Shape: (N, C)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # Shape: (N, C)
        loss = 1 - dice  # Shape: (N, C)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Shape: (N, C)


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

class CompositeLoss(nn.Module):
    def __init__(self, l1, l2, weight=.5):
        super(CompositeLoss, self).__init__()
        self.l1_loss = l1
        self.l2_loss = l2
        self.weight = weight

    def forward(self,outputs,targets):
        l1_loss = self.l1_loss(outputs,targets)
        l2_loss = self.l2_loss(outputs,targets)
        return l1_loss*self.weight+l2_loss*(1-self.weight)