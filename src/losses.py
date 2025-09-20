import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        alpha (Tensor, optional): Class weights (1D tensor of shape [num_classes]).
        gamma (float): Focusing parameter (default=2.0).
        reduction (str): "mean", "sum", or "none".
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) - raw logits
        targets: (batch_size,) - ground truth labels
        """
        # Cross entropy loss (per sample, no reduction)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")

        # p_t: probability of the true class
        pt = torch.exp(-ce_loss)

        # Focal Loss formula
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
