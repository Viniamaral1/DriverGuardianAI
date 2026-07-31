import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


class FocalLossLabelSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing.
    """

    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):

        num_classes = inputs.size(1)

        with torch.no_grad():

            true_dist = torch.zeros_like(inputs)

            true_dist.fill_(
                self.smoothing / (num_classes - 1)
            )

            true_dist.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.smoothing
            )

        log_probs = F.log_softmax(inputs, dim=1)

        probs = torch.exp(log_probs)

        focal = (1 - probs) ** self.gamma

        loss = -true_dist * focal * log_probs

        if self.alpha is not None:
            loss *= self.alpha.unsqueeze(0)

        return loss.sum(dim=1).mean()


def get_loss(loss_name, class_weights=None):
    """
    Factory function for selecting the loss function.
    """

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == "focal":
        return FocalLoss(alpha=class_weights)

    elif loss_name == "focal_smoothing":
        return FocalLossLabelSmoothing(alpha=class_weights)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")