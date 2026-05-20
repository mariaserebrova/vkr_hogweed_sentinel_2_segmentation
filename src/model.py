import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class TverskyLossFromLogits(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky.mean()


def build_model(
    *,
    encoder_name: str,
    encoder_weights,
    in_channels: int,
    num_classes: int,
    device: str,
):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return model.to(device)


def build_losses(
    *,
    pos_weight: float,
    alpha: float,
    beta: float,
    device: str,
):
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    tversky_loss = TverskyLossFromLogits(alpha=alpha, beta=beta)
    return bce_loss, tversky_loss
