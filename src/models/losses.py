import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# GAN Loss (supports LSGAN and vanilla modes)
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """GAN loss wrapper.  mode='lsgan' → MSE,  mode='vanilla' → BCEWithLogits."""

    def __init__(self, mode='lsgan'):
        super().__init__()
        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown GAN mode: {mode}")

    def forward(self, prediction, target_is_real):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


# ---------------------------------------------------------------------------
# Perceptual (VGG) Loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """L1 distance between VGG16 feature maps (relu1_2, relu2_2, relu3_3, relu4_3).
    The VGG backbone is frozen — no gradients flow through it."""

    def __init__(self, layer_weights=None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # Slice VGG into blocks ending at desired ReLU outputs
        # relu1_2 = index 4, relu2_2 = 9, relu3_3 = 16, relu4_3 = 23
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
        ])
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization constants (applied inside forward)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Per-layer weights (default: equal)
        if layer_weights is None:
            layer_weights = [1.0, 1.0, 1.0, 1.0]
        self.layer_weights = layer_weights

    def _normalize(self, x):
        """Normalize from [0,1] to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)
        loss = 0.0
        x_pred, x_tgt = pred, target
        for block, w in zip(self.blocks, self.layer_weights):
            x_pred = block(x_pred)
            x_tgt = block(x_tgt)
            loss += w * F.l1_loss(x_pred, x_tgt)
        return loss


# ---------------------------------------------------------------------------
# SSIM Loss (self-contained, no external deps)
# ---------------------------------------------------------------------------

def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)  # [1, 1, size]


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: returns 1 - SSIM (so lower is better).
    Operates per-channel then averages."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        # 1-D Gaussian → 2-D via outer product
        g1d = _fspecial_gauss_1d(window_size, sigma)          # [1, 1, W]
        window = g1d.transpose(-1, -2) @ g1d                   # [1, 1, W, W]
        self.register_buffer('window', window)

    def forward(self, pred, target):
        C = pred.shape[1]
        # Expand window to match channel count
        window = self.window.expand(C, -1, -1, -1)  # [C, 1, W, W]
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_tgt  = F.conv2d(target, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_tgt_sq  = mu_tgt ** 2
        mu_cross   = mu_pred * mu_tgt

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_pred_sq
        sigma_tgt_sq  = F.conv2d(target * target, window, padding=pad, groups=C) - mu_tgt_sq
        sigma_cross   = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_cross

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred_sq + mu_tgt_sq + C1) * (sigma_pred_sq + sigma_tgt_sq + C2))

        return 1.0 - ssim_map.mean()

    def ssim_value(self, pred, target):
        """Returns the actual SSIM value (higher is better)."""
        return 1.0 - self.forward(pred, target)
