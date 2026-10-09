import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (pix2pix-style).

    Input:  concatenation of [model_input (4ch), image (3ch)] → 7 channels
    Output: 2-D patch map of real/fake logits (no sigmoid — use with LSGAN or BCEWithLogits).

    Uses spectral normalization on all conv layers for training stability.
    """

    def __init__(self, in_channels=7, base_features=64, n_layers=3):
        super().__init__()
        layers = []

        # First layer — no normalization
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(in_channels, base_features, kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers — spectral norm + instance norm
        nf = base_features
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1)
            ))
            layers.append(nn.InstanceNorm2d(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Penultimate layer — stride 1
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1)
        ))
        layers.append(nn.InstanceNorm2d(nf))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer — 1-channel output, no activation
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)
        ))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, 7, H, W]  (concatenated input + image)
        Returns:
            Patch map of logits [B, 1, H', W']
        """
        return self.model(x)
