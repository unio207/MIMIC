import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Down path
        chs = in_channels
        for feat in features:
            self.downs.append(UNetBlock(chs, feat))
            chs = feat
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        chs = features[-1] * 2

        # Up path (symmetric with down path)
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(chs, feat, 2, stride=2))
            self.ups.append(UNetBlock(feat * 2, feat))
            chs = feat

        self.final = nn.Conv2d(features[0], out_channels, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, 4, H, W] (RGB + outline mask)
        Returns:
            Output tensor of shape [batch, 3, H, W] (inpainted RGB)
        """
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)
        return torch.sigmoid(self.final(x))
