from tqdm import tqdm
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.unet_inpaint import UNet
from src.data.healthy_inpainting_dataset import PairedHealthyInpaintingDataset



# --- Config ---
healthy_masked_dir = repo_root / 'data' / 'healthy_masked'
healthy_dir = repo_root / 'data' / 'healthy_target'
batch_size = 8
num_epochs = 50
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.transforms as T


def main():
    # --- Dataset & Loader ---
    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = PairedHealthyInpaintingDataset(healthy_masked_dir, healthy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --- Model ---
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for masked_img, healthy_img in progress:
            masked_img = masked_img.to(device)
            healthy_img = healthy_img.to(device)
            out = model(masked_img)
            loss = criterion(out, healthy_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * masked_img.size(0)
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | L1 Loss: {avg_loss:.4f}")
        # Optionally: save model checkpoint, run validation, visualize samples


if __name__ == "__main__":
    main()
