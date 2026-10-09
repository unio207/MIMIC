import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Ensure repo root is importable when running as script.
cwd = Path.cwd().resolve()
repo_root = next((p for p in [cwd, *cwd.parents] if (p / "src").exists()), cwd)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.unet_ganb_dualhead import UNetDualHead
from src.models.patchgan_discriminator import PatchGANDiscriminator
from src.models.losses import GANLoss, PerceptualLoss, SSIMLoss


@dataclass
class Config:
    seed: int = 42
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 0
    num_epochs: int = 80
    train_split: float = 0.9

    g_lr: float = 1e-4
    d_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    lambda_adv: float = 1.0
    lambda_l1: float = 100.0
    lambda_ssim: float = 10.0
    lambda_perc: float = 10.0
    lambda_mask_bce: float = 25.0
    lambda_mask_dice: float = 25.0

    save_every: int = 5
    sample_every: int = 2
    run_name: str = "ganB_baseline_no_severity"


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASED_DIR = repo_root / "data" / "diseased" / "gls"
MASK_DIR = repo_root / "data" / "masks" / "lesion_real_expanded" / "gls"
CHECKPOINT_DIR = repo_root / "src" / "models" / "checkpoints"
LOG_DIR = repo_root / "results" / "logs"
SAMPLE_DIR = repo_root / "results" / "figures" / "training_samples" / "ganB_baseline"

for p in [CHECKPOINT_DIR, LOG_DIR, SAMPLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DirectGanBDataset(Dataset):
    """GAN-B direct baseline dataset.

    For each diseased image and real lesion mask pair, create pseudo-healthy input by
    inpainting lesion pixels from the diseased image using the lesion mask.

    Returns:
        input_pseudohealthy: [3, H, W]
        target_diseased: [3, H, W]
        target_mask: [1, H, W] in {0, 1}
        filename: original diseased filename
    """

    def __init__(self, pairs, image_size: int):
        self.pairs = pairs
        self.rgb_tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_tfm = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        diseased_path, mask_path, fname = self.pairs[idx]

        diseased_pil = Image.open(diseased_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")

        diseased_np = np.array(diseased_pil)
        mask_np = np.array(mask_pil)
        _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        # Telea inpainting gives a fast pseudo-healthy proxy for this baseline.
        pseudohealthy_np = cv2.inpaint(diseased_np, mask_bin, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        input_t = self.rgb_tfm(Image.fromarray(pseudohealthy_np))
        target_img_t = self.rgb_tfm(diseased_pil)
        target_mask_t = self.mask_tfm(Image.fromarray(mask_bin)).clamp(0, 1)

        return input_t, target_img_t, target_mask_t, fname


def parse_leaf_index_from_image(name: str):
    stem = Path(name).stem
    if stem.startswith("leaf"):
        token = stem[4:]
        if token.isdigit():
            return int(token)
    return None


def parse_leaf_index_from_mask(name: str):
    stem = Path(name).stem
    if stem.startswith("mask_leaf"):
        token = stem[len("mask_leaf"):]
        if token.isdigit():
            return int(token)
    return None


def build_pairs_and_audit(diseased_dir: Path, mask_dir: Path):
    diseased_files = sorted([f for f in os.listdir(diseased_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    diseased_by_idx = {}
    for f in diseased_files:
        idx = parse_leaf_index_from_image(f)
        if idx is not None:
            diseased_by_idx[idx] = f

    masks_by_idx = {}
    for f in mask_files:
        idx = parse_leaf_index_from_mask(f)
        if idx is not None:
            masks_by_idx[idx] = f

    common = sorted(set(diseased_by_idx).intersection(masks_by_idx))
    only_diseased = sorted(set(diseased_by_idx) - set(masks_by_idx))
    only_masks = sorted(set(masks_by_idx) - set(diseased_by_idx))

    pairs = [
        (diseased_dir / diseased_by_idx[i], mask_dir / masks_by_idx[i], diseased_by_idx[i])
        for i in common
    ]

    audit = {
        "n_diseased_files": len(diseased_files),
        "n_mask_files": len(mask_files),
        "n_parsed_diseased": len(diseased_by_idx),
        "n_parsed_masks": len(masks_by_idx),
        "n_pairs": len(pairs),
        "only_diseased_indices": only_diseased[:20],
        "only_mask_indices": only_masks[:20],
    }
    return pairs, audit


def split_pairs(pairs, train_split: float, seed: int):
    indices = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(len(indices) * train_split)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    return train_pairs, val_pairs


def dice_loss_from_logits(mask_logits, target_mask, eps=1e-6):
    pred = torch.sigmoid(mask_logits)
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target_mask.view(target_mask.shape[0], -1)

    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def save_sample_panel(epoch, batch, input_t, target_img_t, target_mask_t, gen_img_t, gen_mask_logits_t):
    in_np = input_t[0].detach().cpu().permute(1, 2, 0).numpy()
    tgt_np = target_img_t[0].detach().cpu().permute(1, 2, 0).numpy()
    g_np = gen_img_t[0].detach().cpu().permute(1, 2, 0).numpy()

    tgt_mask = target_mask_t[0, 0].detach().cpu().numpy()
    pred_mask = torch.sigmoid(gen_mask_logits_t[0, 0]).detach().cpu().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(24, 5))
    axs[0].imshow(in_np)
    axs[0].set_title("Input (Pseudo-Healthy)")
    axs[0].axis("off")

    axs[1].imshow(g_np)
    axs[1].set_title("Generated Diseased")
    axs[1].axis("off")

    axs[2].imshow(tgt_np)
    axs[2].set_title("Target Diseased")
    axs[2].axis("off")

    axs[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axs[3].set_title("Predicted Mask")
    axs[3].axis("off")

    axs[4].imshow(tgt_mask, cmap="gray", vmin=0, vmax=1)
    axs[4].set_title("Target Mask")
    axs[4].axis("off")

    plt.tight_layout()
    out_path = SAMPLE_DIR / f"{CFG.run_name}_ep{epoch:03d}_b{batch:04d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(model_g, loader, criterion_l1, criterion_ssim, bce_logits):
    model_g.eval()

    total_l1 = 0.0
    total_ssim = 0.0
    total_mask_bce = 0.0
    total_mask_dice = 0.0
    n = 0

    with torch.no_grad():
        for inp, target_img, target_mask, _ in loader:
            inp = inp.to(DEVICE, non_blocking=True)
            target_img = target_img.to(DEVICE, non_blocking=True)
            target_mask = target_mask.to(DEVICE, non_blocking=True)

            out_img, out_mask_logits = model_g(inp)

            total_l1 += criterion_l1(out_img, target_img).item()
            total_ssim += criterion_ssim.ssim_value(out_img, target_img).item()
            total_mask_bce += bce_logits(out_mask_logits, target_mask).item()
            total_mask_dice += dice_loss_from_logits(out_mask_logits, target_mask).item()
            n += 1

    if n == 0:
        return {"l1": 0.0, "ssim": 0.0, "mask_bce": 0.0, "mask_dice": 0.0}

    return {
        "l1": total_l1 / n,
        "ssim": total_ssim / n,
        "mask_bce": total_mask_bce / n,
        "mask_dice": total_mask_dice / n,
    }


def main():
    set_seed(CFG.seed)
    print(f"Device: {DEVICE}")
    print(f"Repo root: {repo_root}")

    pairs, audit = build_pairs_and_audit(DISEASED_DIR, MASK_DIR)
    print("Dataset audit:")
    print(json.dumps(audit, indent=2))

    if len(pairs) < 8:
        raise RuntimeError("Not enough paired data to train GAN-B baseline.")

    audit_path = LOG_DIR / f"{CFG.run_name}_dataset_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    train_pairs, val_pairs = split_pairs(pairs, train_split=CFG.train_split, seed=CFG.seed)
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    train_ds = DirectGanBDataset(train_pairs, image_size=CFG.image_size)
    val_ds = DirectGanBDataset(val_pairs, image_size=CFG.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    net_g = UNetDualHead(in_channels=3, image_out_channels=3, mask_out_channels=1).to(DEVICE)
    net_d = PatchGANDiscriminator(in_channels=6, base_features=64, n_layers=3).to(DEVICE)

    criterion_gan = GANLoss(mode="lsgan").to(DEVICE)
    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_ssim = SSIMLoss().to(DEVICE)
    criterion_perc = PerceptualLoss().to(DEVICE)
    criterion_bce_logits = nn.BCEWithLogitsLoss().to(DEVICE)

    opt_g = optim.Adam(net_g.parameters(), lr=CFG.g_lr, betas=(CFG.beta1, CFG.beta2))
    opt_d = optim.Adam(net_d.parameters(), lr=CFG.d_lr, betas=(CFG.beta1, CFG.beta2))

    history = []

    for epoch in range(1, CFG.num_epochs + 1):
        net_g.train()
        net_d.train()

        running = {
            "d": 0.0,
            "g": 0.0,
            "g_adv": 0.0,
            "g_l1": 0.0,
            "g_ssim": 0.0,
            "g_perc": 0.0,
            "g_mask_bce": 0.0,
            "g_mask_dice": 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.num_epochs}")
        for step, (inp, target_img, target_mask, _) in enumerate(pbar, start=1):
            inp = inp.to(DEVICE, non_blocking=True)
            target_img = target_img.to(DEVICE, non_blocking=True)
            target_mask = target_mask.to(DEVICE, non_blocking=True)

            # -------------------------
            # Train D
            # -------------------------
            with torch.no_grad():
                fake_img_detach, _ = net_g(inp)

            real_pair = torch.cat([inp, target_img], dim=1)
            fake_pair = torch.cat([inp, fake_img_detach], dim=1)

            pred_real = net_d(real_pair)
            pred_fake = net_d(fake_pair)

            loss_d_real = criterion_gan(pred_real, True)
            loss_d_fake = criterion_gan(pred_fake, False)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # -------------------------
            # Train G
            # -------------------------
            out_img, out_mask_logits = net_g(inp)

            pred_fake_for_g = net_d(torch.cat([inp, out_img], dim=1))
            loss_g_adv = criterion_gan(pred_fake_for_g, True)
            loss_g_l1 = criterion_l1(out_img, target_img)
            loss_g_ssim = criterion_ssim(out_img, target_img)
            loss_g_perc = criterion_perc(out_img, target_img)
            loss_g_mask_bce = criterion_bce_logits(out_mask_logits, target_mask)
            loss_g_mask_dice = dice_loss_from_logits(out_mask_logits, target_mask)

            loss_g = (
                CFG.lambda_adv * loss_g_adv
                + CFG.lambda_l1 * loss_g_l1
                + CFG.lambda_ssim * loss_g_ssim
                + CFG.lambda_perc * loss_g_perc
                + CFG.lambda_mask_bce * loss_g_mask_bce
                + CFG.lambda_mask_dice * loss_g_mask_dice
            )

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            running["d"] += loss_d.item()
            running["g"] += loss_g.item()
            running["g_adv"] += loss_g_adv.item()
            running["g_l1"] += loss_g_l1.item()
            running["g_ssim"] += loss_g_ssim.item()
            running["g_perc"] += loss_g_perc.item()
            running["g_mask_bce"] += loss_g_mask_bce.item()
            running["g_mask_dice"] += loss_g_mask_dice.item()

            pbar.set_postfix(
                d=f"{loss_d.item():.3f}",
                g=f"{loss_g.item():.3f}",
                l1=f"{loss_g_l1.item():.3f}",
                m_dice=f"{loss_g_mask_dice.item():.3f}",
            )

            if epoch % CFG.sample_every == 0 and step == 1:
                save_sample_panel(epoch, step, inp, target_img, target_mask, out_img, out_mask_logits)

        n_steps = max(len(train_loader), 1)
        train_log = {f"train_{k}": v / n_steps for k, v in running.items()}

        val_log = evaluate(net_g, val_loader, criterion_l1, criterion_ssim, criterion_bce_logits)
        val_log = {f"val_{k}": v for k, v in val_log.items()}

        epoch_log = {"epoch": epoch, **train_log, **val_log}
        history.append(epoch_log)

        print(
            f"[Epoch {epoch}] "
            f"D={epoch_log['train_d']:.4f} G={epoch_log['train_g']:.4f} "
            f"Val L1={epoch_log['val_l1']:.4f} Val SSIM={epoch_log['val_ssim']:.4f} "
            f"Val DiceLoss={epoch_log['val_mask_dice']:.4f}"
        )

        if epoch % CFG.save_every == 0 or epoch == CFG.num_epochs:
            ckpt_path = CHECKPOINT_DIR / f"{CFG.run_name}_epoch{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "config": CFG.__dict__,
                    "netG_state_dict": net_g.state_dict(),
                    "netD_state_dict": net_d.state_dict(),
                    "optG_state_dict": opt_g.state_dict(),
                    "optD_state_dict": opt_d.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    history_json = LOG_DIR / f"{CFG.run_name}_history.json"
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history json: {history_json}")

    history_csv = LOG_DIR / f"{CFG.run_name}_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    print(f"Saved history csv: {history_csv}")


if __name__ == "__main__":
    main()
