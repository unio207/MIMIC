import os
import random
from PIL import Image
import numpy as np
torch = None
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    pass

def list_images(folder, exts={'.png', '.jpg', '.jpeg'}):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]

class HealthyInpaintingDataset(Dataset):
    """
    Loads healthy images and applies random irregular masks for self-supervised inpainting.
    Returns: masked_image, mask, original_image
    """
    def __init__(self, image_dir, mask_generator, transform=None, target_size=(256, 256)):
        self.image_paths = list_images(image_dir)
        self.mask_generator = mask_generator
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB').resize(self.target_size)
        img_np = np.array(img) / 255.0
        mask = self.mask_generator(self.target_size)
        masked_img = img_np * (1 - mask[..., None])
        if self.transform:
            img_np = self.transform(img_np)
            masked_img = self.transform(masked_img)
            mask = self.transform(mask)
        return masked_img, mask, img_np

# New class for GAN-A phase 1: loads paired masked and healthy images
class PairedHealthyInpaintingDataset(Dataset):
    """
    Loads healthy masked images and true healthy images for GAN-A phase 1 training.
    Returns: masked_image, original_image
    """
    def __init__(self, healthy_masked_dir, healthy_dir, transform=None):
        self.masked_files = sorted([f for f in os.listdir(healthy_masked_dir) if f.lower().endswith('.png')])
        self.healthy_files = sorted([f for f in os.listdir(healthy_dir) if f.lower().endswith('.png')])
        assert len(self.masked_files) == len(self.healthy_files), "Mismatch between masked and healthy images"
        self.masked_dir = healthy_masked_dir
        self.healthy_dir = healthy_dir
        self.transform = transform

    def __len__(self):
        return len(self.masked_files)

    def __getitem__(self, idx):
        masked_img = Image.open(os.path.join(self.masked_dir, self.masked_files[idx])).convert('RGB')
        healthy_img = Image.open(os.path.join(self.healthy_dir, self.healthy_files[idx])).convert('RGB')
        if self.transform:
            masked_img = self.transform(masked_img)
            healthy_img = self.transform(healthy_img)
        else:
            masked_img = np.array(masked_img) / 255.0
            healthy_img = np.array(healthy_img) / 255.0
        return masked_img, healthy_img
