import matplotlib.pyplot as plt
import numpy as np
import torch

def show_inpainting_sample(masked_img, mask, output, gt_img, idx=0):
    """
    Visualize a single inpainting result (batch idx=0 by default)
    """
    masked_img = masked_img[idx].permute(1,2,0).cpu().numpy()
    mask = mask[idx,0].cpu().numpy()
    output = output[idx].permute(1,2,0).detach().cpu().numpy()
    gt_img = gt_img[idx].permute(1,2,0).cpu().numpy()
    fig, axs = plt.subplots(1,4,figsize=(12,4))
    axs[0].imshow(masked_img)
    axs[0].set_title('Masked Input')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[2].imshow(output)
    axs[2].set_title('Output')
    axs[3].imshow(gt_img)
    axs[3].set_title('Ground Truth')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
