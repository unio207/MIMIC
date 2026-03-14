
# MIMIC: A Multi-Stage Inpainting Model Training Pipeline for Biologically Constrained Diseased Leaf Image Creation to Enhance Agricultural Disease Detection

## Abstract

Agricultural crop disease management relies heavily on accurate disease detection to mitigate yield loss and guide farmers’ decisions. However, deep learning models used for leaf disease detection are constrained by small, imbalanced, and insufficiently annotated datasets. Although image generation models have been used for data augmentation, existing approaches are either unsupervised, failing to preserve biological realism, or rely on high-level class labels, lacking lesion-level paired annotations.

This study proposes MIMIC: a multi-stage inpainting model pipeline for diseased leaf image creation that produces biologically constrained synthetic data to improve detection model accuracy and reliability, supporting earlier intervention and reducing yield loss. Unlike conventional methods that attempt to generate diseased leaves in one step, the proposed method separates the process into two stages. Each stage is made up of a generative adversarial network (GAN) operating sequentially: the first model (GAN-A) removes diseased lesions from real diseased leaves through inpainting to create pseudo-healthy pairs, and the second model (GAN-B) generates synthetic disease lesions from real healthy leaves by training on pseudo-healthy pairs created by the first. By decoupling lesion removal and synthesis, real leaves can be introduced in each training stage, allowing MIMIC to maintain anatomical structure. 

The output images are used in conjunction with real leaves to train and compare a severity estimation model against one trained on real data alone. The model trained on MIMIC-augmented data showed improvements in accuracy and generalizability on unseen field images, suggesting that MIMIC can strengthen existing precision disease management technology, even across other diseases.


## Goal

Improve segmentation and severity estimation of gray leaf spot (GLS) in corn leaves under limited labeled data by generating **realistic synthetic diseased leaves with pixel-level lesion masks** via a two-stage GAN pipeline.

Key challenges addressed:
- Scarcity of real GLS-infected images with lesion annotations
- Class imbalance (few early/mid severity cases)
- Need for paired (image, mask) training data for downstream segmentation

## Design Philosophy

Disease generation uses a **closed two-stage GAN loop** that separates lesion removal from lesion synthesis:

- **GAN-A (diseased → healthy inpainting)**: Removes real lesions and reconstructs plausible healthy tissue, producing pseudo-healthy versions of real diseased leaves.
- **GAN-B (healthy → diseased + mask)**: Adds realistic lesions back onto healthy leaves and simultaneously predicts lesion masks, trained on paired data produced by GAN-A.

This approach:
- Preserves real leaf anatomy — never hallucinating global structure from scratch
- Produces **paired synthetic data (image + mask)** from purely healthy inputs at inference time
- Avoids the overfitting/hallucination problems of training a single unconstrained image GAN on limited diseased data

## Pipeline Overview

1. Collect real diseased images and extract/annotate lesion masks
2. **GAN-A** inpaints lesion regions → produces pseudo-healthy versions of real diseased leaves
3. Triplets (pseudo-healthy, diseased, lesion mask) become training data for **GAN-B**
4. **GAN-B** learns to synthesize lesions on healthy leaves and predict their masks
5. At synthesis time: feed new healthy leaves into GAN-B → synthetic diseased images **and** masks
6. Train segmentation/severity models on synthetic data, evaluate on held-out real data

Synthetic data is never mixed across train/test boundaries.

---

## Current Progress

### ✅ GAN-A Phase 1 — Complete (trained & converged)

Phase 1 trains GAN-A as a self-supervised inpainting model on healthy leaf images using synthetic lesion masks — no diseased data needed.

**Architecture:**
- **Generator**: UNet encoder-decoder with skip connections. 4-channel input (RGB + binary lesion mask) → 3-channel RGB output. Feature progression [64, 128, 256, 512]. Sigmoid output activation.
- **Discriminator**: 70×70 PatchGAN with spectral normalization and instance normalization.

**Loss stack:**
- L1 reconstruction (dominant weight, λ=100)
- SSIM structural similarity (λ=10)
- VGG-based perceptual loss — relu1_2, relu2_2, relu3_3, relu4_3 (λ=10)
- LSGAN adversarial loss (λ=1)

**Training:**
- Mixed precision (AMP), A100 GPU (Lambda Labs cloud)
- Checkpoint saving and resume support
- Trained to 150 epochs; converged with high SSIM and low L1
- Saved checkpoints: epoch 50, 100, 150 (final)

**Data pipeline:**
- ~380 healthy leaf images with synthetic GLS lesion masks composited on top
- 90/10 train/test split; dataset class loads (masked_image, binary_mask, healthy_target) triplets
- Training data organized under `data/gan_a/phase_1/` with `train/` and `test/` splits each containing `image/`, `mask/`, `binary_mask/`, and `outline/` subdirectories

---

### ✅ Lesion Mask Infrastructure — Complete

**Real lesion masks (`data/masks/lesion_real/`):**
- HSV color thresholding pipeline with a per-image threshold database (`hsv_thresholds.txt`) manually tuned across all ~260 GLS images
- Post-processing: morphological close (5×5, iter=2) + dilation (5×5, iter=1) clamped to leaf silhouette — biased toward over-coverage so GAN-A fully covers each lesion
- Expanded versions saved to `lesion_real_expanded/`

**Synthetic lesion masks (`data/masks/lesion_synthetic/`):**
- Real GLS lesion shapes extracted from ~260 diseased images and composited onto healthy leaves
- PCA-based leaf orientation alignment, random scaling/translation, leaf boundary clipping
- Preserves authentic lesion geometry and RGB tissue colors
- Binary masks saved as both viewable PNGs (`lesion_synthetic_binary/`) and strict 0/1 `.npy` arrays (`lesion_synthetic_binary01_npy/`)
- Same morphological refinement (close + dilate, clamped to leaf) applied before saving

---

### ✅ Models and Losses — Complete

- UNet generator (`src/models/unet_inpaint.py`)
- PatchGAN discriminator (`src/models/patchgan_discriminator.py`)
- Loss module: GANLoss (LSGAN/vanilla), PerceptualLoss (VGG16), SSIMLoss (`src/models/losses.py`)

---

### 🔄 Image Segmentation Baseline — In Progress

- UNet trained on real GLS data for lesion segmentation (`src/image_seg/unet_image_seg_real_data.ipynb`)
- Serves as the real-data-only baseline to compare against MIMIC-augmented training
- Outputs binary lesion predictions; evaluation via IoU, Dice, SSIM, PSNR

---

### 🔄 Experiment Structure — In Progress

Three experiment tracks set up under `experiments/`:
- `baselines/` — real-data-only models
- `synthetic_only/` — models trained purely on MIMIC-generated data (algorithmic synthetic mask generation explored here)
- `mixed_real_synthetic/` — real + synthetic combined

---

## Remaining Work

### GAN-A Phase 2 — Fine-tuning on real diseased data
- Fine-tune the Phase 1 checkpoint on real GLS images: diseased input + real lesion mask → pseudo-healthy output
- Adversarial loss against real healthy patches
- Produces the pseudo-healthy images needed as GAN-B training inputs
- Training notebook: `src/gan/ganA_train_with_lesion_masklinux.ipynb`

### GAN-B — Healthy → Diseased + Mask
- Generator takes a healthy leaf and produces a realistic diseased RGB image + binary lesion mask (dual-head output)
- Trained on (pseudo-healthy, diseased, lesion_mask) triplets produced by GAN-A Phase 2
- Optional severity conditioning to control lesion area/density
- At inference: unlimited synthetic paired (diseased image, lesion mask) data from any healthy leaf

### Evaluation Pipeline
- Segmentation IoU / Dice on held-out real GLS test images using models trained on synthetic data
- Severity estimation MAE (predicted vs true lesion area ratio)
- Image quality metrics (FID)
- Baselines: no augmentation, copy-paste synthesis, real-data-only UNet

### Experiments
- Synthetic-only vs mixed real+synthetic vs real-only baselines
- Ablation on synthetic dataset size and severity distribution
- Failure mode analysis, especially at low severity

---

## Project Structure

```
configs/                    — Experiment configuration files
data/
  healthy/                  — ~380 healthy corn leaf images
  diseased/gls/             — ~260 real GLS-infected images
  masks/
    leaf/                   — Leaf boundary masks
    lesion_real/            — Real lesion masks (HSV thresholding, per-image tuned)
    lesion_real_expanded/   — Morphologically expanded real lesion masks
    lesion_synthetic/       — Synthetic RGBA lesion masks (real lesion overlay)
    lesion_synthetic_binary/         — Viewable binary PNGs (0/255)
    lesion_synthetic_binary01_npy/   — Strict 0/1 .npy arrays for training
  healthy_masked/           — Healthy images with synthetic lesions composited
  gan_a/phase_1/            — GAN-A Phase 1 training data
    train/{image,mask,binary_mask,outline}/
    test/{image,mask,binary_mask,outline}/
src/
  data/                     — Dataset classes, synthetic mask generator
  gan/                      — GAN training notebooks and scripts
  image_seg/                — UNet segmentation (real-data baseline)
  models/                   — UNet, PatchGAN, loss functions
    checkpoints/            — Saved model weights (epoch 50, 100, 150)
  lesion_mask/              — HSV mask extraction and refinement tools
  geometry/                 — Lesion geometry utilities
experiments/
  baselines/                — Real-data-only baselines
  synthetic_only/           — Synthetic-only training experiments
  mixed_real_synthetic/     — Mixed augmentation experiments
notebooks/                  — Exploration and sanity checks
results/
  figures/                  — Training samples, pair previews
  logs/                     — Training loss histories (JSON)
```

## Research Question

Does biologically constrained synthetic lesion generation improve segmentation and severity estimation under limited labeled data, compared to training on real data alone?

## Data Sources

1. Wu, D., Barbedo, J. G. A., Stewart, E. L., Zhang, J., & Nelson, R. J. (2020). A curated dataset of maize leaf images for disease identification. Mendeley Data, V1. https://data.mendeley.com/datasets/vy629dngm8/1

2. Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. PlantVillage Dataset. https://plantvillage.psu.edu/

3. Wang, G., Sun, Y., & Wang, J. (2017). CD&S Dataset: Handheld Imagery Dataset Acquired Under Field Conditions for Corn Disease Identification and Severity Estimation. https://www.kaggle.com/datasets/gpiosenka/corn-disease-and-severity

---

## Full Project Context (Copy-Paste for External Use)

> Use the section below to give another model or collaborator complete context on this project — what it is, what has been built, what is left, and how everything fits together.

### What this project is

This is an ISEF 2026 research project investigating whether synthetic data generation can improve disease segmentation and severity estimation for gray leaf spot (GLS) in corn, under limited real labeled data.

### Core approach: Two-stage GAN pipeline

The pipeline has two learned components:

**GAN-A (diseased to healthy inpainting):** Takes a real diseased leaf image and its binary lesion mask, inpaints the lesion regions to produce a pseudo-healthy version of that leaf. This creates paired (pseudo-healthy, diseased, mask) triplets without ever needing a real healthy/diseased pair of the exact same leaf.

**GAN-B (healthy to diseased + mask):** Takes a healthy leaf image and generates a realistic diseased version plus a corresponding binary lesion mask. Trained on the triplets produced by GAN-A. At inference time, GAN-B converts any healthy leaf into a synthetic diseased image with an exact pixel-level mask — unlimited paired training data for downstream models.

### Why two stages instead of one

A single GAN trained on limited diseased data tends to hallucinate unrealistic leaf anatomy. GAN-A anchors the pipeline in real leaf structure by learning inpainting on actual leaves. GAN-B then only needs to learn lesion placement and appearance, a much more constrained task.

### What has been built

**GAN-A Phase 1 is fully trained and converged** (150 epochs, A100 GPU). The generator is a UNet (4-channel input: RGB + binary lesion mask, 3-channel RGB output, features [64, 128, 256, 512]). The discriminator is a 70×70 PatchGAN with spectral and instance normalization. Loss stack: L1 (λ=100), SSIM (λ=10), VGG perceptual (λ=10), LSGAN adversarial (λ=1). Trained with mixed precision on ~380 healthy leaves with synthetic GLS masks. Three checkpoints saved (epoch 50, 100, 150).

**Lesion mask infrastructure is complete.** Real lesion masks extracted from all ~260 GLS images via a per-image HSV threshold database (manually tuned). Synthetic lesion masks generated by overlaying real lesion shapes onto healthy leaves (PCA orientation, random transforms, leaf boundary clipping). Both mask types go through the same morphological post-processing: close (5×5, iter=2) + dilate (5×5, iter=1) clamped to the leaf silhouette — biased toward over-coverage so GAN-A always inpaints the full lesion.

**Image segmentation baseline exists.** A UNet trained on real GLS data provides the real-data-only benchmark for lesion segmentation.

**Experiment structure is set up.** Three tracks: `baselines/`, `synthetic_only/`, `mixed_real_synthetic/`.

**Model code is modular.** UNet generator, PatchGAN discriminator, and all loss functions are in separate reusable files under `src/models/`.

### What remains

1. **GAN-A Phase 2**: Fine-tune the Phase 1 checkpoint on real GLS diseased images — (diseased RGB + real lesion mask) → pseudo-healthy output, with adversarial loss against real healthy patches. Produces the pseudo-healthy leaves needed as GAN-B inputs.

2. **GAN-B**: Dual-head generator (healthy leaf → diseased RGB + binary lesion mask), trained on (pseudo-healthy, diseased, mask) triplets from GAN-A Phase 2. Optional severity conditioning. Architecture builds on same UNet/PatchGAN foundation.

3. **Evaluation**: Segmentation IoU/Dice and severity estimation MAE on held-out real GLS test images, using models trained on MIMIC-augmented data vs real-data-only baseline. FID for image quality.

4. **Experiments**: Synthetic-only vs mixed vs real-only training comparisons. Dataset size and severity distribution ablations. Failure mode analysis.

### Data available

- ~380 healthy corn leaf images with leaf boundary masks
- ~260 real GLS-infected images with per-image HSV-tuned lesion masks (close+dilate refined)
- Synthetic RGBA lesion masks for all healthy images (binary PNGs + .npy arrays)
- GAN-A training data fully organized (`data/gan_a/phase_1/train+test` with image/mask/binary_mask/outline splits)
- GAN-A Phase 1 trained checkpoints (epoch 50, 100, 150)

### Key files

- `src/models/unet_inpaint.py` — UNet generator
- `src/models/patchgan_discriminator.py` — PatchGAN discriminator
- `src/models/losses.py` — All loss functions (GAN, perceptual, SSIM)
- `src/data/healthy_inpainting_dataset.py` — Dataset class
- `src/data/synthetic_mask_generator.py` — Real lesion overlay system
- `src/gan/ganA_train.ipynb` — Main GAN-A training notebook
- `src/gan/ganA_data_setup.ipynb` — Data preparation and augmentation
- `src/lesion_mask/hsv_thresholds_loader.py` — HSV mask extraction utility

### Research question

Does biologically constrained synthetic lesion generation improve segmentation and severity estimation under limited labeled data?
