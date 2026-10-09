import numpy as np
import cv2
import random
import os
import glob

# =============================================================================
# GLS (Gray Leaf Spot) Synthetic Mask Generator — Real Lesion Overlay Approach
# =============================================================================
#
# APPROACH:
#   Instead of generating lesion shapes algorithmically, this module takes
#   real binary lesion masks extracted from actual GLS-infected leaves and
#   overlays them onto healthy leaf images. This preserves the authentic
#   shape, texture, and spatial distribution of real GLS lesions.
#
# HOW IT WORKS:
#
#   1. LOAD REAL LESION MASKS:
#      We load all binary masks from data/masks/lesion_real/gls/. Each mask
#      was extracted from a real GLS-infected leaf image using color
#      thresholding + morphological cleanup.
#
#   2. ESTIMATE SOURCE & TARGET LEAF ORIENTATION (PCA):
#      For both the source (diseased) leaf and the target (healthy) leaf,
#      we estimate the principal axis using PCA on pixel coordinates.
#      Given a binary mask, we collect all (x, y) where mask > 0, compute:
#
#          C = [[ var(x),  cov(x,y) ],
#               [ cov(x,y), var(y) ]]
#
#      The eigenvector of the largest eigenvalue gives the leaf's long axis.
#      The angle is: theta = atan2(eigvec[1], eigvec[0])
#
#   3. ROTATION ALIGNMENT:
#      The rotation needed to align the source lesion mask to the target
#      leaf is: delta_angle = target_angle - source_angle (in degrees).
#      We rotate the source mask by delta_angle around its center using
#      cv2.getRotationMatrix2D + cv2.warpAffine.
#
#   4. SCALING:
#      After rotation, the lesion mask is resized to match the target leaf
#      dimensions. An optional random scale factor (e.g., 0.8–1.2) adds
#      variation so the same source mask produces different-looking results
#      on different healthy leaves.
#
#   5. RANDOM TRANSLATION:
#      The rotated+scaled mask is shifted by a random offset so the lesions
#      don't always land in the same spot. The shift is constrained so the
#      mask stays mostly within the target leaf area.
#
#   6. LEAF CLIPPING:
#      The final overlay is AND-ed with the target leaf mask so no lesions
#      appear outside the leaf boundary.
#
#   7. RANDOMIZATION:
#      Each call randomly selects a source mask, optionally flips it
#      horizontally or vertically, applies a random scale, and shifts
#      placement — so even the same healthy leaf gets different synthetic
#      lesion patterns each time.
#
# =============================================================================

# Default path to real GLS lesion masks
_DEFAULT_REAL_MASK_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'masks', 'lesion_real', 'gls'
)
# Default path to source diseased images (for orientation estimation)
_DEFAULT_DISEASED_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'diseased', 'gls'
)

# Cache loaded masks so we don't re-read from disk every call
_cached_masks = None
_cached_source_angles = None


def _load_real_masks(mask_dir=None, diseased_dir=None):
    """Load all real GLS lesion masks and their corresponding diseased images.

    Returns a list of (mask_filename, grayscale_mask, diseased_rgb_or_None).
    The diseased image is loaded in RGB (converted from OpenCV BGR) so that
    actual lesion colors can be sampled later.
    """
    global _cached_masks
    if _cached_masks is not None:
        return _cached_masks

    if mask_dir is None:
        mask_dir = _DEFAULT_REAL_MASK_DIR
    if diseased_dir is None:
        diseased_dir = _DEFAULT_DISEASED_DIR

    mask_dir = os.path.normpath(mask_dir)
    diseased_dir = os.path.normpath(diseased_dir)
    paths = sorted(glob.glob(os.path.join(mask_dir, 'mask_*.png')))
    masks = []
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            basename = os.path.basename(p)
            # Derive the diseased image name: "mask_leaf201.png" -> "leaf201.png"
            leaf_name = basename.replace('mask_', '')
            diseased_path = os.path.join(diseased_dir, leaf_name)
            if os.path.exists(diseased_path):
                bgr = cv2.imread(diseased_path, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
            else:
                rgb = None
            masks.append((basename, m, rgb))
    _cached_masks = masks
    return masks


def _get_source_angles(diseased_dir=None):
    """Estimate and cache the leaf orientation for each source diseased image."""
    global _cached_source_angles
    if _cached_source_angles is not None:
        return _cached_source_angles

    if diseased_dir is None:
        diseased_dir = _DEFAULT_DISEASED_DIR

    diseased_dir = os.path.normpath(diseased_dir)
    angles = {}
    paths = sorted(glob.glob(os.path.join(diseased_dir, 'leaf*.png')))
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, leaf = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
            angles[os.path.basename(p)] = estimate_leaf_angle(leaf)
    _cached_source_angles = angles
    return angles


def estimate_leaf_angle(leaf_mask):
    """
    Estimate the orientation angle of a leaf from its binary mask using PCA.

    Collects all nonzero pixel coordinates, computes the 2x2 covariance
    matrix, and returns the angle (radians) of the principal axis.

    Returns:
        angle_rad (float): Leaf orientation in radians (CCW from +x axis).
    """
    ys, xs = np.nonzero(leaf_mask)
    if len(xs) < 10:
        return 0.0
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle_rad = np.arctan2(principal[1], principal[0])
    return angle_rad


def gls_synthetic_mask(
    size,
    leaf_mask=None,
    leaf_angle_rad=None,
    mask_dir=None,
    diseased_dir=None,
    scale_range=(0.8, 1.2),
    shift_range=(-50, 50),
    flip_random=True,
):
    """
    Generate a synthetic GLS lesion mask by overlaying a real lesion mask
    (from a diseased leaf) onto a healthy leaf, aligned by orientation.

    Args:
        size: (H, W) of the output mask.
        leaf_mask: Binary mask (H, W) of the target healthy leaf. If provided,
                   the result is clipped to the leaf area and orientation is
                   auto-detected.
        leaf_angle_rad: Override target leaf angle. If None, estimated via PCA.
        mask_dir: Path to folder of real lesion masks. Defaults to
                  data/masks/lesion_real/gls/.
        diseased_dir: Path to folder of source diseased images for orientation.
                      Defaults to data/diseased/gls/.
        scale_range: (min, max) random scale factor applied to the source mask.
        shift_range: (min, max) random pixel shift in x and y for placement.
        flip_random: If True, randomly flip the source mask horizontally
                     and/or vertically for variation.

    Returns:
        mask: np.float32 array (H, W, 4) RGBA image. Lesion pixels use
              the actual RGB colors from the corresponding diseased leaf
              image (same geometric transform applied), alpha=1.
              Non-lesion pixels are (0, 0, 0, 0).
    """
    h, w = size
    output = np.zeros((h, w, 4), dtype=np.float32)

    # --- Load real masks (with diseased images) ---
    real_masks = _load_real_masks(mask_dir, diseased_dir)
    if len(real_masks) == 0:
        return output

    # --- Pick a random source mask ---
    src_name, src_mask, src_diseased_rgb = random.choice(real_masks)

    # --- Determine source leaf angle ---
    src_leaf_name = src_name.replace('mask_', '')
    source_angles = _get_source_angles(diseased_dir)
    src_angle = source_angles.get(src_leaf_name, 0.0)

    # --- Determine target leaf angle ---
    if leaf_angle_rad is None:
        if leaf_mask is not None:
            leaf_angle_rad = estimate_leaf_angle(leaf_mask)
        else:
            leaf_angle_rad = 0.0

    # --- Compute rotation delta (degrees) ---
    delta_angle_deg = np.degrees(leaf_angle_rad - src_angle)

    # --- Random scale ---
    scale = random.uniform(scale_range[0], scale_range[1])

    # --- Prepare diseased color image (same size as mask) ---
    # If the diseased image exists, resize it to match the source mask so
    # both undergo identical geometric transforms.
    if src_diseased_rgb is not None:
        src_colors = cv2.resize(
            src_diseased_rgb, (src_mask.shape[1], src_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )  # (sh, sw, 3) uint8 RGB
    else:
        src_colors = None

    # --- Random flips (applied identically to mask AND color image) ---
    warped = src_mask.copy()
    warped_colors = src_colors.copy() if src_colors is not None else None
    if flip_random:
        if random.random() > 0.5:
            warped = cv2.flip(warped, 0)  # vertical flip
            if warped_colors is not None:
                warped_colors = cv2.flip(warped_colors, 0)
        if random.random() > 0.5:
            warped = cv2.flip(warped, 1)  # horizontal flip
            if warped_colors is not None:
                warped_colors = cv2.flip(warped_colors, 1)

    # --- Rotate around center (same matrix for mask and colors) ---
    sh, sw = warped.shape[:2]
    center = (sw / 2.0, sh / 2.0)
    M = cv2.getRotationMatrix2D(center, -delta_angle_deg, scale)

    # Compute new bounding size after rotation so nothing is clipped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(sh * sin_a + sw * cos_a)
    new_h = int(sh * cos_a + sw * sin_a)
    M[0, 2] += (new_w - sw) / 2.0
    M[1, 2] += (new_h - sh) / 2.0

    rotated = cv2.warpAffine(warped, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
    if warped_colors is not None:
        rotated_colors = cv2.warpAffine(
            warped_colors, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
        )

    # --- Resize to target dimensions ---
    resized = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_NEAREST)
    if warped_colors is not None:
        resized_colors = cv2.resize(rotated_colors, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- Random shift (same offset for mask and colors) ---
    dx = random.randint(shift_range[0], shift_range[1])
    dy = random.randint(shift_range[0], shift_range[1])
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(resized, T, (w, h), flags=cv2.INTER_NEAREST)
    if warped_colors is not None:
        shifted_colors = cv2.warpAffine(resized_colors, T, (w, h), flags=cv2.INTER_LINEAR)

    # --- Binarize ---
    binary = (shifted > 127).astype(np.float32)

    # --- Clip to leaf mask ---
    if leaf_mask is not None:
        binary = binary * (leaf_mask > 0).astype(np.float32)

    # --- Colorize lesion pixels with actual diseased-image colors ---
    lesion_ys, lesion_xs = np.nonzero(binary)
    n_lesion = len(lesion_ys)

    output = np.zeros((h, w, 4), dtype=np.float32)  # RGBA output
    if n_lesion > 0:
        if warped_colors is not None:
            # Sample real RGB values from the transformed diseased image
            real_rgb = shifted_colors[lesion_ys, lesion_xs]  # (N, 3) uint8
            output[lesion_ys, lesion_xs, 0] = real_rgb[:, 0] / 255.0  # R
            output[lesion_ys, lesion_xs, 1] = real_rgb[:, 1] / 255.0  # G
            output[lesion_ys, lesion_xs, 2] = real_rgb[:, 2] / 255.0  # B
        else:
            # Fallback: random GLS-like colors if diseased image not found
            color_low = np.array([166, 131, 83], dtype=np.float32)
            color_high = np.array([235, 221, 201], dtype=np.float32)
            t = np.random.uniform(0.0, 1.0, size=(n_lesion, 1)).astype(np.float32)
            base_colors = color_low + t * (color_high - color_low)
            noise = np.random.uniform(-10, 10, size=(n_lesion, 3)).astype(np.float32)
            final_colors = np.clip(base_colors + noise, 0, 255)
            output[lesion_ys, lesion_xs, 0] = final_colors[:, 0] / 255.0
            output[lesion_ys, lesion_xs, 1] = final_colors[:, 1] / 255.0
            output[lesion_ys, lesion_xs, 2] = final_colors[:, 2] / 255.0
        output[lesion_ys, lesion_xs, 3] = 1.0  # Alpha = fully opaque for lesions

    return output
