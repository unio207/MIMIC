import argparse
from pathlib import Path
import cv2
import numpy as np


def process_mask(img: np.ndarray) -> np.ndarray:
    # Ensure single-channel grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: force binary-like values (0/255)
    _, bin_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Elliptical 5x5 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 1) Close: fills small gaps/holes
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) Dilate: expands lesion edges slightly
    dilated = cv2.dilate(closed, kernel, iterations=1)

    return dilated


def main():
    parser = argparse.ArgumentParser(
        description="Apply morphological Close (2x) then Dilate (1x) to mask images."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "gan_a" / "phase_1" / "lesion_synthetic_binary"),
        help="Input folder with mask images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "gan_a" / "phase_1" / "lesion_synthetic_binary_processed"),
        help="Output folder for processed masks",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images recursively in subfolders",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = (
        [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]
        if args.recursive
        else [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )

    if not paths:
        print(f"No images found in: {in_dir}")
        return

    count = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipped unreadable file: {p}")
            continue

        processed = process_mask(img)

        # Preserve relative structure if recursive
        rel = p.relative_to(in_dir) if args.recursive else Path(p.name)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dst), processed)
        count += 1

    print(f"Processed {count} image(s).")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()