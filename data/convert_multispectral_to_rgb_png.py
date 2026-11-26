#!/usr/bin/env python3
"""
convert_multispectral_to_rgb_png.py

Batch-convert the multispectral/hyperspectral specimen images in the
database (6-band .tif files, see simclr/data/multispectral_data.py) into
plain RGB PNGs with a transparent background, so they can be dropped straight
into image viewers, slide decks, or any tool that only understands regular
RGB(A) images.

For each input image:
  - RGB bands (Red-edge, Green, Blue -> R, G, B) are percentile-stretched to
    [0, 255], matching the convention used in
    simclr/visualisations/visualize_multispectral_specimen.py.
  - If a 7th "mask" band is present (non-zero = foreground), it is used as
    the alpha channel so the background becomes fully transparent. Otherwise
    the image is saved fully opaque.
  - The output is written as a PNG using the *same filename* (extension
    swapped to .png), preserving the input file's relative sub-directory
    structure inside the output directory.

Usage:
    python convert_multispectral_to_rgb_png.py \\
        --input-dir /path/to/database \\
        --output-dir /path/to/rgb_png_output

Optional flags let you skip files that already have an output PNG, and
control the percentile stretch used for contrast normalisation.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Band order convention used throughout the pipeline:
#   0: Blue   1: Green   2: Red-edge   3: NIR   4: UV-Red   5: UV-Blue
#   (optional band 6: segmentation mask, non-zero = foreground)
RGB_BAND_INDICES = (2, 1, 0)  # (Red, Green, Blue) source band indices


def load_multispectral(path: str) -> tuple[np.ndarray, "np.ndarray | None"]:
    """Load a multispectral .tif and return (spectral, mask).

    spectral: float32 array of shape (>=3, H, W)
    mask: bool array of shape (H, W), or None if no mask band is present.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
    except Exception:
        import tifffile as tiff
        arr = tiff.imread(path).astype(np.float32)
        if arr.ndim == 2:
            data = arr[np.newaxis]
        elif arr.ndim == 3:
            # Heuristic: channel-first if the first axis is small (few bands)
            # and the other two axes look like spatial dimensions.
            if arr.shape[0] <= 10 and arr.shape[1] > 32 and arr.shape[2] > 32:
                data = arr
            else:
                data = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} for {path}")

    mask = data[6] > 0 if data.shape[0] >= 7 else None
    return data, mask


def normalize(band: np.ndarray, mask: "np.ndarray | None",
              plow: float = 2.0, phigh: float = 98.0) -> np.ndarray:
    """Percentile-stretch a single band to [0, 1] using foreground pixels only
    (when a mask is available), matching visualize_multispectral_specimen.py.
    """
    fg = band[mask] if mask is not None and mask.any() else band.ravel()
    lo, hi = float(np.percentile(fg, plow)), float(np.percentile(fg, phigh))
    if hi - lo < 1e-8:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gray_world_white_balance(bands: list[np.ndarray],
                              mask: "np.ndarray | None" = None) -> list[np.ndarray]:
    """Rescale each band so their foreground means are equal (gray-world
    white balance).

    Different spectral bands from this sensor often have genuinely different
    average digital counts for the *same* neutral surface, due to sensor
    quantum efficiency, filter transmission, and illumination spectrum -- not
    because the scene is actually that colour. Gray-world balancing corrects
    for this systematic per-band gain (assuming the specimen's average
    reflectance is roughly neutral) while leaving pixel-to-pixel colour
    contrast intact, since it applies a single flat scale factor per band.
    """
    fg_values = [band[mask] if mask is not None and mask.any() else band.ravel()
                 for band in bands]
    means = np.array([float(v.mean()) for v in fg_values])
    target = float(means.mean())
    gains = target / np.clip(means, 1e-8, None)
    return [band.astype(np.float32) * g for band, g in zip(bands, gains)]


def joint_normalize(bands: list[np.ndarray], mask: "np.ndarray | None",
                     plow: float = 2.0, phigh: float = 98.0) -> list[np.ndarray]:
    """Percentile-stretch a set of bands (e.g. R, G, B) using a single shared
    (lo, hi) range computed across all of them together.

    Stretching each band independently (as ``normalize`` above does) discards
    the relative brightness between bands and tends to introduce false colour
    casts. Combine with ``gray_world_white_balance`` first to also correct
    for systematic per-band gain differences.
    """
    fg_values = [band[mask] if mask is not None and mask.any() else band.ravel()
                 for band in bands]
    pooled = np.concatenate(fg_values)
    lo, hi = float(np.percentile(pooled, plow)), float(np.percentile(pooled, phigh))
    if hi - lo < 1e-8:
        return [np.zeros_like(band, dtype=np.float32) for band in bands]
    return [np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) for band in bands]


def convert_to_rgba(spectral: np.ndarray, mask: "np.ndarray | None",
                     plow: float, phigh: float) -> np.ndarray:
    """Return an (H, W, 4) uint8 RGBA array with transparent background."""
    raw = [spectral[RGB_BAND_INDICES[0]], spectral[RGB_BAND_INDICES[1]], spectral[RGB_BAND_INDICES[2]]]
    balanced = gray_world_white_balance(raw, mask)
    r, g, b = joint_normalize(balanced, mask, plow, phigh)
    rgb = np.stack([r, g, b], axis=-1)

    if mask is not None:
        alpha = mask.astype(np.float32)
    else:
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)

    rgba = np.dstack([rgb, alpha])
    return (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)


def find_input_files(input_dir: str) -> list[str]:
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files = set()
    for pattern in patterns:
        files.update(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    return sorted(files)


def _convert_one(task: dict) -> tuple[str, str, str]:
    """Convert a single .tif file. Runs in a worker process.

    Returns (status, rel_path, message) where status is one of
    "ok", "skipped", "failed".
    """
    path = task["path"]
    input_dir = task["input_dir"]
    output_dir = task["output_dir"]

    rel_path = os.path.relpath(path, input_dir)
    rel_stem, _ = os.path.splitext(rel_path)
    out_path = os.path.join(output_dir, rel_stem + ".png")

    if task["skip_existing"] and os.path.exists(out_path):
        return "skipped", rel_path, out_path

    try:
        from PIL import Image

        spectral, mask = load_multispectral(path)
        if spectral.shape[0] < 3:
            return "failed", rel_path, f"only {spectral.shape[0]} band(s), need >= 3"

        rgba = convert_to_rgba(spectral, mask, task["plow"], task["phigh"])

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        return "ok", rel_path, out_path
    except Exception as e:
        return "failed", rel_path, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert multispectral .tif specimen images to RGB PNGs with a "
            "transparent background, preserving filenames and folder structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                         help="Directory containing the multispectral .tif database (searched recursively).")
    parser.add_argument("--output-dir", required=True,
                         help="Directory to write the converted RGB .png files into.")
    parser.add_argument("--plow", type=float, default=2.0,
                         help="Lower percentile for contrast stretching (default 2.0).")
    parser.add_argument("--phigh", type=float, default=98.0,
                         help="Upper percentile for contrast stretching (default 98.0).")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip files whose output .png already exists.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process this many files (useful for testing).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of files to convert in parallel using separate processes "
                              "(default: number of CPU cores). This is an I/O- and CPU-bound "
                              "batch job (file reads + NumPy/SciPy array processing), so it "
                              "benefits from multiple CPU cores, not a GPU/MPS device. "
                              "Use --workers 1 to run sequentially (e.g. for debugging).")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_input_files(input_dir)
    if not files:
        print(f"No .tif files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        files = files[: args.limit]

    workers = args.workers or os.cpu_count() or 1
    print(f"Found {len(files)} .tif files under {input_dir} "
          f"(using {workers} worker process{'es' if workers != 1 else ''})")
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        {
            "path": path,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "plow": args.plow,
            "phigh": args.phigh,
            "skip_existing": args.skip_existing,
        }
        for path in files
    ]

    n_ok, n_skipped, n_failed = 0, 0, 0
    n_done = 0
    start = time.time()

    if workers <= 1:
        results = (_convert_one(task) for task in tasks)
        for status, rel_path, message in results:
            n_done += 1
            if status == "ok":
                n_ok += 1
                print(f"  [{n_done}/{len(tasks)}] {rel_path} -> {os.path.relpath(message, output_dir)}")
            elif status == "skipped":
                n_skipped += 1
            else:
                n_failed += 1
                print(f"  FAILED {rel_path}: {message}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_convert_one, task) for task in tasks]
            for future in as_completed(futures):
                status, rel_path, message = future.result()
                n_done += 1
                if status == "ok":
                    n_ok += 1
                    print(f"  [{n_done}/{len(tasks)}] {rel_path} -> "
                          f"{os.path.relpath(message, output_dir)}")
                elif status == "skipped":
                    n_skipped += 1
                else:
                    n_failed += 1
                    print(f"  FAILED {rel_path}: {message}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Converted: {n_ok}, skipped: {n_skipped}, failed: {n_failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
