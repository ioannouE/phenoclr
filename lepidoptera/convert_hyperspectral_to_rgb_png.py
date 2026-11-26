#!/usr/bin/env python3
"""
convert_hyperspectral_to_rgb_png.py

Batch-convert hyperspectral specimen scans stored as paired .bil / .bil.hdr
files (see read_iml_hyp.py) into plain RGB PNGs, so they can be dropped
straight into image viewers, slide decks, or any tool that only understands
regular RGB(A) images.

For each .bil/.bil.hdr pair found under --input-dir:
  - The image is loaded via read_iml_hyp(), giving a (lines, samples, bands)
    array plus its wavelength calibration.
  - For each of R (650nm), G (550nm), B (450nm), a window of adjacent bands
    (see --band-window-nm) is averaged together -- mimicking the broad
    spectral passband of a real camera -- since a single ~1nm hyperspectral
    band is noisy enough that per-pixel noise shows up as colour speckle
    rather than grayscale grain. The three channels are then white-balanced
    (gray-world) and jointly percentile-stretched to [0, 255].
  - Unlike the multispectral bird .tif files, these hyperspectral scans do
    not carry an explicit foreground/background mask. Each specimen was
    photographed on a roughly uniform matte background, so by default this
    script estimates a background colour from the four image corners and
    treats near-background pixels as transparent (alpha = 0). Disable this
    with --no-mask-background to save fully opaque images instead.
  - The output is written as a PNG using the *same base filename* (i.e. the
    part before ``.bil``), preserving the input file's relative
    sub-directory structure inside the output directory.

Usage:
    python convert_hyperspectral_to_rgb_png.py \\
        --input-dir /path/to/bil/database \\
        --output-dir /path/to/rgb_png_output
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# read_iml_hyp.py lives alongside this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from read_iml_hyp import read_iml_hyp  # noqa: E402

RGB_TARGET_WAVELENGTHS_NM = (650.0, 550.0, 450.0)  # (Red, Green, Blue)


def find_bil_stems(input_dir: str) -> list[str]:
    """Find all .bil files under input_dir that have a matching .bil.hdr,
    returning their paths without the .bil extension."""
    bil_files = glob.glob(os.path.join(input_dir, "**", "*.bil"), recursive=True)
    stems = []
    for bil_path in bil_files:
        base = bil_path[:-4]  # strip ".bil"
        if os.path.exists(base + ".bil.hdr"):
            stems.append(base)
    return sorted(stems)


def rgb_band_indices(wavelengths: np.ndarray, n_bands: int,
                      targets=RGB_TARGET_WAVELENGTHS_NM) -> tuple[int, int, int]:
    """Pick the band index closest to each target wavelength, clipped to
    the image's actual number of bands (which can differ slightly from the
    reference wavelength table if dimensions were adjusted on read)."""
    n = min(n_bands, len(wavelengths))
    idxs = [int(np.argmin(np.abs(wavelengths[:n] - t))) for t in targets]
    return tuple(min(i, n_bands - 1) for i in idxs)


def rgb_band_windows(wavelengths: np.ndarray, n_bands: int, window_nm: float,
                      targets=RGB_TARGET_WAVELENGTHS_NM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each target wavelength, return the array of band indices falling
    within +/- window_nm/2 of it (always including at least the single
    nearest band).

    A single hyperspectral band is only ~1nm wide and has much lower
    signal-to-noise ratio than a real camera's broad RGB passband. Because
    the per-pixel noise in the R/G/B bands is statistically independent,
    building a composite from single bands turns that noise into random
    colour speckle rather than grayscale grain. Averaging a small window of
    adjacent bands for each channel (mimicking a real camera's broader
    spectral response) suppresses this speckle.
    """
    n = min(n_bands, len(wavelengths))
    wl = wavelengths[:n]
    windows = []
    for t in targets:
        in_window = np.where(np.abs(wl - t) <= window_nm / 2.0)[0]
        if in_window.size == 0:
            in_window = np.array([int(np.argmin(np.abs(wl - t)))])
        windows.append(np.clip(in_window, 0, n_bands - 1))
    return tuple(windows)


def percentile_stretch(band: np.ndarray, plow: float, phigh: float,
                        fg_mask: "np.ndarray | None" = None) -> np.ndarray:
    """Percentile-stretch a single band to [0, 1], optionally restricting
    the percentile computation to foreground pixels only."""
    fg = band[fg_mask] if fg_mask is not None and fg_mask.any() else band.ravel()
    lo, hi = float(np.percentile(fg, plow)), float(np.percentile(fg, phigh))
    if hi - lo < 1e-8:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gray_world_white_balance(bands: list[np.ndarray],
                              fg_mask: "np.ndarray | None" = None) -> list[np.ndarray]:
    """Rescale each band so their foreground means are equal (gray-world
    white balance).

    Different spectral bands from a hyperspectral/multispectral sensor often
    have genuinely different average digital counts for the *same* neutral
    surface, due to sensor quantum efficiency, filter transmission, and
    illumination spectrum -- not because the scene is actually that colour.
    Gray-world balancing corrects for this systematic per-band gain (assuming
    the specimen's average reflectance is roughly neutral) while leaving
    pixel-to-pixel colour *contrast* (e.g. dark spots vs. pale background)
    intact, since it applies a single flat scale factor per band.
    """
    fg_values = [band[fg_mask] if fg_mask is not None and fg_mask.any() else band.ravel()
                 for band in bands]
    means = np.array([float(v.mean()) for v in fg_values])
    target = float(means.mean())
    gains = target / np.clip(means, 1e-8, None)
    return [band.astype(np.float32) * g for band, g in zip(bands, gains)]


def joint_percentile_stretch(bands: list[np.ndarray], plow: float, phigh: float,
                              fg_mask: "np.ndarray | None" = None) -> list[np.ndarray]:
    """Percentile-stretch a set of bands (e.g. R, G, B) using a single shared
    (lo, hi) range computed across all of them together.

    Unlike stretching each band independently, this preserves the *relative*
    brightness between bands and only adjusts overall exposure/contrast.
    Combine with ``gray_world_white_balance`` first to also correct for
    systematic per-band gain differences.
    """
    fg_values = [band[fg_mask] if fg_mask is not None and fg_mask.any() else band.ravel()
                 for band in bands]
    pooled = np.concatenate(fg_values)
    lo, hi = float(np.percentile(pooled, plow)), float(np.percentile(pooled, phigh))
    if hi - lo < 1e-8:
        return [np.zeros_like(band, dtype=np.float32) for band in bands]
    return [np.clip((band.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            for band in bands]


def estimate_background_mask(rgb: np.ndarray, corner_frac: float = 0.06,
                              distance_thresh: float = 0.12) -> np.ndarray:
    """Estimate a foreground mask by comparing each pixel's colour to a
    background colour sampled (median) from the four image corners.

    Assumes a roughly uniform matte background, as used for the specimen
    photography, and specimens that do not touch the image corners.
    """
    h, w = rgb.shape[:2]
    ch = max(1, int(h * corner_frac))
    cw = max(1, int(w * corner_frac))
    corners = np.concatenate([
        rgb[:ch, :cw].reshape(-1, 3),
        rgb[:ch, -cw:].reshape(-1, 3),
        rgb[-ch:, :cw].reshape(-1, 3),
        rgb[-ch:, -cw:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - bg_color[None, None, :], axis=-1)
    mask = dist > distance_thresh

    try:
        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
        mask = binary_closing(mask, structure=np.ones((5, 5)))
        mask = binary_opening(mask, structure=np.ones((3, 3)))
        mask = binary_fill_holes(mask)
    except ImportError:
        pass  # scipy not available; fall back to the raw threshold mask

    return mask


def convert_to_rgba(
    image: np.ndarray,
    wavelengths: np.ndarray,
    plow: float,
    phigh: float,
    mask_background: bool,
    corner_frac: float,
    distance_thresh: float,
    band_window_nm: float = 20.0,
) -> np.ndarray:
    """Return an (H, W, 4) uint8 RGBA array from a (lines, samples, bands)
    hyperspectral image."""
    r_win, g_win, b_win = rgb_band_windows(wavelengths, image.shape[2], band_window_nm)

    def _stack(fg_mask=None):
        raw = [
            image[:, :, r_win].mean(axis=2),
            image[:, :, g_win].mean(axis=2),
            image[:, :, b_win].mean(axis=2),
        ]
        balanced = gray_world_white_balance(raw, fg_mask)
        r, g, b = joint_percentile_stretch(balanced, plow, phigh, fg_mask)
        return np.stack([r, g, b], axis=-1)

    rgb = _stack()

    if mask_background:
        fg_mask = estimate_background_mask(rgb, corner_frac, distance_thresh)
        # Re-stretch contrast using only foreground pixels for a nicer result.
        rgb = _stack(fg_mask)
        alpha = fg_mask.astype(np.float32)
    else:
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)

    rgba = np.dstack([rgb, alpha])
    return (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)


def _convert_one(task: dict) -> tuple[str, str, str]:
    """Convert a single .bil/.bil.hdr pair. Runs in a worker process.

    Returns (status, rel_display, message) where status is one of
    "ok", "skipped", "failed".
    """
    stem_path = task["stem_path"]
    input_dir = task["input_dir"]
    output_dir = task["output_dir"]

    directory = os.path.dirname(stem_path)
    name = os.path.basename(stem_path)
    rel_dir = os.path.relpath(directory, input_dir)
    out_dir = output_dir if rel_dir == "." else os.path.join(output_dir, rel_dir)
    out_path = os.path.join(out_dir, name + ".png")
    rel_display = os.path.join(rel_dir, name) if rel_dir != "." else name

    if task["skip_existing"] and os.path.exists(out_path):
        return "skipped", rel_display, out_path

    try:
        from PIL import Image

        image, wavelengths, scan, gain = read_iml_hyp(directory, name, full_read=task["full_read"])
        rgba = convert_to_rgba(
            image, wavelengths, task["plow"], task["phigh"],
            mask_background=task["mask_background"],
            corner_frac=task["corner_frac"],
            distance_thresh=task["distance_thresh"],
            band_window_nm=task["band_window_nm"],
        )
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        return "ok", rel_display, out_path
    except Exception as e:
        return "failed", rel_display, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert hyperspectral .bil/.bil.hdr specimen scans to RGB PNGs, "
            "preserving filenames and folder structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                         help="Directory containing the .bil/.bil.hdr database (searched recursively).")
    parser.add_argument("--output-dir", required=True,
                         help="Directory to write the converted RGB .png files into.")
    parser.add_argument("--plow", type=float, default=2.0,
                         help="Lower percentile for contrast stretching (default 2.0).")
    parser.add_argument("--phigh", type=float, default=98.0,
                         help="Upper percentile for contrast stretching (default 98.0).")
    parser.add_argument("--band-window-nm", type=float, default=20.0,
                         help="Width (nm) of the band window averaged for each of R/G/B, "
                              "centred on 650/550/450nm. Wider windows reduce single-band "
                              "sensor noise/speckle at the cost of spectral precision "
                              "(default 20.0nm; use a small value e.g. 1 to approximate "
                              "single-band selection).")
    parser.add_argument("--no-mask-background", action="store_true",
                         help="Disable background transparency estimation; save fully opaque images.")
    parser.add_argument("--corner-frac", type=float, default=0.06,
                         help="Fraction of image height/width sampled from each corner "
                              "to estimate the background colour (default 0.06).")
    parser.add_argument("--bg-distance-threshold", type=float, default=0.12,
                         help="RGB distance (in [0,1] space) above which a pixel is "
                              "considered foreground (default 0.12).")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip files whose output .png already exists.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process this many files (useful for testing).")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of files to convert in parallel using separate processes "
                              "(default: number of CPU cores). This is an I/O- and CPU-bound "
                              "batch job (file reads + NumPy/SciPy array processing), so it "
                              "benefits from multiple CPU cores, not a GPU/MPS device. "
                              "If --input-dir is on a network/shared drive, keep this modest "
                              "(e.g. 2-4) to avoid saturating the network link -- more workers "
                              "issuing concurrent reads can make things *slower*, not faster.")
    parser.add_argument("--no-full-read", action="store_true",
                         help="Use lazy memory-mapped reads instead of loading each .bil file "
                              "fully into RAM in one sequential read. Full reads (the default) "
                              "are strongly recommended when --input-dir is on a network/shared "
                              "drive: since we only need a subset of bands, memmap page faults "
                              "become thousands of small scattered reads across the file, which "
                              "is very slow over a network (each incurs real round-trip latency) "
                              "even though it's fine on a local SSD. Fall back to memmap (this "
                              "flag) if files are too large to fit comfortably in RAM, especially "
                              "when combined with multiple --workers.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    stems = find_bil_stems(input_dir)
    if not stems:
        print(f"No .bil/.bil.hdr pairs found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        stems = stems[: args.limit]

    workers = args.workers or os.cpu_count() or 1
    print(f"Found {len(stems)} .bil/.bil.hdr pairs under {input_dir} "
          f"(using {workers} worker process{'es' if workers != 1 else ''})")
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        {
            "stem_path": stem_path,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "plow": args.plow,
            "phigh": args.phigh,
            "mask_background": not args.no_mask_background,
            "corner_frac": args.corner_frac,
            "distance_thresh": args.bg_distance_threshold,
            "band_window_nm": args.band_window_nm,
            "skip_existing": args.skip_existing,
            "full_read": not args.no_full_read,
        }
        for stem_path in stems
    ]

    n_ok, n_skipped, n_failed = 0, 0, 0
    n_done = 0
    start = time.time()

    if workers <= 1:
        results = (_convert_one(task) for task in tasks)
        for status, rel_display, message in results:
            n_done += 1
            if status == "ok":
                n_ok += 1
                print(f"  [{n_done}/{len(tasks)}] {rel_display}.bil -> "
                      f"{os.path.relpath(message, output_dir)}")
            elif status == "skipped":
                n_skipped += 1
            else:
                n_failed += 1
                print(f"  FAILED {rel_display}: {message}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_convert_one, task) for task in tasks]
            for future in as_completed(futures):
                status, rel_display, message = future.result()
                n_done += 1
                if status == "ok":
                    n_ok += 1
                    print(f"  [{n_done}/{len(tasks)}] {rel_display}.bil -> "
                          f"{os.path.relpath(message, output_dir)}")
                elif status == "skipped":
                    n_skipped += 1
                else:
                    n_failed += 1
                    print(f"  FAILED {rel_display}: {message}", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Converted: {n_ok}, skipped: {n_skipped}, failed: {n_failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
