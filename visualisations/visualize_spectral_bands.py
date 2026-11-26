"""
Visualize individual spectral bands of a randomly chosen multispectral (.tif)
or hyperspectral (.bil) image, producing a panel layout suitable for
manuscript figures (matching the INPUT DATA section of the framework diagram).

Multispectral mode
    Each of the 6 spectral bands is shown as a separate panel with its label
    (e.g. Blue, Green, Red edge, NIR, UV-Red, UV-Blue).  The background mask
    (band 7, if present) is applied so only the specimen is visible.

Hyperspectral mode
    A selection of evenly-spaced bands across the full wavelength range
    (348–810 nm) is displayed, together with a shared colour-bar that maps
    wavelength to colour.  The exact number of panels is configurable.

Both modes produce a single-row panel strip that can be directly inserted into
a multi-panel figure.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Multispectral loading (rasterio / tifffile)
# ---------------------------------------------------------------------------

def _load_multispectral_tif(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (bands, mask) where bands is (C, H, W) float32 and mask is (H, W) bool or None."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
    except Exception:
        try:
            import tifffile as tiff
            arr = tiff.imread(path).astype(np.float32)
            if arr.ndim == 2:
                data = arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                if arr.shape[0] <= 10 and arr.shape[1] > 32 and arr.shape[2] > 32:
                    data = arr  # already (C, H, W)
                else:
                    data = np.moveaxis(arr, -1, 0)  # (H, W, C) -> (C, H, W)
            else:
                raise ValueError(f"Unexpected shape {arr.shape}")
        except Exception as e:
            raise RuntimeError(f"Cannot load {path}: {e}") from e

    if data.shape[0] >= 7:
        spectral = data[:6]
        mask = data[6] > 0
    else:
        spectral = data
        mask = None

    return spectral, mask


def _normalize_band(
    band: np.ndarray,
    mask: np.ndarray | None = None,
    plow: float = 2.0,
    phigh: float = 98.0,
) -> np.ndarray:
    """Percentile-stretch a 2-D array to [0, 1] using foreground pixels only."""
    if mask is not None:
        fg = band[mask]
        if fg.size > 0:
            lo = float(np.percentile(fg, plow))
            hi = float(np.percentile(fg, phigh))
        else:
            lo = float(np.percentile(band, plow))
            hi = float(np.percentile(band, phigh))
    else:
        lo = float(np.percentile(band, plow))
        hi = float(np.percentile(band, phigh))
    if hi - lo < 1e-8:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Hyperspectral loading (.bil via read_iml_hyp)
# ---------------------------------------------------------------------------

def _load_hyperspectral_bil(directory: str, basename: str):
    """Return (image, wavelengths) where image is (H, W, bands) and wavelengths is 1-D."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lep_dir = os.path.join(os.path.dirname(script_dir), "lepidoptera")
    if lep_dir not in sys.path:
        sys.path.insert(0, lep_dir)
    from read_iml_hyp import read_iml_hyp

    image, wavelengths, scan, _ = read_iml_hyp(directory, basename)
    return image, wavelengths


# ---------------------------------------------------------------------------
# Default band labels for multispectral (matching the figure style)
# ---------------------------------------------------------------------------

DEFAULT_MS_LABELS = [
    "Blue ($B_1$)",
    "Green ($B_2$)",
    "Red edge ($B_3$)",
    "NIR ($B_4$)",
    "UV-Red ($B_5$)",
    "UV-Blue ($B_6$)",
]

# RGB tint for each multispectral band so panels show spectral colour
# rather than plain greyscale.  Each entry is (R, G, B) in [0, 1].
DEFAULT_MS_COLORS: list[tuple[float, float, float]] = [
    (0.15, 0.30, 0.90),   # Blue
    (0.20, 0.75, 0.20),   # Green
    (0.90, 0.35, 0.10),   # Red edge
    (0.80, 0.10, 0.10),   # NIR (deep red)
    (0.70, 0.10, 0.55),   # UV-Red (magenta)
    (0.45, 0.10, 0.85),   # UV-Blue (violet)
]

# ---------------------------------------------------------------------------
# Wavelength → approximate visible colour (for hyperspectral colour bar)
# ---------------------------------------------------------------------------

def _wavelength_to_rgb(wl: float) -> tuple[float, float, float]:
    """Map a wavelength in nm to an approximate (R, G, B) tuple in [0, 1]."""
    if wl < 380:
        r, g, b = 0.4, 0.0, 0.4
    elif wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    elif wl <= 810:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r, g, b = 0.5, 0.0, 0.0

    # intensity fall-off at spectrum edges
    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 700 < wl <= 810:
        factor = 0.3 + 0.7 * (810 - wl) / (810 - 700)
    else:
        factor = 1.0

    return (r * factor, g * factor, b * factor)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _tint_band(
    intensity: np.ndarray,
    color: tuple[float, float, float],
    mask: np.ndarray | None = None,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Create an (H, W, 3) RGB image by modulating *color* with *intensity*.

    Where mask is False the pixel is filled with *bg*.
    """
    h, w = intensity.shape
    rgb = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = intensity * color[c]
    if mask is not None:
        for c in range(3):
            rgb[:, :, c] = np.where(mask, rgb[:, :, c], bg[c])
    return np.clip(rgb, 0.0, 1.0)


def _make_rgb_composite_ms(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Build a true-colour RGB composite from multispectral bands.

    Band order is assumed to be [Blue, Green, Red-edge, …].  The composite
    maps Red-edge → R, Green → G, Blue → B so the result approximates what a
    human would see.
    """
    n_bands = spectral.shape[0]
    r_idx, g_idx, b_idx = min(2, n_bands - 1), min(1, n_bands - 1), 0
    r = _normalize_band(spectral[r_idx], mask)
    g = _normalize_band(spectral[g_idx], mask)
    b = _normalize_band(spectral[b_idx], mask)
    rgb = np.stack([r, g, b], axis=-1)
    if mask is not None:
        for c in range(3):
            rgb[:, :, c] = np.where(mask, rgb[:, :, c], bg[c])
    return np.clip(rgb, 0.0, 1.0)


def _make_rgb_composite_hyp(
    image: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Build a true-colour RGB composite from hyperspectral data.

    Picks the bands closest to 650 nm (R), 550 nm (G), 450 nm (B).
    """
    targets = [650, 550, 450]
    channels = []
    for target in targets:
        idx = int(np.argmin(np.abs(wavelengths - target)))
        band = image[:, :, idx].astype(np.float32)
        lo = np.percentile(band, 2)
        hi = np.percentile(band, 98)
        if hi - lo < 1e-8:
            channels.append(np.zeros_like(band))
        else:
            channels.append(np.clip((band - lo) / (hi - lo), 0.0, 1.0))
    return np.stack(channels, axis=-1)


def plot_multispectral_bands(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    band_labels: list[str] | None = None,
    band_colors: list[tuple[float, float, float]] | None = None,
    title: str | None = None,
    output_path: str = "multispectral_bands.png",
    bg_color: str = "white",
    dpi: int = 300,
    show_rgb: bool = True,
) -> None:
    """Plot each spectral band as a colour-tinted panel, with an optional
    leading RGB composite showing what a human would see."""
    n_bands = spectral.shape[0]
    if band_labels is None:
        if n_bands <= len(DEFAULT_MS_LABELS):
            band_labels = DEFAULT_MS_LABELS[:n_bands]
        else:
            band_labels = [f"$B_{{{i+1}}}$" for i in range(n_bands)]

    if band_colors is None:
        if n_bands <= len(DEFAULT_MS_COLORS):
            band_colors = DEFAULT_MS_COLORS[:n_bands]
        else:
            band_colors = [_wavelength_to_rgb(wl) for wl in np.linspace(400, 800, n_bands)]

    bg = (1.0, 1.0, 1.0) if bg_color == "white" else (0.0, 0.0, 0.0)

    n_cols = n_bands + (1 if show_rgb else 0)
    fig_w = max(12, 2.4 * n_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 3.2))
    if n_cols == 1:
        axes = [axes]

    col = 0

    if show_rgb:
        rgb_composite = _make_rgb_composite_ms(spectral, mask, bg)
        axes[col].imshow(rgb_composite)
        axes[col].set_title("RGB (human eye)", fontsize=13, fontweight="bold", color="#333333")
        axes[col].axis("off")
        col += 1

    for i in range(n_bands):
        ax = axes[col]
        intensity = _normalize_band(spectral[i], mask)
        rgb = _tint_band(intensity, band_colors[i], mask, bg)
        ax.imshow(rgb)
        ax.set_title(band_labels[i], fontsize=13, fontweight="bold", color="#1a5276")
        ax.axis("off")
        col += 1

    if title:
        fig.suptitle(title, fontsize=14, y=1.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved multispectral band figure → {output_path}")


def plot_hyperspectral_bands(
    image: np.ndarray,
    wavelengths: np.ndarray,
    n_panels: int = 5,
    title: str | None = None,
    output_path: str = "hyperspectral_bands.png",
    dpi: int = 300,
    show_rgb: bool = True,
    show_ellipsis: bool = True,
) -> None:
    """Plot a few representative hyperspectral bands with ``...`` gaps between
    them, preceded by an optional RGB composite.

    The layout mirrors the reference figure style::

        [RGB]  [λ₁ 348nm]  ...  [λ₂ 460nm]  ...  [λ₃ 580nm]  ...  [λₙ 810nm]
                          ─── wavelength colour bar ───

    Parameters
    ----------
    n_panels : int
        Number of actual band images to show (default 5).
    show_ellipsis : bool
        If True, insert ``...`` spacer columns between bands to convey the
        hundreds of intervening channels.
    """
    n_total = image.shape[2]
    if n_panels > n_total:
        n_panels = n_total

    indices = np.linspace(0, n_total - 1, n_panels, dtype=int)

    wl_colors = np.array([_wavelength_to_rgb(w) for w in wavelengths])

    # Build the column sequence: interleave band panels with ellipsis spacers
    # Each entry is ("band", band_idx) or ("ellipsis", None)
    columns: list[tuple[str, int | None]] = []
    if show_rgb:
        columns.append(("rgb", None))
    for i, band_idx in enumerate(indices):
        columns.append(("band", int(band_idx)))
        if show_ellipsis and i < len(indices) - 1:
            columns.append(("ellipsis", None))

    n_cols = len(columns)
    # Narrow ellipsis columns, wider image columns
    width_ratios = []
    for kind, _ in columns:
        if kind == "ellipsis":
            width_ratios.append(0.25)
        else:
            width_ratios.append(1.0)

    fig_w = max(12, sum(width_ratios) * 2.0)
    fig = plt.figure(figsize=(fig_w, 4.0))

    # The colour bar only spans the band+ellipsis columns (not the RGB column)
    cb_start = 1 if show_rgb else 0
    gs = gridspec.GridSpec(
        2, n_cols,
        height_ratios=[1, 0.04],
        hspace=0.18,
        wspace=0.05,
        width_ratios=width_ratios,
    )

    band_counter = 0
    for col_idx, (kind, band_idx) in enumerate(columns):
        ax = fig.add_subplot(gs[0, col_idx])

        if kind == "rgb":
            rgb_composite = _make_rgb_composite_hyp(image, wavelengths)
            ax.imshow(np.clip(rgb_composite, 0.0, 1.0))
            ax.set_title("RGB\n(human eye)", fontsize=10, fontweight="bold",
                         color="#333333")
            ax.axis("off")

        elif kind == "ellipsis":
            ax.text(0.5, 0.5, "...", transform=ax.transAxes,
                    ha="center", va="center", fontsize=20,
                    fontweight="bold", color="#888888")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        else:  # "band"
            band = image[:, :, band_idx].astype(np.float32)
            lo = np.percentile(band, 2)
            hi = np.percentile(band, 98)
            if hi - lo < 1e-8:
                normed = np.zeros_like(band)
            else:
                normed = np.clip((band - lo) / (hi - lo), 0.0, 1.0)

            wl = wavelengths[band_idx]
            color = _wavelength_to_rgb(wl)
            h, w = normed.shape
            rgb = np.empty((h, w, 3), dtype=np.float32)
            for c in range(3):
                rgb[:, :, c] = normed * color[c]
            rgb = np.clip(rgb, 0.0, 1.0)

            ax.imshow(rgb)
            band_counter += 1
            if band_counter == len(indices):
                label = f"$\\lambda_N$\n{wl:.0f} nm"
            else:
                label = f"$\\lambda_{{{band_counter}}}$\n{wl:.0f} nm"
            ax.set_title(label, fontsize=10, color="#1a5276",
                         fontweight="bold")
            ax.axis("off")

    # Wavelength colour bar spanning band + ellipsis columns
    ax_cb = fig.add_subplot(gs[1, cb_start:])
    gradient = wl_colors[np.newaxis, :, :]
    ax_cb.imshow(gradient, aspect="auto",
                 extent=[wavelengths[0], wavelengths[-1], 0, 1])
    ax_cb.set_xlim(wavelengths[0], wavelengths[-1])
    ax_cb.set_yticks([])
    ax_cb.set_xlabel("Wavelength (nm)", fontsize=10)
    ax_cb.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=14, y=1.04)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved hyperspectral band figure → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize spectral bands of a randomly chosen multispectral or hyperspectral image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Directory containing .tif (multispectral) or .bil (hyperspectral) files.",
    )
    p.add_argument(
        "--mode", choices=["multispectral", "hyperspectral", "auto"], default="auto",
        help="Data modality. 'auto' inspects file extensions in the directory.",
    )
    p.add_argument(
        "--sample", default=None,
        help="Specific filename (stem or full) to visualize. If omitted, one is chosen at random.",
    )
    p.add_argument(
        "--output", default=None,
        help="Output figure path. Default: spectral_bands_<stem>.png in the current directory.",
    )
    p.add_argument(
        "--n-panels", type=int, default=8,
        help="Number of band panels for hyperspectral mode (default 8).",
    )
    p.add_argument(
        "--band-labels", nargs="+", default=None,
        help="Custom band labels for multispectral mode (one per band).",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for sample selection.")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (default 300).")
    p.add_argument(
        "--bg-color", choices=["white", "black"], default="white",
        help="Background colour for masked-out regions (multispectral only).",
    )
    p.add_argument(
        "--no-rgb", action="store_true", default=False,
        help="Omit the leading RGB (human-eye) composite panel.",
    )
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect mode ----------------------------------------------------------
    tif_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True))
    tif_files += sorted(glob.glob(os.path.join(data_dir, "**", "*.tiff"), recursive=True))
    bil_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.bil"), recursive=True))

    mode = args.mode
    if mode == "auto":
        if tif_files and not bil_files:
            mode = "multispectral"
        elif bil_files and not tif_files:
            mode = "hyperspectral"
        elif tif_files and bil_files:
            print("Both .tif and .bil files found; defaulting to multispectral. "
                  "Use --mode to override.")
            mode = "multispectral"
        else:
            print("No .tif or .bil files found in the directory.", file=sys.stderr)
            sys.exit(1)

    # Select sample --------------------------------------------------------
    if mode == "multispectral":
        pool = tif_files
        if not pool:
            print("No .tif files found.", file=sys.stderr)
            sys.exit(1)

        if args.sample:
            match = [f for f in pool if args.sample in f]
            if not match:
                print(f"No file matching '{args.sample}' in {data_dir}", file=sys.stderr)
                sys.exit(1)
            chosen = match[0]
        else:
            chosen = random.choice(pool)

        stem = os.path.splitext(os.path.basename(chosen))[0]
        print(f"[multispectral] Loading: {chosen}")
        spectral, mask = _load_multispectral_tif(chosen)
        print(f"  shape: {spectral.shape} (bands, H, W) — mask {'present' if mask is not None else 'absent'}")

        out = args.output or f"spectral_bands_{stem}.png"
        plot_multispectral_bands(
            spectral, mask,
            band_labels=args.band_labels,
            title=stem,
            output_path=out,
            bg_color=args.bg_color,
            dpi=args.dpi,
            show_rgb=not args.no_rgb,
        )

    else:  # hyperspectral
        pool = bil_files
        if not pool:
            print("No .bil files found.", file=sys.stderr)
            sys.exit(1)

        if args.sample:
            match = [f for f in pool if args.sample in f]
            if not match:
                print(f"No file matching '{args.sample}' in {data_dir}", file=sys.stderr)
                sys.exit(1)
            chosen = match[0]
        else:
            chosen = random.choice(pool)

        directory = os.path.dirname(chosen)
        basename = os.path.splitext(os.path.basename(chosen))[0]
        print(f"[hyperspectral] Loading: {chosen}")
        image, wavelengths = _load_hyperspectral_bil(directory, basename)
        print(f"  shape: {image.shape} (H, W, bands), wavelengths {wavelengths[0]:.1f}–{wavelengths[-1]:.1f} nm")

        out = args.output or f"spectral_bands_{basename}.png"
        plot_hyperspectral_bands(
            image, wavelengths,
            n_panels=args.n_panels,
            title=basename,
            output_path=out,
            dpi=args.dpi,
            show_rgb=not args.no_rgb,
        )


if __name__ == "__main__":
    main()
