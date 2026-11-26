"""
Produce publication-quality visualisations of a single 6-channel multispectral
bird specimen (.tif), highlighting the multi-band nature of the data.

Four outputs are always generated:

1.  **Stacked bands** — an isometric, exploded-layer view where each of the six
    spectral bands floats above the previous one, tinted by its characteristic
    spectral colour. Transparent background.

2.  **RGB composite** — a human-eye-like rendering (bands 0-2), transparent BG.

3.  **UV composite** — false-colour rendering (bands 4-5), transparent BG.

4.  **RGB+UV combined** — a blend of the RGB and UV composites (controllable
    via ``--blend``), transparent BG.

With ``--augmentations``, an additional panel figure is produced:

5.  **Augmentation panels** — each common 6-channel augmentation (rotate, flip,
    blur, crop, noise, erasing) applied to the full multispectral data and
    visualised side-by-side as both RGB and UV rows.

Band order convention (same as the rest of the pipeline):
    0: Blue   1: Green   2: Red-edge   3: NIR   4: UV-Red   5: UV-Blue
    (optional band 6: segmentation mask, non-zero = foreground)

Usage examples
--------------
    # Four standard outputs:
    python visualize_multispectral_specimen.py --tif /path/to/specimen.tif

    # With augmentation panel:
    python visualize_multispectral_specimen.py --tif /path/to/specimen.tif --augmentations

    # Pick a random TIF from a directory:
    python visualize_multispectral_specimen.py --data-dir /path/to/tifs/

    # Custom blend and output directory:
    python visualize_multispectral_specimen.py --tif /path/to/specimen.tif \\
        --output-dir ./figures --blend 0.4 --augmentations
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Band metadata
# ---------------------------------------------------------------------------

BAND_LABELS = [
    "Blue ($B_1$)",
    "Green ($B_2$)",
    "Red edge ($B_3$)",
    "NIR ($B_4$)",
    "UV-Red ($B_5$)",
    "UV-Blue ($B_6$)",
]

BAND_COLORS: list[tuple[float, float, float]] = [
    (0.15, 0.30, 0.90),   # Blue
    (0.20, 0.75, 0.20),   # Green
    (0.90, 0.35, 0.10),   # Red edge
    (0.80, 0.10, 0.10),   # NIR (deep red)
    (0.70, 0.10, 0.55),   # UV-Red (magenta)
    (0.45, 0.10, 0.85),   # UV-Blue (violet)
]

# Short, clean labels for the stacked view
BAND_SHORT = ["Blue", "Green", "Red edge", "NIR", "UV-Red", "UV-Blue"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_multispectral(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (spectral, mask) where spectral is (6, H, W) float32."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
    except Exception:
        try:
            import tifffile as tiff
            arr = tiff.imread(path).astype(np.float32)
            if arr.ndim == 2:
                data = arr[np.newaxis]
            elif arr.ndim == 3:
                if arr.shape[0] <= 10 and arr.shape[1] > 32 and arr.shape[2] > 32:
                    data = arr
                else:
                    data = np.moveaxis(arr, -1, 0)
            else:
                raise ValueError(f"Unexpected shape {arr.shape}")
        except Exception as e:
            raise RuntimeError(f"Cannot load {path}: {e}") from e

    mask = data[6] > 0 if data.shape[0] >= 7 else None
    spectral = data[:6]
    return spectral, mask


def _normalize(band: np.ndarray, mask: np.ndarray | None,
               plow: float = 2.0, phigh: float = 98.0) -> np.ndarray:
    """Percentile-stretch to [0, 1], using foreground pixels only when mask is available."""
    fg = band[mask] if mask is not None and mask.any() else band.ravel()
    lo, hi = float(np.percentile(fg, plow)), float(np.percentile(fg, phigh))
    if hi - lo < 1e-8:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Stacked / exploded-layer visualisation
# ---------------------------------------------------------------------------


def plot_stacked_bands(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    output_path: str = "stacked_bands.png",
    dpi: int = 300,
    layer_gap: float = 0.12,
    shear_angle: float = 15.0,
    add_labels: bool = True,
    add_shadow: bool = True,
) -> None:
    """Render 6 bands as perspective-sheared layers stacked vertically.

    Each layer is tinted with its spectral colour, sheared for an isometric
    look, and composited bottom-to-top onto a transparent canvas.  Labels
    float beside each layer.

    The output is a RGBA PNG with transparent background — ideal for
    compositing into manuscript figures.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    n_bands = spectral.shape[0]
    h, w = spectral.shape[1], spectral.shape[2]
    normed = [_normalize(spectral[i], mask) for i in range(n_bands)]

    # Shear coefficient (tan of angle)
    shear_k = np.tan(np.radians(shear_angle))
    extra_w = int(abs(shear_k) * h)

    # Vertical step between layers (pixels)
    v_step = max(int(h * layer_gap), 8)

    # Canvas dimensions
    canvas_w = w + extra_w + (280 if add_labels else 20)
    canvas_h = h + (n_bands - 1) * v_step + 20
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    def _shear_layer(rgba_f32: np.ndarray) -> np.ndarray:
        """Apply horizontal shear to an RGBA float32 image, return uint8."""
        out_w = w + extra_w
        out = np.zeros((h, out_w, 4), dtype=np.float32)
        for row in range(h):
            shift = int(shear_k * (h - 1 - row))
            out[row, shift:shift + w, :] = rgba_f32[row, :, :]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    sheared_layers: list[np.ndarray] = []
    for i in range(n_bands):
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        col = BAND_COLORS[i]
        for c in range(3):
            rgba[:, :, c] = normed[i] * col[c]
        alpha = normed[i] * 0.88 + 0.08
        if mask is not None:
            alpha = alpha * mask.astype(np.float32)
        rgba[:, :, 3] = alpha
        sheared_layers.append(_shear_layer(rgba))

    # Composite layers bottom-to-top
    for i in range(n_bands):
        layer = sheared_layers[i]
        lh, lw = layer.shape[:2]
        y0 = canvas_h - h - i * v_step
        y1 = y0 + lh
        x0 = 0
        x1 = min(x0 + lw, canvas_w)

        # Alpha-composite onto canvas
        src = layer[:, :x1 - x0, :].astype(np.float32) / 255.0
        dst = canvas[y0:y1, x0:x1, :].astype(np.float32) / 255.0
        sa = src[:, :, 3:4]
        da = dst[:, :, 3:4]
        out_a = sa + da * (1 - sa)
        safe_a = np.where(out_a > 0, out_a, 1.0)
        out_rgb = (src[:, :, :3] * sa + dst[:, :, :3] * da * (1 - sa)) / safe_a
        canvas[y0:y1, x0:x1, :3] = (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8)
        canvas[y0:y1, x0:x1, 3] = (np.clip(out_a[:, :, 0], 0, 1) * 255).astype(np.uint8)

    img = PILImage.fromarray(canvas, mode="RGBA")

    if add_labels:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial Bold", size=max(16, h // 22))
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(16, h // 22))
            except OSError:
                font = ImageFont.load_default()

        label_x = w + extra_w + 14
        for i in range(n_bands):
            y_centre = canvas_h - (h // 2) - i * v_step
            col_8bit = tuple(int(c * 255) for c in BAND_COLORS[i])
            draw.text(
                (label_x, y_centre),
                BAND_SHORT[i],
                fill=(*col_8bit, 230),
                font=font,
                anchor="lm",
            )

    if add_shadow:
        # Thin drop-shadow line under each layer for depth cue
        draw = ImageDraw.Draw(img) if not add_labels else draw
        for i in range(n_bands):
            y_bottom = canvas_h - i * v_step - 1
            x_left = int(shear_k * 0)
            x_right = w + extra_w
            if 0 <= y_bottom < canvas_h:
                for dx in range(x_left, min(x_right, canvas_w)):
                    cur = canvas[y_bottom, dx, 3]
                    if cur > 0:
                        draw.point((dx, min(y_bottom + 1, canvas_h - 1)),
                                   fill=(0, 0, 0, 35))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path, dpi=(dpi, dpi))
    print(f"Saved stacked-bands figure → {output_path}")


# ---------------------------------------------------------------------------
# 2. RGB composite with transparent background
# ---------------------------------------------------------------------------


def save_rgb_transparent(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    output_path: str = "rgb_transparent.png",
    dpi: int = 300,
) -> None:
    """Save an RGB composite (Red-edge→R, Green→G, Blue→B) with transparent BG."""
    r = _normalize(spectral[2], mask)  # Red edge
    g = _normalize(spectral[1], mask)  # Green
    b = _normalize(spectral[0], mask)  # Blue
    rgb = np.stack([r, g, b], axis=-1)

    if mask is not None:
        alpha = mask.astype(np.float32)
    else:
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)

    rgba = np.dstack([rgb, alpha])
    rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

    from PIL import Image as PILImage
    img = PILImage.fromarray(rgba, mode="RGBA")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path, dpi=(dpi, dpi))
    print(f"Saved RGB composite → {output_path}")


# ---------------------------------------------------------------------------
# 3. UV composite with transparent background
# ---------------------------------------------------------------------------


def save_uv_transparent(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    output_path: str = "uv_transparent.png",
    dpi: int = 300,
) -> None:
    """Save a false-colour UV composite (UV-Red→R, UV-Blue→B, average→G) with transparent BG."""
    uv_red = _normalize(spectral[4], mask)   # UV-Red
    uv_blue = _normalize(spectral[5], mask)  # UV-Blue
    uv_green = (uv_red + uv_blue) * 0.5

    rgb = np.stack([uv_red, uv_green, uv_blue], axis=-1)

    if mask is not None:
        alpha = mask.astype(np.float32)
    else:
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)

    rgba = np.dstack([rgb, alpha])
    rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

    from PIL import Image as PILImage
    img = PILImage.fromarray(rgba, mode="RGBA")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path, dpi=(dpi, dpi))
    print(f"Saved UV composite → {output_path}")


# ---------------------------------------------------------------------------
# 4. Combined RGB + UV composite
# ---------------------------------------------------------------------------


def _compute_band_stats(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    plow: float = 2.0,
    phigh: float = 98.0,
) -> list[tuple[float, float]]:
    """Compute (lo, hi) percentile per band from the full image foreground.

    These fixed stats can be reused to normalise crops with consistent
    colour balance.
    """
    stats = []
    for c in range(spectral.shape[0]):
        band = spectral[c]
        fg = band[mask] if mask is not None and mask.any() else band.ravel()
        lo = float(np.percentile(fg, plow))
        hi = float(np.percentile(fg, phigh))
        stats.append((lo, hi))
    return stats


def _normalize_with_stats(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalise a band to [0, 1] using pre-computed (lo, hi) stats."""
    if hi - lo < 1e-8:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _make_rgb_array(spectral: np.ndarray, mask: np.ndarray | None,
                    stats: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Return (H, W, 3) float32 RGB composite in [0, 1]."""
    if stats is not None:
        r = _normalize_with_stats(spectral[2], *stats[2])
        g = _normalize_with_stats(spectral[1], *stats[1])
        b = _normalize_with_stats(spectral[0], *stats[0])
    else:
        r = _normalize(spectral[2], mask)
        g = _normalize(spectral[1], mask)
        b = _normalize(spectral[0], mask)
    return np.stack([r, g, b], axis=-1)


def _make_uv_array(spectral: np.ndarray, mask: np.ndarray | None,
                   stats: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Return (H, W, 3) float32 false-colour UV composite in [0, 1]."""
    if stats is not None:
        uv_r = _normalize_with_stats(spectral[4], *stats[4])
        uv_b = _normalize_with_stats(spectral[5], *stats[5])
    else:
        uv_r = _normalize(spectral[4], mask)
        uv_b = _normalize(spectral[5], mask)
    uv_g = (uv_r + uv_b) * 0.5
    return np.stack([uv_r, uv_g, uv_b], axis=-1)


def save_rgbuv_combined(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    output_path: str = "rgbuv_combined.png",
    dpi: int = 300,
    blend: float = 0.5,
) -> None:
    """Save a blended RGB + UV composite with transparent background.

    ``blend`` controls the mix: 0.0 = pure RGB, 1.0 = pure UV, 0.5 = equal.
    """
    from PIL import Image as PILImage

    rgb = _make_rgb_array(spectral, mask)
    uv = _make_uv_array(spectral, mask)
    combined = rgb * (1 - blend) + uv * blend

    if mask is not None:
        alpha = mask.astype(np.float32)
    else:
        alpha = np.ones(combined.shape[:2], dtype=np.float32)

    rgba = np.dstack([combined, alpha])
    rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

    img = PILImage.fromarray(rgba, mode="RGBA")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path, dpi=(dpi, dpi))
    print(f"Saved RGB+UV combined composite → {output_path}")


# ---------------------------------------------------------------------------
# 5. Augmentations on 6-channel data
# ---------------------------------------------------------------------------


def _resize_array(arr: np.ndarray, target_h: int, target_w: int,
                   order: int = 1) -> np.ndarray:
    """Resize a 2-D array to (target_h, target_w) using scipy zoom."""
    from scipy.ndimage import zoom
    zh = target_h / arr.shape[0]
    zw = target_w / arr.shape[1]
    return zoom(arr.astype(np.float32), (zh, zw), order=order)


def _random_resized_crop_6ch(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    crop_size: int,
    rng: np.random.RandomState,
    scale: tuple[float, float] = (0.08, 0.60),
) -> tuple[np.ndarray, np.ndarray | None]:
    """SimCLR-style RandomResizedCrop: extract a sub-patch and resize it.

    The crop rectangle is sampled as a fraction of the **foreground bounding
    box** area (not the full image), then positioned randomly *within* the
    bounding box.  This guarantees every crop contains bird plumage detail
    while producing the zoomed-in patches characteristic of SimCLR training.
    """
    n_ch, h, w = spectral.shape

    # Foreground bounding box
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    else:
        by0, bx0, by1, bx1 = 0, 0, h, w

    bh, bw = by1 - by0, bx1 - bx0

    # Sample a sub-rectangle *inside* the foreground bbox
    cy, cx, ch, cw = by0, bx0, bh, bw  # fallback = full bbox
    for _ in range(50):
        frac = rng.uniform(scale[0], scale[1])
        aspect = np.exp(rng.uniform(np.log(3.0 / 4.0), np.log(4.0 / 3.0)))
        target_area = frac * bh * bw
        cw_t = int(round(np.sqrt(target_area * aspect)))
        ch_t = int(round(np.sqrt(target_area / aspect)))
        if cw_t < 4 or ch_t < 4 or cw_t > bw or ch_t > bh:
            continue
        cy = by0 + rng.randint(0, bh - ch_t + 1)
        cx = bx0 + rng.randint(0, bw - cw_t + 1)
        ch, cw = ch_t, cw_t
        break

    cropped = np.ascontiguousarray(spectral[:, cy:cy + ch, cx:cx + cw])
    resized = np.stack([
        _resize_array(cropped[c], crop_size, crop_size, order=1)
        for c in range(n_ch)
    ])

    if mask is not None:
        m_crop = np.ascontiguousarray(mask[cy:cy + ch, cx:cx + cw])
        m_resized = _resize_array(m_crop.astype(np.float32),
                                  crop_size, crop_size, order=0) > 0.5
    else:
        m_resized = None

    return resized, m_resized


def _apply_augmentations_6ch(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    seed: int = 42,
    crop_size: int = 224,
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Apply each augmentation to the full 6-channel image, SimCLR-style.

    Following the SimCLR pipeline, every augmentation panel gets a fresh
    ``RandomResizedCrop`` to ``(crop_size, crop_size)`` *before* the
    augmentation itself is applied.  This matches how training views are
    actually generated.

    Returns a list of (name, augmented_spectral, augmented_mask) tuples.
    """
    from scipy.ndimage import rotate as nd_rotate, gaussian_filter
    rng = np.random.RandomState(seed)

    results: list[tuple[str, np.ndarray, np.ndarray | None]] = []

    def _fresh_crop():
        return _random_resized_crop_6ch(spectral, mask, crop_size, rng)

    def _rot_bands(bands, angle, msk):
        out = np.stack([
            nd_rotate(bands[c], angle, reshape=False, order=1,
                      mode='constant', cval=0)
            for c in range(bands.shape[0])
        ])
        m_out = (nd_rotate(msk.astype(np.float32), angle, reshape=False,
                           order=0, mode='constant', cval=0) > 0.5
                 if msk is not None else None)
        return out, m_out

    # 1. Random Resized Crop (standalone — shows the spatial crop itself)
    cr_s, cr_m = _fresh_crop()
    results.append(("Random crop", cr_s, cr_m))

    # 2. Rotation (crop → rotate)
    cr_s, cr_m = _fresh_crop()
    angle = rng.uniform(15, 45) * rng.choice([-1, 1])
    rot_s, rot_m = _rot_bands(cr_s, angle, cr_m)
    results.append((f"Rotate {angle:.0f}°", rot_s, rot_m))

    # 3. Vertical flip (crop → flip)
    cr_s, cr_m = _fresh_crop()
    results.append(("Vertical flip", cr_s[:, ::-1, :].copy(),
                     cr_m[::-1, :].copy() if cr_m is not None else None))

    # 4. Horizontal flip (crop → flip)
    cr_s, cr_m = _fresh_crop()
    results.append(("Horizontal flip", cr_s[:, :, ::-1].copy(),
                     cr_m[:, ::-1].copy() if cr_m is not None else None))

    # 5. Gaussian blur (crop → blur)
    cr_s, cr_m = _fresh_crop()
    sigma = rng.uniform(1.5, 3.0)
    blurred = np.stack([gaussian_filter(cr_s[c], sigma=sigma)
                        for c in range(cr_s.shape[0])])
    results.append((f"Gaussian blur σ={sigma:.1f}", blurred, cr_m))

    # 6. Gaussian noise (crop → noise)
    cr_s, cr_m = _fresh_crop()
    noise_std = rng.uniform(0.03, 0.08)
    noisy = cr_s + (rng.randn(*cr_s.shape).astype(np.float32)
                    * noise_std * cr_s.max())
    results.append((f"Gaussian noise σ={noise_std:.2f}", noisy.clip(0), cr_m))

    # 7. Random erasing (crop → erase)
    cr_s, cr_m = _fresh_crop()
    erased = cr_s.copy()
    h, w = cr_s.shape[1], cr_s.shape[2]
    er_h = rng.randint(h // 8, h // 3)
    er_w = rng.randint(w // 8, w // 3)
    er_y = rng.randint(0, h - er_h)
    er_x = rng.randint(0, w - er_w)
    erased[:, er_y:er_y + er_h, er_x:er_x + er_w] = 0
    results.append(("Random erasing", erased, cr_m))

    return results


def _make_combined_array(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    blend: float = 0.5,
    stats: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return (H, W, 3) float32 blended RGB+UV composite."""
    rgb = _make_rgb_array(spectral, mask, stats)
    uv = _make_uv_array(spectral, mask, stats)
    return rgb * (1 - blend) + uv * blend


def plot_augmentation_panels(
    spectral: np.ndarray,
    mask: np.ndarray | None,
    output_path: str = "augmentation_panels.png",
    dpi: int = 300,
    seed: int = 42,
    crop_size: int = 224,
    blend: float = 0.5,
) -> None:
    """Produce a panel grid: original + augmentations as RGB, UV, and Combined.

    Each augmentation is preceded by a fresh SimCLR-style RandomResizedCrop
    to ``(crop_size, crop_size)``.  The first column shows the uncropped
    original at full resolution for comparison.

    Layout::

        Row 0 (RGB):      Original | Crop | Rotate | V-flip | ...
        Row 1 (UV):       Original | Crop | Rotate | V-flip | ...
        Row 2 (Combined): Original | Crop | Rotate | V-flip | ...
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    augmented = _apply_augmentations_6ch(spectral, mask, seed=seed,
                                         crop_size=crop_size)
    n_cols = 1 + len(augmented)
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_rows + 0.6))
    fig.patch.set_facecolor("white")

    row_labels = ["RGB", "UV", "Combined"]

    def _show(ax, img_rgb, msk, title=None, ylabel=None):
        clipped = np.clip(img_rgb, 0, 1)
        if msk is not None:
            bg = np.ones_like(clipped)
            m3 = msk.astype(np.float32)[:, :, np.newaxis]
            display = clipped * m3 + bg * (1 - m3)
        else:
            display = clipped
        ax.set_facecolor("white")
        ax.imshow(display)
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", rotation=90,
                          labelpad=8)

    # Compute normalisation stats once from the full original image so all
    # crops use the same colour mapping.
    band_stats = _compute_band_stats(spectral, mask)

    # Column 0: uncropped original
    orig_rgb = _make_rgb_array(spectral, mask, band_stats)
    orig_uv = _make_uv_array(spectral, mask, band_stats)
    orig_comb = _make_combined_array(spectral, mask, blend, band_stats)

    _show(axes[0, 0], orig_rgb, mask, "Original", ylabel=row_labels[0])
    _show(axes[1, 0], orig_uv, mask, "Original", ylabel=row_labels[1])
    _show(axes[2, 0], orig_comb, mask, "Original", ylabel=row_labels[2])

    # Augmented columns — use the same band_stats for consistent colours
    for col, (name, aug_s, aug_m) in enumerate(augmented, start=1):
        aug_rgb = _make_rgb_array(aug_s, aug_m, band_stats)
        aug_uv = _make_uv_array(aug_s, aug_m, band_stats)
        aug_comb = _make_combined_array(aug_s, aug_m, blend, band_stats)
        _show(axes[0, col], aug_rgb, aug_m, name)
        _show(axes[1, col], aug_uv, aug_m, name)
        _show(axes[2, col], aug_comb, aug_m, name)

    plt.tight_layout(pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved augmentation panels → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Visualise a single 6-channel multispectral bird specimen. "
            "Produces a stacked-bands figure, an RGB composite, and a UV composite — "
            "all with transparent backgrounds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tif", help="Path to a specific .tif file.")
    src.add_argument("--data-dir", help="Pick a random .tif from this directory.")

    p.add_argument("--sample", default=None,
                   help="When using --data-dir, select this filename stem instead of random.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--output-stacked", default=None,
                   help="Output path for the stacked-bands figure.")
    p.add_argument("--output-rgb", default=None,
                   help="Output path for the RGB composite.")
    p.add_argument("--output-uv", default=None,
                   help="Output path for the UV composite.")
    p.add_argument("--output-combined", default=None,
                   help="Output path for the RGB+UV blended composite.")
    p.add_argument("--output-dir", default=None,
                   help="Directory for auto-named outputs (default: current dir).")

    p.add_argument("--blend", type=float, default=0.5,
                   help="RGB/UV blend ratio for combined output: 0=pure RGB, 1=pure UV (default 0.5).")
    p.add_argument("--layer-gap", type=float, default=0.12,
                   help="Vertical gap between layers as fraction of image height (default 0.12).")
    p.add_argument("--shear-angle", type=float, default=15.0,
                   help="Shear angle in degrees for isometric tilt (default 15).")
    p.add_argument("--augmentations", action="store_true", default=False,
                   help="Also produce an augmentation panel figure showing 6-channel "
                        "augmentations visualised as RGB, UV, and Combined rows.")
    p.add_argument("--crop-size", type=int, default=224,
                   help="Square crop size for augmentations, matching SimCLR input_size (default 224).")
    p.add_argument("--output-augmentations", default=None,
                   help="Output path for the augmentation panel (requires --augmentations).")

    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Resolve TIF path
    if args.tif:
        tif_path = os.path.abspath(args.tif)
    else:
        data_dir = os.path.abspath(args.data_dir)
        pool = sorted(
            glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True)
            + glob.glob(os.path.join(data_dir, "**", "*.tiff"), recursive=True)
        )
        if not pool:
            print(f"No .tif files found under {data_dir}", file=sys.stderr)
            sys.exit(1)
        if args.sample:
            match = [f for f in pool if args.sample in f]
            if not match:
                print(f"No file matching '{args.sample}'", file=sys.stderr)
                sys.exit(1)
            tif_path = match[0]
        else:
            tif_path = random.choice(pool)

    stem = os.path.splitext(os.path.basename(tif_path))[0]
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {tif_path}")
    spectral, mask = load_multispectral(tif_path)
    print(f"  Bands: {spectral.shape[0]}, H×W: {spectral.shape[1]}×{spectral.shape[2]}, "
          f"mask: {'yes' if mask is not None else 'no'}")

    out_stacked = args.output_stacked or os.path.join(out_dir, f"{stem}_stacked_bands.png")
    out_rgb = args.output_rgb or os.path.join(out_dir, f"{stem}_rgb.png")
    out_uv = args.output_uv or os.path.join(out_dir, f"{stem}_uv.png")
    out_combined = args.output_combined or os.path.join(out_dir, f"{stem}_rgbuv_combined.png")

    plot_stacked_bands(spectral, mask, output_path=out_stacked,
                       dpi=args.dpi, layer_gap=args.layer_gap,
                       shear_angle=args.shear_angle)
    save_rgb_transparent(spectral, mask, output_path=out_rgb, dpi=args.dpi)
    save_uv_transparent(spectral, mask, output_path=out_uv, dpi=args.dpi)
    save_rgbuv_combined(spectral, mask, output_path=out_combined,
                        dpi=args.dpi, blend=args.blend)

    saved = [
        f"  1. Stacked bands:    {out_stacked}",
        f"  2. RGB composite:    {out_rgb}",
        f"  3. UV composite:     {out_uv}",
        f"  4. RGB+UV combined:  {out_combined}",
    ]

    if args.augmentations:
        out_aug = args.output_augmentations or os.path.join(out_dir, f"{stem}_augmentations.png")
        plot_augmentation_panels(spectral, mask, output_path=out_aug,
                                 dpi=args.dpi, seed=args.seed or 42,
                                 crop_size=args.crop_size, blend=args.blend)
        saved.append(f"  5. Augmentations:    {out_aug}")

    print(f"\nDone. {len(saved)} files saved:")
    for line in saved:
        print(line)


if __name__ == "__main__":
    main()
