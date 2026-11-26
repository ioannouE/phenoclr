"""Recreate the "PC axes of colour pattern trait space" figure (e.g. manuscript Fig. S1).

For one (or two, side-by-side as panels A/B) embedding CSVs, this script:
  1. Z-scores the raw embedding features and fits a PCA.
  2. Reports the variance explained by each of the first ``--n_components`` PCs
     (default 8) and the *total* variance explained by all of them together
     (the number you report in the caption, e.g. "78%").
  3. Re-scales (z-transforms) the PC scores themselves, purely for plotting
     clarity, exactly as described in the caption ("PC scores have been
     re-scaled (z-transformed) for clarity").
  4. Plots each specimen as its own (lateral/side-view, by default) thumbnail
     image at its PC coordinates, for PC1 vs PC2, PC3 vs PC4, PC5 vs PC6 and
     PC7 vs PC8 (a 2x2 grid per panel), with axis labels annotated with the
     percentage of variance explained by that PC.

Specimens are matched to display images by the ``Genus_species_id_SEX``
prefix parsed from the embeddings filename (not by exact filename), since the
embeddings are usually computed on one body part/view (e.g. Back) while the
figure displays a different view (e.g. Side/lateral) of the same specimen.

Examples
--------
Single panel (one embedding set, e.g. patch/local-loss embeddings):

    python plot_pca_pattern_space.py \\
        --embeddings_csv patch_embeddings.csv \\
        --image_dir /path/to/Passerines-Segmented-Side-RGB-PNG \\
        --output_dir results/pca_patch

Two panels side by side (A = patch embeddings, B = whole-plumage embeddings),
recreating the two-panel manuscript figure in one image:

    python plot_pca_pattern_space.py \\
        --embeddings_csv patch_embeddings.csv \\
        --embeddings_csv_b whole_plumage_embeddings.csv \\
        --image_dir /path/to/Passerines-Segmented-Side-RGB-PNG \\
        --label_a "Patch" --label_b "Whole-plumage" \\
        --output_dir results/pca_patch_vs_whole
"""

from __future__ import annotations

import argparse
import os

import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from plot_embeddings_multispectral import (
    build_multispectral_index,
    load_multispectral_image_rgb,
)


# ---------------------------------------------------------------------------
# Loading embeddings
# ---------------------------------------------------------------------------

def load_embeddings_from_csv(path):
    """Load embeddings from CSV. Returns (embeddings, filenames)."""
    df = pd.read_csv(path)
    first = df.columns[0].lower()
    if first in ("filename", "file", "name", "image"):
        filenames = df.iloc[:, 0].astype(str).values
        embeddings = df.iloc[:, 1:].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    else:
        print(
            f"WARNING: first column of {path} ('{df.columns[0]}') does not look like a "
            "filename column; using row index as identifier."
        )
        filenames = np.array([str(i) for i in range(len(df))])
        embeddings = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    return embeddings, filenames


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(embeddings, n_components=8, random_state=42):
    """Standardize features, fit PCA, and z-transform the resulting PC scores.

    Returns a dict with:
      raw_scores:               (N, n_components) PCA scores (not re-scaled).
      z_scores:                 raw_scores z-transformed per-PC (for plotting only).
      explained_variance_ratio: fraction of total variance explained by each PC.
      total_variance_explained: sum of explained_variance_ratio (fraction, 0-1).
    """
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=n_components, random_state=random_state)
    raw_scores = pca.fit_transform(scaled)

    pc_mean = raw_scores.mean(axis=0)
    pc_std = raw_scores.std(axis=0, ddof=1)
    pc_std[pc_std == 0] = 1.0
    z_scores = (raw_scores - pc_mean) / pc_std

    explained_variance_ratio = pca.explained_variance_ratio_
    return {
        "pca": pca,
        "scaler": scaler,
        "raw_scores": raw_scores,
        "z_scores": z_scores,
        "explained_variance_ratio": explained_variance_ratio,
        "total_variance_explained": float(explained_variance_ratio.sum()),
    }


def save_variance_report(result, filenames, label, output_dir):
    """Write per-PC and cumulative variance explained to CSV, and print a summary."""
    n_components = len(result["explained_variance_ratio"])
    evr_pct = result["explained_variance_ratio"] * 100
    cumulative_pct = np.cumsum(evr_pct)

    variance_df = pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(n_components)],
        "explained_variance_pct": evr_pct,
        "cumulative_variance_pct": cumulative_pct,
    })
    variance_path = os.path.join(output_dir, f"pca_variance_explained_{label}.csv")
    variance_df.to_csv(variance_path, index=False, float_format="%.4f")

    scores_df = pd.DataFrame(
        result["raw_scores"], columns=[f"PC{i + 1}_raw" for i in range(n_components)]
    )
    z_df = pd.DataFrame(
        result["z_scores"], columns=[f"PC{i + 1}_z" for i in range(n_components)]
    )
    scores_out = pd.concat([pd.DataFrame({"filename": filenames}), scores_df, z_df], axis=1)
    scores_path = os.path.join(output_dir, f"pca_scores_{label}.csv")
    scores_out.to_csv(scores_path, index=False, float_format="%.6f")

    total_pct = evr_pct.sum()
    print(f"\n[{label}] Variance explained by each of the first {n_components} PCs:")
    for i, pct in enumerate(evr_pct):
        print(f"  PC{i + 1}: {pct:.1f}%")
    print(f"  --> Total variance explained by these {n_components} PCs: {total_pct:.1f}%")
    print(f"  Saved per-PC variance table to: {variance_path}")
    print(f"  Saved PCA scores to: {scores_path}")

    return variance_df, total_pct


# ---------------------------------------------------------------------------
# Matching specimens (embedding filenames) to display images
# ---------------------------------------------------------------------------

def _specimen_prefix(filename, n_parts=4):
    """``Genus_species_id_SEX_BodyPart...`` -> ``Genus_species_id_SEX`` (first n_parts tokens)."""
    stem = os.path.splitext(os.path.basename(str(filename).strip().replace("\\", "/")))[0]
    parts = stem.split("_")
    return "_".join(parts[:n_parts]) if len(parts) >= n_parts else stem


def find_display_image(filename, index, index_lower):
    """Find a display image for `filename`'s specimen, by decreasing prefix specificity.

    Tries an exact stem match first (display dir == embedding dir case), then
    the ``Genus_species_id_SEX`` prefix, then ``Genus_species_id``, then
    ``Genus_species``, matching against any indexed image that starts with
    that prefix followed by ``_`` (so a different body part/view is fine).
    """
    stem = os.path.splitext(os.path.basename(str(filename).strip().replace("\\", "/")))[0]

    if stem in index:
        return index[stem]
    if stem.lower() in index_lower:
        return index_lower[stem.lower()]

    for n_parts in (4, 3, 2):
        prefix = _specimen_prefix(filename, n_parts=n_parts)
        candidates = sorted(k for k in index if k == prefix or k.startswith(prefix + "_"))
        if candidates:
            if len(candidates) > 1:
                print(
                    f"Note: multiple display images match specimen prefix '{prefix}' "
                    f"for '{filename}' (e.g. {candidates[:3]}) — using '{candidates[0]}'"
                )
            return index[candidates[0]]

    return None


def load_thumbnail(path, thumb_size, transparent_bg=True):
    """Load an image and resize it (preserving aspect ratio) into a PIL Image thumbnail."""
    array = load_multispectral_image_rgb(path, transparent_bg=transparent_bg)
    if array is None:
        return None
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        print(f"Warning: unexpected image shape {array.shape} for {path}, skipping")
        return None
    mode = "RGBA" if array.shape[2] == 4 else "RGB"
    img = Image.fromarray(array, mode=mode)
    img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Panel data assembly (PCA + matched thumbnails, ready to plot)
# ---------------------------------------------------------------------------

def build_panel(embeddings_csv, image_dir, label, output_dir, n_components,
                 max_images, random_state, thumb_size, transparent_bg, index_workers):
    print(f"\n=== Panel {label}: loading embeddings from {embeddings_csv} ===")
    embeddings, filenames = load_embeddings_from_csv(embeddings_csv)
    print(f"Loaded {len(embeddings)} specimens, {embeddings.shape[1]} embedding dimensions")

    result = run_pca(embeddings, n_components=n_components, random_state=random_state)
    save_variance_report(result, filenames, label, output_dir)

    n = len(filenames)
    if max_images and n > max_images:
        print(f"Subsampling {max_images} of {n} specimens for the thumbnail plot (variance report above uses all {n})...")
        rng = np.random.default_rng(random_state)
        plot_indices = rng.choice(n, max_images, replace=False)
    else:
        plot_indices = np.arange(n)

    plot_filenames = filenames[plot_indices]
    plot_z_scores = result["z_scores"][plot_indices]

    print(f"Indexing display images under {image_dir}...")
    index, index_lower = build_multispectral_index(image_dir, index_workers=index_workers)
    print(f"Indexed {len(index)} raster files")

    print("Matching specimens to display images...")
    image_paths = []
    matched_indices = []
    for i, fname in enumerate(tqdm(plot_filenames, desc=f"[{label}] matching images")):
        path = find_display_image(fname, index, index_lower)
        if path is not None:
            image_paths.append(path)
            matched_indices.append(i)
        else:
            print(f"Warning: no display image found for specimen '{fname}', skipping")

    if not image_paths:
        raise ValueError(f"Panel {label}: no display images could be matched under {image_dir}")

    plot_z_scores = plot_z_scores[matched_indices]

    print(f"Loading {len(image_paths)} thumbnails...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        thumbnails = list(tqdm(
            executor.map(lambda p: load_thumbnail(p, thumb_size, transparent_bg), image_paths),
            total=len(image_paths),
            desc=f"[{label}] loading thumbnails",
        ))

    keep = [i for i, t in enumerate(thumbnails) if t is not None]
    thumbnails = [thumbnails[i] for i in keep]
    plot_z_scores = plot_z_scores[keep]
    print(f"[{label}] {len(thumbnails)} specimens ready to plot")

    return {
        "label": label,
        "z_scores": plot_z_scores,
        "thumbnails": thumbnails,
        "explained_variance_ratio": result["explained_variance_ratio"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panels(panels, output_path, n_components, thumb_zoom, dpi, fig_transparent):
    """Lay out each panel as a 2x2 grid of (PC{2k-1} vs PC{2k}) subplots, stacked vertically."""
    n_pairs = n_components // 2
    pair_cols = 2
    pair_rows = int(np.ceil(n_pairs / pair_cols))

    n_panels = len(panels)
    total_rows = pair_rows * n_panels
    fig, axes = plt.subplots(
        total_rows, pair_cols,
        figsize=(6 * pair_cols, 5 * total_rows),
        dpi=dpi,
    )
    axes = np.atleast_2d(axes)

    for p_idx, panel in enumerate(panels):
        z_scores = panel["z_scores"]
        thumbnails = panel["thumbnails"]
        evr_pct = panel["explained_variance_ratio"] * 100

        for pair_idx in range(n_pairs):
            pc_i = 2 * pair_idx
            pc_j = 2 * pair_idx + 1
            row = p_idx * pair_rows + pair_idx // pair_cols
            col = pair_idx % pair_cols
            ax = axes[row][col]

            x = z_scores[:, pc_i]
            y = z_scores[:, pc_j]
            for xi, yi, thumb in zip(x, y, thumbnails):
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(np.array(thumb), zoom=thumb_zoom),
                    (xi, yi),
                    frameon=False,
                    pad=0.0,
                )
                ax.add_artist(imagebox)

            x_pad = (x.max() - x.min()) * 0.1 or 1.0
            y_pad = (y.max() - y.min()) * 0.1 or 1.0
            ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
            ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
            ax.set_title(f"PC{pc_i + 1} vs. PC{pc_j + 1}", fontsize=11)
            ax.set_xlabel(f"PC{pc_i + 1} ({evr_pct[pc_i]:.0f}%)", fontsize=10)
            ax.set_ylabel(f"PC{pc_j + 1} ({evr_pct[pc_j]:.0f}%)", fontsize=10)

        # Panel letter, positioned above/left of the panel's first (top-left) subplot.
        top_left_ax = axes[p_idx * pair_rows][0]
        top_left_ax.text(
            -0.28, 1.18, panel["label"], transform=top_left_ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15, transparent=fig_transparent)
    plt.close(fig)
    print(f"\nSaved figure to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Plot PC1..PCn axes of colour pattern trait space with specimen thumbnails "
                     "(recreates the manuscript's 'PC axes of avian colour pattern trait space' figure)."
    )
    parser.add_argument("--embeddings_csv", type=str, required=True,
                        help="Path to the embeddings CSV for panel A (first column: filename)")
    parser.add_argument("--embeddings_csv_b", type=str, default=None,
                        help="Optional path to a second embeddings CSV, plotted as panel B "
                             "(e.g. patch vs. whole-plumage embeddings) stacked below panel A")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of display images (e.g. lateral/side-view images) used as thumbnails for panel A")
    parser.add_argument("--image_dir_b", type=str, default=None,
                        help="Display image directory for panel B (defaults to --image_dir)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the figure and variance-explained/PCA-score CSVs")
    parser.add_argument("--label_a", type=str, default="A", help="Label for panel A (default: 'A')")
    parser.add_argument("--label_b", type=str, default="B", help="Label for panel B (default: 'B')")
    parser.add_argument("--n_components", type=int, default=8,
                        help="Number of PCs to compute/plot; must be even (default: 8)")
    parser.add_argument("--max_images", type=int, default=800,
                        help="Max number of specimens to render as thumbnails per panel (variance report always uses all specimens; default: 800)")
    parser.add_argument("--thumb_size", type=int, default=40,
                        help="Max thumbnail pixel dimension before placement (default: 40)")
    parser.add_argument("--thumb_zoom", type=float, default=1.0,
                        help="Zoom factor applied to thumbnails when placed on the plot (default: 1.0)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    parser.add_argument("--transparent_bg", action="store_true", default=True,
                        help="Use the image's alpha/mask channel for transparent thumbnails (default: True)")
    parser.add_argument("--no_transparent_bg", dest="transparent_bg", action="store_false",
                        help="Use a solid background for thumbnails instead of transparency")
    parser.add_argument("--fig_transparent", action="store_true", default=False,
                        help="Save the figure with a transparent canvas background")
    parser.add_argument("--index_workers", type=int, default=-1,
                        help="Passed to the image indexer: 0 = single-threaded, -1 = auto-parallel (default: -1)")
    return parser.parse_args()


def main(args):
    if args.n_components % 2 != 0:
        raise ValueError(f"--n_components must be even (got {args.n_components})")

    os.makedirs(args.output_dir, exist_ok=True)

    panels = [
        build_panel(
            args.embeddings_csv, args.image_dir, args.label_a, args.output_dir,
            args.n_components, args.max_images, args.random_state,
            args.thumb_size, args.transparent_bg, args.index_workers,
        )
    ]

    if args.embeddings_csv_b:
        panels.append(
            build_panel(
                args.embeddings_csv_b, args.image_dir_b or args.image_dir, args.label_b, args.output_dir,
                args.n_components, args.max_images, args.random_state,
                args.thumb_size, args.transparent_bg, args.index_workers,
            )
        )

    output_path = os.path.join(args.output_dir, "pca_pattern_space.png")
    plot_panels(panels, output_path, args.n_components, args.thumb_zoom, args.dpi, args.fig_transparent)


if __name__ == "__main__":
    main(get_args())
