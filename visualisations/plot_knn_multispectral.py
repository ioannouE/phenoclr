#!/usr/bin/env python3
"""
Plot KNN examples from an embeddings CSV using multispectral bird images.

Each row shows a randomly selected query image followed by its N nearest
neighbors in embedding space.  Bird backgrounds are transparent (the mask
band is used as the alpha channel).

Usage:
    python plot_knn_multispectral.py \
        --embeddings_csv path/to/embeddings.csv \
        --data_dir path/to/images/ \
        --output_path path/to/knn_examples.png \
        --num_examples 6 \
        --n_neighbors 5
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Reuse shared utilities from the sibling script
sys.path.insert(0, os.path.dirname(__file__))
from plot_embeddings_multispectral import (
    load_embeddings_from_csv,
    load_multispectral_image_rgb,
    find_multispectral_image,
)

from PIL import Image


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_thumbnail(args):
    filepath, thumb_size, invert_mask, norm_foreground_only = args
    if not os.path.exists(filepath):
        return None
    array = load_multispectral_image_rgb(
        filepath,
        transparent_bg=True,
        invert_mask=invert_mask,
        norm_foreground_only=norm_foreground_only,
    )
    if array is None:
        return None
    try:
        mode = "RGBA" if array.shape[2] == 4 else "RGB"
        img = Image.fromarray(array, mode=mode)
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Thumbnail error for {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_knn_examples(
    embeddings,
    filenames,
    data_dir,
    output_path,
    n_neighbors=5,
    num_examples=6,
    thumb_size=150,
    random_state=42,
    invert_mask=False,
    norm_foreground_only=True,
    transparent_fig=False,
):
    """Plot KNN examples.

    Creates a grid with *num_examples* rows.  The first column of each row is a
    randomly selected query image; the remaining *n_neighbors* columns show its
    nearest neighbors in embedding space, labelled with their L2 distance.
    Bird backgrounds are transparent (alpha channel derived from the mask band).

    Parameters
    ----------
    embeddings:           (N, D) numpy array
    filenames:            list of N filename strings
    data_dir:             root directory containing the .tif images
    output_path:          where to save the resulting PNG
    n_neighbors:          number of neighbors to display per row (not counting query)
    num_examples:         number of rows in the visualisation
    thumb_size:           max pixel size for each thumbnail
    random_state:         seed for reproducibility
    invert_mask:          invert mask semantics (0 → foreground)
    norm_foreground_only: normalise each band using foreground pixels only
    transparent_fig:      save figure canvas with transparent background
    """
    np.random.seed(random_state)

    # ---- Nearest-neighbour search ----------------------------------------
    # Request n_neighbors+1 so index 0 is the query itself and [1:] are
    # the true neighbors.
    print("Computing nearest neighbours…")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Pick random query rows (avoiding duplicates)
    max_samples = min(num_examples, len(indices))
    samples_idx = np.random.choice(len(indices), size=max_samples, replace=False)

    # ---- Resolve image paths --------------------------------------------
    needed_fnames = set()
    for idx in samples_idx:
        for neighbor_idx in indices[idx]:
            needed_fnames.add(filenames[neighbor_idx])

    print("Resolving image paths…")
    path_map = {}
    for fname in tqdm(needed_fnames, desc="Finding images"):
        full_path = find_multispectral_image(data_dir, fname)
        if full_path:
            path_map[fname] = full_path

    # ---- Load thumbnails in parallel ------------------------------------
    unique_fnames = list(path_map.keys())
    load_args = [
        (path_map[f], thumb_size, invert_mask, norm_foreground_only)
        for f in unique_fnames
    ]

    print("Loading thumbnails…")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        loaded = list(
            tqdm(
                executor.map(_load_thumbnail, load_args),
                total=len(load_args),
                desc="Loading images",
            )
        )
    thumb_map = {
        fname: thumb
        for fname, thumb in zip(unique_fnames, loaded)
        if thumb is not None
    }

    # ---- Build figure ---------------------------------------------------
    # n_cols: 1 query + n_neighbors neighbors
    n_cols = n_neighbors + 1
    cell_w = 2.2   # inches per column
    cell_h = 1.8   # shorter rows to reduce vertical space

    fig, axs = plt.subplots(
        max_samples,
        n_cols,
        figsize=(n_cols * cell_w, max_samples * cell_h),
    )

    # Ensure axs is always 2-D (row, col)
    if max_samples == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    fig.patch.set_facecolor("white")
    if transparent_fig:
        fig.patch.set_alpha(0.0)

    for row, idx in enumerate(samples_idx):
        # indices[idx][0] is the query itself; [1:] are genuine neighbors
        col_indices = indices[idx]      # length n_neighbors+1
        col_distances = distances[idx]  # length n_neighbors+1

        for col in range(n_cols):
            ax = axs[row, col]
            ax.set_facecolor("white")
            ax.axis("off")

            neighbor_idx = col_indices[col]
            dist = col_distances[col]
            fname = filenames[neighbor_idx]
            thumb = thumb_map.get(fname)

            if thumb is not None:
                ax.imshow(np.array(thumb))

            if col == 0:
                ax.set_title("Query", fontsize=20, pad=3)
            else:
                ax.set_title(f"d={dist:.3f}", fontsize=20, pad=3)

    plt.tight_layout(pad=0.02, h_pad=0.01)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving visualisation → {output_path}")
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=transparent_fig,
    )
    plt.close()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot KNN examples from a SimCLR embeddings CSV using "
            "multispectral bird images with transparent backgrounds."
        )
    )
    parser.add_argument(
        "--embeddings_csv",
        type=str,
        required=True,
        help="Path to CSV file with embeddings (first column must be filenames).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the multispectral .tif images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=6,
        help="Number of KNN rows to display (default: 6). Use 1 for a single row.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors shown per row, excluding the query (default: 5).",
    )
    parser.add_argument(
        "--thumb_size",
        type=int,
        default=150,
        help="Max thumbnail size in pixels (default: 150).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--invert_mask",
        action="store_true",
        default=False,
        help="Invert mask semantics: treat pixel value 0 as foreground.",
    )
    parser.add_argument(
        "--norm_foreground_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise each RGB band using only foreground pixels (default: True).",
    )
    parser.add_argument(
        "--transparent_fig",
        action="store_true",
        default=False,
        help="Save the figure canvas with a transparent background.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Loading embeddings from {args.embeddings_csv}…")
    embeddings, filenames = load_embeddings_from_csv(args.embeddings_csv)
    print(f"Loaded {len(embeddings)} embeddings.")

    if filenames is None:
        print(
            "Error: no filenames found in CSV. "
            "Ensure the first column contains image filenames."
        )
        return

    plot_knn_examples(
        embeddings=embeddings,
        filenames=filenames,
        data_dir=args.data_dir,
        output_path=args.output_path,
        n_neighbors=args.n_neighbors,
        num_examples=args.num_examples,
        thumb_size=args.thumb_size,
        random_state=args.random_state,
        invert_mask=args.invert_mask,
        norm_foreground_only=args.norm_foreground_only,
        transparent_fig=args.transparent_fig,
    )


if __name__ == "__main__":
    main()
