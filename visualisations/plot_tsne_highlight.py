#!/usr/bin/env python3
"""
Highlight the first N unique species_sex classes (in file order) on t-SNE / UMAP plots.
All samples of each chosen class are coloured; everything else is grey.

Usage:
    python plot_tsne_highlight.py \
        --embeddings_csv path/to/embeddings.csv \
        --output_dir     path/to/output/ \
        --num_references 3
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

sys.path.insert(0, os.path.dirname(__file__))
from plot_embeddings_multispectral import load_embeddings_from_csv


def parse_class(fname):
    """Return genus_species (e.g. Chloropsis_jerdoni).
    Stops before the first single-letter sex token (M/F) or numeric individual ID."""
    base = os.path.splitext(os.path.basename(fname))[0]
    result = []
    for p in base.split("_"):
        if p.isdigit() or p in ("M", "F"):
            break
        result.append(p)
    return "_".join(result)


def pick_references(classes, n_refs):
    """Return indices of the first occurrence of the first N unique classes."""
    refs, seen = [], set()
    for i, cls in enumerate(classes):
        if cls not in seen:
            seen.add(cls)
            refs.append(i)
        if len(refs) == n_refs:
            break
    return refs


def parse_sex(fname):
    """Extract M or F from the filename, or '' if not found."""
    for part in os.path.splitext(os.path.basename(fname))[0].split("_"):
        if part in ("M", "F"):
            return part
    return ""


def _draw_scatter(ax, emb2d, classes, filenames, refs, ref_colours):
    """Shared drawing logic used by both the full and zoomed plots."""
    highlighted = np.zeros(len(filenames), dtype=bool)
    for ref in refs:
        highlighted |= (classes == classes[ref])

    ax.scatter(emb2d[~highlighted, 0], emb2d[~highlighted, 1],
               c="#cccccc", s=18, alpha=0.5, linewidths=0,
               zorder=1, label="Other samples")

    for r_i, ref in enumerate(refs):
        ref_class = classes[ref]
        colour    = ref_colours[r_i]
        same      = np.where(classes == ref_class)[0]
        ax.scatter(emb2d[same, 0], emb2d[same, 1],
                   color=colour, s=18, alpha=0.35, linewidths=0,
                   zorder=2 + r_i, label=ref_class)
        for i in same:
            sex = parse_sex(filenames[i])
            if sex:
                ax.text(emb2d[i, 0], emb2d[i, 1], sex,
                        fontsize=5, ha="center", va="bottom",
                        color=colour, alpha=0.8, zorder=20 + r_i)


def make_plot(emb2d, classes, filenames, refs, ref_colours, method, output_dir):
    """Save the full plot and one cropped zoom per reference (same figure, different limits)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_scatter(ax, emb2d, classes, filenames, refs, ref_colours)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0.9, fontsize=10, loc="best", markerscale=1.1)

    label = method.upper()
    ax.set_xlabel(f"{label} 1", fontsize=12)
    ax.set_ylabel(f"{label} 2", fontsize=12)
    ax.set_title(f"{label} — highlighted classes", fontsize=13)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{method}_highlight.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_csv",  required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--num_references",  type=int, default=1,
                   help="Number of classes to highlight (default: 1)")
    p.add_argument("--random_state",    type=int, default=42)
    p.add_argument("--method",          choices=["tsne", "umap", "both"], default="both")
    args = p.parse_args()

    embeddings, filenames = load_embeddings_from_csv(args.embeddings_csv)
    classes = np.array([parse_class(f) for f in filenames])

    scaler = StandardScaler()
    emb = scaler.fit_transform(embeddings)

    refs        = pick_references(classes, args.num_references)
    palette     = plt.get_cmap("tab10").colors
    ref_colours = [palette[i % len(palette)] for i in range(len(refs))]

    for r_i, ref in enumerate(refs):
        print(f"Reference {r_i+1}: {filenames[ref]}  (class: {classes[ref]})")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method in ("tsne", "both"):
        print("\nRunning t-SNE…")
        emb2d = TSNE(n_components=2, perplexity=30,
                     random_state=args.random_state).fit_transform(emb)
        make_plot(emb2d, classes, filenames, refs, ref_colours, "tsne", args.output_dir)

    if args.method in ("umap", "both"):
        print("Running UMAP…")
        emb2d = umap.UMAP(n_neighbors=15, min_dist=0.1,
                          random_state=args.random_state).fit_transform(emb)
        make_plot(emb2d, classes, filenames, refs, ref_colours, "umap", args.output_dir)


if __name__ == "__main__":
    main()
