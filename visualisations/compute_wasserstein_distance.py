"""Compute Wasserstein distance between two sets of embeddings in latent space.

Inspired by the MSDM paper (arXiv:2510.09121) which uses Wasserstein distances
between latent space embeddings to evaluate how closely two embedding distributions
align (e.g. real vs. synthetic, or two different models/modalities).

Supports:
  - Sliced Wasserstein Distance (SWD): projects embeddings onto many random 1D
    directions and averages the 1D Wasserstein distances — the standard practical
    estimator for high-dimensional distributions.
  - Per-dimension Wasserstein: independent 1D Wasserstein per embedding dimension,
    then averaged — fast but ignores cross-dimensional correlations.
  - Per-group analysis: when a ``--group_column`` is present in both CSVs, computes
    pairwise Wasserstein distances between all group pairs across the two sets and
    produces a heatmap (analogous to Figure 3 of the MSDM paper).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as wd_1d
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings_from_csv(embeddings_path):
    """Load embeddings from CSV file.

    Returns (embeddings, filenames, df) where *filenames* may be ``None`` and
    *df* is the raw DataFrame (kept for potential group column look-ups).
    """
    df = pd.read_csv(embeddings_path)
    if df.columns[0].lower() in ("filename", "file", "name", "image"):
        filenames = df.iloc[:, 0].values
        embeddings = df.iloc[:, 1:].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    else:
        filenames = None
        embeddings = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    return embeddings, filenames, df


# ---------------------------------------------------------------------------
# Wasserstein distance estimators
# ---------------------------------------------------------------------------

def per_dimension_wasserstein(emb_a: np.ndarray, emb_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean Wasserstein-1 distance computed independently per dimension."""
    assert emb_a.shape[1] == emb_b.shape[1], "Embedding dimensions must match"
    D = emb_a.shape[1]
    dists = np.array([wd_1d(emb_a[:, d], emb_b[:, d]) for d in range(D)])
    return float(np.mean(dists)), dists


def sliced_wasserstein_distance(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_projections: int = 1000,
    seed: int = 42,
) -> float:
    """Sliced Wasserstein Distance (SWD).

    Projects both distributions onto ``n_projections`` random unit vectors drawn
    uniformly on the unit sphere and averages the resulting 1D Wasserstein-1
    distances.
    """
    assert emb_a.shape[1] == emb_b.shape[1], "Embedding dimensions must match"
    D = emb_a.shape[1]
    rng = np.random.default_rng(seed)

    directions = rng.standard_normal((n_projections, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    proj_a = emb_a @ directions.T  # (N_a, n_projections)
    proj_b = emb_b @ directions.T  # (N_b, n_projections)

    dists = np.array([wd_1d(proj_a[:, i], proj_b[:, i]) for i in range(n_projections)])
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Per-group pairwise analysis + heatmap
# ---------------------------------------------------------------------------

def _extract_metadata_from_filename(filename):
    """Extract species from filename, matching the convention in eval_silhouette_variance.py.

    Supported formats:
      Format 1: Genus_species_id_SEX...  (e.g. 'Anthochaera_chrysoptera_1_M_12345')
      Format 2: SpeciesName_ID1_ID2      (e.g. 'Abraxas_fulvobasalis_1D_040917')
    """
    base_name = os.path.basename(str(filename))
    parts = base_name.split("_")

    if len(parts) >= 4 and parts[3] in ("M", "F", "U"):
        return f"{parts[0]}_{parts[1]}"

    if len(parts) > 2:
        return "_".join(parts[:-2])

    return base_name


def _extract_groups(df, group_column, filenames=None):
    """Return a Series of group labels aligned to the DataFrame rows.

    First tries to find an explicit column in *df*.  If the column is not found
    and *group_column* is ``"species"`` (case-insensitive), species labels are
    parsed from *filenames* using the naming convention used across this project
    (``Genus_species_id_SEX...``).
    """
    col_lower = {c.lower(): c for c in df.columns}
    key = col_lower.get(group_column.lower())
    if key is not None:
        return df[key]

    if group_column.lower() == "species" and filenames is not None:
        labels = [_extract_metadata_from_filename(f) for f in filenames]
        return pd.Series(labels, index=df.index)

    return None


def compute_pairwise_group_wasserstein(
    emb_a, groups_a, emb_b, groups_b, n_projections=1000, seed=42,
):
    """Compute SWD between every (group_i from A, group_j from B) pair.

    Returns a DataFrame with groups_a labels as index, groups_b labels as columns.
    """
    unique_a = sorted(groups_a.unique())
    unique_b = sorted(groups_b.unique())
    matrix = np.full((len(unique_a), len(unique_b)), np.nan)

    for i, ga in enumerate(unique_a):
        mask_a = (groups_a == ga).values
        ea = emb_a[mask_a]
        if len(ea) < 2:
            continue
        for j, gb in enumerate(unique_b):
            mask_b = (groups_b == gb).values
            eb = emb_b[mask_b]
            if len(eb) < 2:
                continue
            matrix[i, j] = sliced_wasserstein_distance(ea, eb, n_projections=n_projections, seed=seed)

    return pd.DataFrame(matrix, index=unique_a, columns=unique_b)


def plot_pairwise_heatmap(dist_df, output_path, label_a="Embeddings A", label_b="Embeddings B"):
    """Plot a heatmap of pairwise Wasserstein distances styled after Figure 3 of
    the MSDM paper (arXiv:2510.09121): diverging RdBu_r colormap so that low
    distances (matching distributions on the diagonal) appear blue and high
    distances appear red, with cell values annotated."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    data = dist_df.values
    finite_vals = data[np.isfinite(data)]
    if len(finite_vals) == 0:
        print("WARNING: No finite values in the distance matrix — skipping heatmap.")
        return

    vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))
    vcenter = float(np.median(finite_vals))
    if vmin == vmax:
        vcenter = vmin
        vmax = vmin + 1.0

    n_rows, n_cols = data.shape
    cell_w, cell_h = 0.72, 0.56
    fig_w = max(6, n_cols * cell_w + 2.5)
    fig_h = max(4, n_rows * cell_h + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(dist_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(dist_df.index, fontsize=8)
    ax.set_xlabel(label_b, fontsize=10)
    ax.set_ylabel(label_a, fontsize=10)

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            luminance = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if luminance > 0.75 or luminance < 0.25 else "black"
            fontweight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight=fontweight)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Sliced Wasserstein Distance", fontsize=9)
    ax.set_title("Wasserstein distances in latent space embeddings", fontsize=11, pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# UMAP overlay visualisation
# ---------------------------------------------------------------------------

def plot_joint_umap(emb_a, emb_b, output_path, label_a="A", label_b="B", random_state=42):
    """Fit UMAP on the union of both embedding sets and colour by source."""
    import umap as umap_lib
    import matplotlib.pyplot as plt

    combined = np.vstack([emb_a, emb_b])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
    coords = reducer.fit_transform(combined_scaled)

    n_a = len(emb_a)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:n_a, 0], coords[:n_a, 1], s=8, alpha=0.5, label=label_a, c="#1f77b4")
    ax.scatter(coords[n_a:, 0], coords[n_a:, 1], s=8, alpha=0.5, label=label_b, c="#ff7f0e")
    ax.legend(fontsize=10)
    ax.set_title("Joint UMAP — two embedding sets")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Joint UMAP saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Compute Wasserstein distance between two sets of embeddings in latent space "
                    "(following the MSDM evaluation protocol, arXiv:2510.09121).",
    )
    parser.add_argument("--embeddings_a", type=str, required=True,
                        help="Path to the first embeddings CSV")
    parser.add_argument("--embeddings_b", type=str, required=True,
                        help="Path to the second embeddings CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (heatmap, UMAP overlay, CSV). "
                             "If not given, results are only printed to stdout.")
    parser.add_argument("--n_projections", type=int, default=1000,
                        help="Number of random projections for Sliced Wasserstein Distance")
    parser.add_argument("--group_column", type=str, default=None,
                        help="Column name present in both CSVs to compute per-group pairwise "
                             "Wasserstein distances (e.g. 'species', 'label', 'indication')")
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="Standardize (zero-mean, unit-variance) embeddings before computing distances")
    parser.add_argument("--plot_umap", action="store_true", default=False,
                        help="Plot a joint UMAP of both embedding sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for SWD projections and UMAP")
    parser.add_argument("--label_a", type=str, default=None,
                        help="Display label for the first embedding set (defaults to filename)")
    parser.add_argument("--label_b", type=str, default=None,
                        help="Display label for the second embedding set (defaults to filename)")
    return parser.parse_args()


def main(args):
    # --- Load ---------------------------------------------------------------
    print(f"Loading embeddings A from {args.embeddings_a} ...")
    emb_a, fnames_a, df_a = load_embeddings_from_csv(args.embeddings_a)
    print(f"  → {emb_a.shape[0]} samples, {emb_a.shape[1]} dimensions")

    print(f"Loading embeddings B from {args.embeddings_b} ...")
    emb_b, fnames_b, df_b = load_embeddings_from_csv(args.embeddings_b)
    print(f"  → {emb_b.shape[0]} samples, {emb_b.shape[1]} dimensions")

    if emb_a.shape[1] != emb_b.shape[1]:
        print(f"ERROR: Embedding dimensions do not match ({emb_a.shape[1]} vs {emb_b.shape[1]})")
        sys.exit(1)

    label_a = args.label_a or os.path.basename(args.embeddings_a)
    label_b = args.label_b or os.path.basename(args.embeddings_b)

    # --- Optional standardization -------------------------------------------
    if args.standardize:
        print("Standardizing embeddings (joint fit) ...")
        scaler = StandardScaler()
        combined = np.vstack([emb_a, emb_b])
        scaler.fit(combined)
        emb_a = scaler.transform(emb_a)
        emb_b = scaler.transform(emb_b)

    # --- Global distances ---------------------------------------------------
    print("\n--- Global Wasserstein distances ---")

    swd = sliced_wasserstein_distance(emb_a, emb_b, n_projections=args.n_projections, seed=args.seed)
    print(f"  Sliced Wasserstein Distance (SWD, {args.n_projections} projections): {swd:.4f}")

    pdw_mean, pdw_per_dim = per_dimension_wasserstein(emb_a, emb_b)
    print(f"  Per-dimension Wasserstein (mean ± std):  {pdw_mean:.4f} ± {np.std(pdw_per_dim):.4f}")
    print()

    # --- Per-group pairwise analysis ----------------------------------------
    dist_df = None
    if args.group_column:
        groups_a = _extract_groups(df_a, args.group_column, filenames=fnames_a)
        groups_b = _extract_groups(df_b, args.group_column, filenames=fnames_b)

        if groups_a is None:
            print(f"WARNING: group column '{args.group_column}' not found in embeddings A — skipping per-group analysis")
        elif groups_b is None:
            print(f"WARNING: group column '{args.group_column}' not found in embeddings B — skipping per-group analysis")
        else:
            print(f"Computing pairwise group Wasserstein (column='{args.group_column}') ...")
            print(f"  Groups in A: {sorted(groups_a.unique())}")
            print(f"  Groups in B: {sorted(groups_b.unique())}")
            dist_df = compute_pairwise_group_wasserstein(
                emb_a, groups_a, emb_b, groups_b,
                n_projections=args.n_projections, seed=args.seed,
            )
            print("\nPairwise SWD matrix:")
            print(dist_df.to_string())
            print()

            shared = set(groups_a.unique()) & set(groups_b.unique())
            if shared:
                diag_vals = [dist_df.loc[g, g] for g in shared if g in dist_df.index and g in dist_df.columns]
                diag_vals = [v for v in diag_vals if np.isfinite(v)]
                if diag_vals:
                    print(f"  Matching-group mean SWD: {np.mean(diag_vals):.4f} ± {np.std(diag_vals):.4f}")
                off_diag = dist_df.values.copy()
                for idx_i, g in enumerate(dist_df.index):
                    if g in dist_df.columns:
                        off_diag[idx_i, list(dist_df.columns).index(g)] = np.nan
                off_vals = off_diag[np.isfinite(off_diag)]
                if len(off_vals):
                    print(f"  Non-matching mean SWD:   {np.mean(off_vals):.4f} ± {np.std(off_vals):.4f}")
                print()

    # --- Outputs ------------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        summary = {
            "metric": ["sliced_wasserstein_distance", "per_dim_wasserstein_mean", "per_dim_wasserstein_std"],
            "value": [swd, pdw_mean, float(np.std(pdw_per_dim))],
        }
        summary_path = os.path.join(args.output_dir, "wasserstein_summary.csv")
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")

        if dist_df is not None:
            csv_path = os.path.join(args.output_dir, "wasserstein_pairwise_groups.csv")
            dist_df.to_csv(csv_path)
            print(f"Pairwise group matrix saved to {csv_path}")

            heatmap_path = os.path.join(args.output_dir, "wasserstein_pairwise_heatmap.png")
            plot_pairwise_heatmap(dist_df, heatmap_path, label_a=label_a, label_b=label_b)

        if args.plot_umap:
            umap_path = os.path.join(args.output_dir, "wasserstein_joint_umap.png")
            plot_joint_umap(emb_a, emb_b, umap_path, label_a=label_a, label_b=label_b,
                            random_state=args.seed)

    print("Done.")


if __name__ == "__main__":
    args = get_args()
    main(args)
