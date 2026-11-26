"""Rank species by sexual dichromatism estimated from embedding distances.

Sanity-check for the embedding space: species that are known to be strongly
sexually dichromatic (e.g. many passerines, waterfowl, birds of paradise)
should show a bigger male vs. female embedding gap than species that are
sexually monomorphic. This script quantifies that gap per species and
returns the top-N most dichromatic species according to the embeddings.

For each species with both male and female specimens, three complementary
scores are computed:
  - centroid_distance:     distance between the mean male embedding and the
                            mean female embedding (the primary score).
  - mean_pairwise_distance: average distance over every (male, female) pair,
                            more robust to a skewed within-sex distribution
                            than the centroid alone.
  - silhouette_score:       how well male vs. female labels separate that
                            species' own embeddings (same idea as
                            eval_silhouette_variance.py), NaN if it can't be
                            computed (e.g. too few samples).

Filenames are expected to follow the convention used across this project:
``Genus_species_id_SEX_...`` (e.g. ``Anthochaera_chrysoptera_1_M_Belly.png``).
Specimens without a parseable sex (format 2 filenames, or any label other
than M/F) are ignored for this analysis.

Example
-------
    python species_sexual_dichromatism.py \\
        --embeddings rgb_embeddings.csv \\
        --top_n 20 \\
        --output_dir results/dichromatism
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler


# ---------------------------------------------------------------------------
# Loading and metadata parsing
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


def extract_species_sex(filename):
    """Extract (species, sex) from a filename, matching the convention in
    eval_silhouette_variance.py: ``Genus_species_id_SEX_...``. Returns
    sex=None when it can't be determined (e.g. the format-2 naming scheme)."""
    base_name = os.path.basename(str(filename))
    parts = base_name.split("_")
    if len(parts) >= 4 and parts[3] in ("M", "F", "U"):
        return f"{parts[0]}_{parts[1]}", parts[3]
    if len(parts) > 2:
        return "_".join(parts[:-2]), None
    return base_name, None


# ---------------------------------------------------------------------------
# Dichromatism scoring
# ---------------------------------------------------------------------------

def _pairwise_distances(a, b, metric):
    if metric == "cosine":
        return cdist(normalize(a), normalize(b), metric="cosine")
    return cdist(a, b, metric="euclidean")


def compute_species_dichromatism(embeddings, species, sex, distance_metric="cosine",
                                  min_samples_per_sex=1):
    """Compute per-species male/female embedding-distance scores.

    Returns a DataFrame with one row per species that has at least
    ``min_samples_per_sex`` male and female specimens, sorted by species name.
    """
    species = np.asarray(species)
    sex = np.asarray(sex)

    rows = []
    for sp in sorted(set(species)):
        sp_mask = species == sp
        male_mask = sp_mask & (sex == "M")
        female_mask = sp_mask & (sex == "F")
        n_male, n_female = int(male_mask.sum()), int(female_mask.sum())

        if n_male < min_samples_per_sex or n_female < min_samples_per_sex:
            continue

        emb_m = embeddings[male_mask]
        emb_f = embeddings[female_mask]

        centroid_m = emb_m.mean(axis=0, keepdims=True)
        centroid_f = emb_f.mean(axis=0, keepdims=True)
        centroid_distance = float(_pairwise_distances(centroid_m, centroid_f, distance_metric)[0, 0])

        pair_distances = _pairwise_distances(emb_m, emb_f, distance_metric)
        mean_pairwise_distance = float(pair_distances.mean())

        silhouette = np.nan
        n_total = n_male + n_female
        if n_total >= 3 and n_male < n_total and n_female < n_total:
            try:
                labels = np.array(["M"] * n_male + ["F"] * n_female)
                silhouette = float(
                    silhouette_score(np.vstack([emb_m, emb_f]), labels, metric=distance_metric)
                )
            except ValueError:
                pass

        rows.append({
            "species": sp,
            "n_male": n_male,
            "n_female": n_female,
            "centroid_distance": centroid_distance,
            "mean_pairwise_distance": mean_pairwise_distance,
            "silhouette_score": silhouette,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_top_species(ranked_df, rank_by, output_path):
    import matplotlib.pyplot as plt

    plot_df = ranked_df.iloc[::-1]  # largest at top of the horizontal bar chart
    fig_h = max(4, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(plot_df["species"], plot_df[rank_by], color="#8e44ad")
    ax.set_xlabel(rank_by.replace("_", " "))
    ax.set_title(f"Top {len(plot_df)} species by sexual dichromatism ({rank_by.replace('_', ' ')})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Image lookup (only used when --image_dir is given)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

_SEX_COLOR = {"M": "#1f77b4", "F": "#d62728"}  # matches conventions used elsewhere in this repo


def _stem(name):
    s = str(name).strip().replace("\\", "/")
    return os.path.splitext(os.path.basename(s))[0]


def build_image_index(image_dir):
    """Recursively index *image_dir* into a {filename_stem: path} dict (first match wins)."""
    index = {}
    for dirpath, _, fnames in os.walk(image_dir):
        for fname in fnames:
            stem, ext = os.path.splitext(fname)
            if ext.lower() in _IMAGE_EXTENSIONS:
                index.setdefault(stem, os.path.join(dirpath, fname))
    return index


def _prefix_match(stem, index):
    """Bidirectional prefix fallback, mirroring the matching used in
    visualisations/plot_embeddings_multispectral.py and eval/mantel_embedding_correlation.py
    (handles e.g. CSV stem ``..._Belly`` vs on-disk ``..._Belly_Vis_G167935``)."""
    if stem in index:
        return index[stem]
    candidates = [k for k in index if k.startswith(stem + "_")] + \
                 [k for k in index if stem.startswith(k + "_")]
    if not candidates:
        return None
    candidates.sort()
    return index[candidates[0]]


def load_rgb_thumbnail(path, size):
    """Load an image as a PIL thumbnail, preserving any existing alpha channel so the
    background stays transparent on the plot instead of being composited onto white."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                thumb = im.convert("RGBA")
            else:
                thumb = im.convert("RGB")
            thumb = thumb.copy()
            thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
            return thumb
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def _load_thumbnail_worker(args):
    path, size = args
    return load_rgb_thumbnail(path, size)


def add_sex_border(img, sex, border=4):
    """Pad *img* with a solid border coloured by sex (blue=M, red=F) so clusters are
    identifiable by eye once thumbnails are placed on the projection."""
    from PIL import ImageOps
    color = _SEX_COLOR.get(sex, "#7f7f7f")
    return ImageOps.expand(img, border=border, fill=color)


def reduce_dimensions(embeddings, method, random_state=42):
    """Project *embeddings* to 2D with UMAP, t-SNE, or PCA (after z-scoring)."""
    scaled = StandardScaler().fit_transform(embeddings)
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=min(15, max(2, len(scaled) - 1)), min_dist=0.1,
                            random_state=random_state)
        return reducer.fit_transform(scaled)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, (len(scaled) - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(scaled)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=random_state).fit_transform(scaled)
    raise ValueError(f"Unknown dim-reduction method: {method}")


def sample_background_indices(n_total, exclude_idx, background_size, random_state=42):
    """Randomly sample up to *background_size* indices from the full dataset (excluding
    *exclude_idx*) to serve as whole-embedding-space context around the highlighted species."""
    rng = np.random.default_rng(random_state)
    pool = np.setdiff1d(np.arange(n_total), exclude_idx)
    budget = max(0, background_size - len(exclude_idx))
    if budget == 0 or len(pool) == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(pool, size=min(budget, len(pool)), replace=False)


def _sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def load_thumbnails_for_filenames(filenames_arr, image_index, thumb_size):
    """Resolve and load a thumbnail per filename. Returns (valid_idx, thumbnails) where
    *valid_idx* indexes into *filenames_arr* for entries whose image was found and loaded."""
    paths = [image_index.get(_stem(f)) or _prefix_match(_stem(f), image_index) for f in filenames_arr]
    found_idx = [i for i, p in enumerate(paths) if p is not None]
    n_missing = len(filenames_arr) - len(found_idx)
    if n_missing:
        print(f"  Warning: {n_missing} specimen(s) had no matching image under --image_dir")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        loaded = list(executor.map(_load_thumbnail_worker, [(paths[i], thumb_size) for i in found_idx]))

    valid_idx = [i for i, t in zip(found_idx, loaded) if t is not None]
    thumbnails = [t for t in loaded if t is not None]
    return valid_idx, thumbnails


def plot_species_highlight(species_name, coords, filenames_arr, species_arr, sex_arr, thumbnails,
                            output_path, method, border=4, max_per_group=None, random_state=42):
    """Plot every specimen in *coords* using its own thumbnail (the whole-embedding-space
    projection); only *species_name*'s male/female thumbnails get a sex-coloured border
    (blue=M, red=F) so they stand out against the rest, which is drawn with plain thumbnails."""
    import matplotlib.pyplot as plt
    import matplotlib.offsetbox as offsetbox

    species_idx = np.where(species_arr == species_name)[0]
    if max_per_group:
        rng = np.random.default_rng(random_state)
        keep = []
        for s in ("M", "F"):
            group_idx = species_idx[sex_arr[species_idx] == s]
            if len(group_idx) > max_per_group:
                group_idx = rng.choice(group_idx, max_per_group, replace=False)
            keep.extend(group_idx.tolist())
        species_idx = np.array(sorted(keep))

    if len(species_idx) == 0:
        print(f"WARNING: no images available for {species_name}; skipping its highlight plot.")
        return

    species_mask = np.zeros(len(species_arr), dtype=bool)
    species_mask[species_idx] = True
    other_idx = np.where(~species_mask)[0]

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in other_idx:
        x, y = coords[i]
        ax.add_artist(offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(np.asarray(thumbnails[i])), (x, y), frameon=False, pad=0.0, zorder=1,
        ))
    for i in species_idx:
        x, y = coords[i]
        bordered = add_sex_border(thumbnails[i], sex_arr[i], border=border)
        ax.add_artist(offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(np.asarray(bordered)), (x, y), frameon=False, pad=0.0, zorder=2,
        ))

    ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"{species_name} — male (blue border) vs. female (red border)\n"
                 f"in whole-embedding-space {method.upper()}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Rank species by male/female embedding distance (sexual dichromatism proxy).",
    )
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to the embeddings CSV (first column = filename).")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of most dichromatic species to report (default: 20).")
    parser.add_argument("--rank_by", type=str, default="centroid_distance",
                        choices=["centroid_distance", "mean_pairwise_distance", "silhouette_score"],
                        help="Score used to rank species (default: centroid_distance).")
    parser.add_argument("--distance_metric", type=str, default="cosine",
                        choices=["cosine", "euclidean"],
                        help="Distance metric used for all scores (default: cosine).")
    parser.add_argument("--min_samples_per_sex", type=int, default=1,
                        help="Minimum male AND female specimens required for a species to be "
                             "considered (default: 1).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the full ranking CSV and top-N plot. "
                             "If omitted, results are only printed.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Save a bar chart of the top-N species to output_dir.")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Optional directory of RGB specimen images. If given, also saves one "
                             "2D projection (UMAP/t-SNE/PCA) per top-N species: the whole embedding "
                             "space (or a --image_background_size sample of it) is shown using each "
                             "specimen's own plain thumbnail, with that species' own male/female "
                             "thumbnails additionally given a blue/red border so they stand out. "
                             "Requires --output_dir.")
    parser.add_argument("--image_dim_reduction", type=str, default="umap",
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction method for the highlight plots (default: umap).")
    parser.add_argument("--image_background_size", type=int, default=500,
                        help="Number of specimens (sampled from the whole dataset, plus all top-N "
                             "species specimens) used to fit the shared background projection "
                             "(default: 300).")
    parser.add_argument("--image_thumb_size", type=int, default=60,
                        help="Thumbnail size in pixels for the highlighted species' images (default: 60).")
    parser.add_argument("--image_max_per_group", type=int, default=None,
                        help="Optional cap on thumbnails per sex shown for the highlighted species, "
                             "to keep each plot readable (default: no cap).")
    return parser.parse_args()


def main(args):
    print(f"Loading embeddings from {args.embeddings} ...")
    embeddings, filenames = load_embeddings_from_csv(args.embeddings)
    print(f"  -> {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")

    species, sex = zip(*(extract_species_sex(f) for f in filenames))
    species, sex = np.array(species), np.array(sex)
    n_male = sum(1 for s in sex if s == "M")
    n_female = sum(1 for s in sex if s == "F")
    n_unknown = len(sex) - n_male - n_female
    print(f"  Parsed sex labels: {n_male} male, {n_female} female, {n_unknown} unknown/other")

    ranking = compute_species_dichromatism(
        embeddings, species, sex,
        distance_metric=args.distance_metric,
        min_samples_per_sex=args.min_samples_per_sex,
    )
    print(f"  Species with >= {args.min_samples_per_sex} male and female specimens: {len(ranking)}")

    if ranking.empty:
        print("ERROR: no species had both male and female specimens; nothing to rank.")
        return

    ranking = ranking.sort_values(args.rank_by, ascending=False).reset_index(drop=True)
    top = ranking.head(args.top_n)

    print(f"\nTop {len(top)} species by {args.rank_by} ({args.distance_metric} distance):")
    print(top.to_string(index=False))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        full_path = os.path.join(args.output_dir, "species_dichromatism_full.csv")
        ranking.to_csv(full_path, index=False)
        print(f"\nFull ranking saved to {full_path}")

        top_path = os.path.join(args.output_dir, "species_dichromatism_top.csv")
        top.to_csv(top_path, index=False)
        print(f"Top-{args.top_n} ranking saved to {top_path}")

        if args.plot:
            plot_path = os.path.join(args.output_dir, "species_dichromatism_top.png")
            plot_top_species(top, args.rank_by, plot_path)

    if args.image_dir:
        if not args.output_dir:
            print("\nWARNING: --image_dir given without --output_dir; skipping image highlight plots.")
        else:
            top_species_order = top["species"].tolist()
            mandatory_mask = np.isin(species, top_species_order) & np.isin(sex, ["M", "F"])
            mandatory_idx = np.where(mandatory_mask)[0]
            background_idx = sample_background_indices(
                len(embeddings), mandatory_idx, args.image_background_size,
            )
            combined_idx = np.concatenate([mandatory_idx, background_idx])

            print(f"\nFitting shared {args.image_dim_reduction.upper()} projection on {len(combined_idx)} "
                  f"embeddings ({len(mandatory_idx)} top-species specimens + "
                  f"{len(background_idx)} background) ...")
            coords = reduce_dimensions(embeddings[combined_idx], args.image_dim_reduction)
            combined_filenames = filenames[combined_idx]
            combined_species = species[combined_idx]
            combined_sex = sex[combined_idx]

            print(f"Indexing images under {args.image_dir} ...")
            image_index = build_image_index(args.image_dir)
            print(f"  Indexed {len(image_index)} image files")

            print(f"Loading {len(combined_filenames)} thumbnails (shared across all species plots) ...")
            valid_idx, thumbnails = load_thumbnails_for_filenames(
                combined_filenames, image_index, args.image_thumb_size
            )
            if not thumbnails:
                print("ERROR: no thumbnails could be loaded; skipping image highlight plots.")
            else:
                coords = coords[valid_idx]
                combined_filenames = combined_filenames[valid_idx]
                combined_species = combined_species[valid_idx]
                combined_sex = combined_sex[valid_idx]

                for rank, sp in enumerate(top_species_order, start=1):
                    image_path = os.path.join(
                        args.output_dir,
                        f"species_dichromatism_{args.image_dim_reduction}_{rank:02d}_{_sanitize_filename(sp)}.png",
                    )
                    plot_species_highlight(
                        sp, coords, combined_filenames, combined_species, combined_sex, thumbnails,
                        image_path, args.image_dim_reduction, max_per_group=args.image_max_per_group,
                    )

    print("\nDone.")


if __name__ == "__main__":
    main(get_args())
