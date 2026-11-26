#!/usr/bin/env python3
"""Cross-modality KNN visualisation: RGB query → nearest neighbours in RGB+UV.

For each query specimen (from embedding set A, typically RGB), this plots the
top-K nearest neighbours retrieved from embedding set B (typically RGB+UV or
UV-only). The paired/true specimen is highlighted with a green border so you
can see at a glance whether modality fusion preserves identity.

Each row of the figure looks like:

    [ Query (A)  |  NN1 (B)  |  NN2 (B)  |  ...  |  NNK (B) ]

The neighbour that matches the query's paired specimen (by filename stem, with
the same prefix-fallback used in ``plot_embeddings_multispectral.py`` and
``paired_retrieval_test.py``) is annotated as ``MATCH`` and outlined in green.
The row's left-axis label shows the rank of the true paired target across the
full B set, so you can see whether the correct match was retrieved at all.

Selection of queries:
  --query_names <name> [<name> ...]   Specific A filenames to use as queries.
  --filter {all,success,failure}      Restrict random sampling to queries
                                      whose true match was/was not retrieved
                                      at rank 1 (default: all).
  --num_examples N                    Number of rows (default: 6).

Image loading mirrors ``plot_knn_multispectral.py``: bird backgrounds are made
transparent via the mask band (when present).
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

# Reuse shared utilities from sibling scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_retrieval_test import (
    load_embeddings_from_csv,
    build_pairs,
    _cosine_sim_matrix,
    _euclidean_dist_matrix,
    compute_ranks,
)
from plot_embeddings_multispectral import (
    build_multispectral_index,
    find_multispectral_image,
    load_multispectral_image_rgb,
)


# ---------------------------------------------------------------------------
# Thumbnail loading
# ---------------------------------------------------------------------------

def _load_thumbnail(args):
    filepath, thumb_size, invert_mask, norm_foreground_only = args
    if filepath is None or not os.path.exists(filepath):
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
# Selection of query specimens
# ---------------------------------------------------------------------------

def _select_query_pairs(
    names_a,
    pairs,
    ranks_for_filter,
    num_examples,
    filter_mode,
    query_names,
    random_state,
):
    """Return list of indices into the pair list (one entry per displayed row).

    Pair list is ``[(idx_a, idx_b), ...]``; this function picks which of those
    entries to plot, respecting --query_names, --filter, and --num_examples.
    """
    if query_names:
        wanted = {os.path.splitext(os.path.basename(str(q)))[0]: q for q in query_names}
        selected = []
        missing = list(wanted.keys())
        for pair_idx, (i, _) in enumerate(pairs):
            stem = os.path.splitext(os.path.basename(str(names_a[i])))[0]
            if stem in wanted:
                selected.append(pair_idx)
                if stem in missing:
                    missing.remove(stem)
        if missing:
            print(f"WARNING: {len(missing)} requested query name(s) were not found "
                  f"among paired A specimens: {missing[:5]}"
                  + (" ..." if len(missing) > 5 else ""))
        if not selected:
            print("ERROR: none of the requested --query_names matched a paired specimen.")
            sys.exit(1)
        if num_examples is not None:
            selected = selected[:num_examples]
        return selected

    rng = np.random.default_rng(random_state)
    candidate_pair_indices = np.arange(len(pairs))

    if filter_mode == "success":
        mask = ranks_for_filter == 1
        candidate_pair_indices = candidate_pair_indices[mask]
        if len(candidate_pair_indices) == 0:
            print("ERROR: no successful (rank=1) queries to plot.")
            sys.exit(1)
    elif filter_mode == "failure":
        mask = ranks_for_filter > 1
        candidate_pair_indices = candidate_pair_indices[mask]
        if len(candidate_pair_indices) == 0:
            print("ERROR: no failure (rank>1) queries to plot.")
            sys.exit(1)

    k = min(num_examples, len(candidate_pair_indices))
    return rng.choice(candidate_pair_indices, size=k, replace=False).tolist()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cross_modal_knn(
    emb_a,
    names_a,
    emb_b,
    names_b,
    data_dir_a,
    data_dir_b,
    output_path,
    n_neighbors=5,
    num_examples=6,
    thumb_size=200,
    metric="cosine",
    id_regex=None,
    query_names=None,
    filter_mode="all",
    random_state=42,
    invert_mask=False,
    norm_foreground_only=True,
    transparent_fig=False,
    label_a="RGB",
    label_b="RGB+UV",
):
    """Plot top-K nearest neighbours in B for queries drawn from A.

    Bird backgrounds are made transparent via the multispectral mask band.
    """
    # ---- Pairing ----------------------------------------------------------
    pairs, unmatched_a, dupes_b = build_pairs(names_a, names_b, id_regex=id_regex)
    if not pairs:
        print("ERROR: no specimens could be paired between the two sets.")
        sys.exit(1)
    print(f"Paired {len(pairs)} / {len(names_a)} A specimens "
          f"({len(unmatched_a)} unpaired, {len(dupes_b)} duplicate B stems).")

    pair_q = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_db = np.array([p[1] for p in pairs], dtype=np.int64)

    # ---- Cross-set scores -------------------------------------------------
    print(f"Computing cross-modal {metric} distances …")
    emb_q_pairs = emb_a[pair_q]
    if metric == "cosine":
        scores = _cosine_sim_matrix(emb_q_pairs, emb_b)
        higher_is_better = True
    elif metric == "l2":
        scores = _euclidean_dist_matrix(emb_q_pairs, emb_b)
        higher_is_better = False
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Per-query: top-K neighbours in B
    if higher_is_better:
        topk = np.argsort(-scores, axis=1)[:, :n_neighbors]
    else:
        topk = np.argsort(scores, axis=1)[:, :n_neighbors]
    topk_scores = np.take_along_axis(scores, topk, axis=1)

    # Rank of true target (used for row label + failure/success filtering)
    ranks = compute_ranks(scores, pair_db, higher_is_better=higher_is_better)

    # ---- Choose which rows to display ------------------------------------
    selected_pair_indices = _select_query_pairs(
        names_a, pairs, ranks, num_examples, filter_mode, query_names, random_state,
    )
    print(f"Plotting {len(selected_pair_indices)} query row(s); filter='{filter_mode}'.")

    # ---- Build image indices for each data dir ---------------------------
    print(f"Indexing image directories …")
    index_a, index_a_lower = build_multispectral_index(data_dir_a)
    if data_dir_b and os.path.abspath(data_dir_b) != os.path.abspath(data_dir_a):
        index_b, index_b_lower = build_multispectral_index(data_dir_b)
    else:
        index_b, index_b_lower = index_a, index_a_lower
    print(f"  A: {len(index_a)} rasters under {data_dir_a}")
    if index_b is not index_a:
        print(f"  B: {len(index_b)} rasters under {data_dir_b}")

    # ---- Resolve paths for every thumbnail we need -----------------------
    needed_a, needed_b = [], []
    for pair_idx in selected_pair_indices:
        ia, _ = pairs[pair_idx]
        needed_a.append(names_a[ia])
        for nb in topk[pair_idx]:
            needed_b.append(names_b[nb])

    print("Resolving image paths …")
    path_map_a = {}
    for fname in tqdm(set(needed_a), desc="A image paths"):
        p = find_multispectral_image(data_dir_a, fname,
                                     index=index_a, index_lower=index_a_lower)
        if p is None and index_b is not index_a:
            p = find_multispectral_image(data_dir_b, fname,
                                         index=index_b, index_lower=index_b_lower)
        path_map_a[fname] = p

    path_map_b = {}
    for fname in tqdm(set(needed_b), desc="B image paths"):
        p = find_multispectral_image(data_dir_b, fname,
                                     index=index_b, index_lower=index_b_lower)
        if p is None and index_a is not index_b:
            p = find_multispectral_image(data_dir_a, fname,
                                         index=index_a, index_lower=index_a_lower)
        path_map_b[fname] = p

    # ---- Load thumbnails in parallel -------------------------------------
    load_args = []
    keys = []
    for fname, p in path_map_a.items():
        load_args.append((p, thumb_size, invert_mask, norm_foreground_only))
        keys.append(("a", fname))
    for fname, p in path_map_b.items():
        load_args.append((p, thumb_size, invert_mask, norm_foreground_only))
        keys.append(("b", fname))

    print("Loading thumbnails …")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        loaded = list(tqdm(ex.map(_load_thumbnail, load_args),
                           total=len(load_args), desc="Thumbnails"))

    thumb_a = {fname: thumb for (side, fname), thumb in zip(keys, loaded)
               if side == "a" and thumb is not None}
    thumb_b = {fname: thumb for (side, fname), thumb in zip(keys, loaded)
               if side == "b" and thumb is not None}

    # ---- Build figure ----------------------------------------------------
    n_rows = len(selected_pair_indices)
    n_cols = 1 + n_neighbors
    cell_w, cell_h = 2.2, 2.0

    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * cell_w, n_rows * cell_h),
                            squeeze=False)
    fig.patch.set_facecolor("white")
    if transparent_fig:
        fig.patch.set_alpha(0.0)

    score_label = "cos" if metric == "cosine" else "L2"
    n_db = emb_b.shape[0]

    for row, pair_idx in enumerate(selected_pair_indices):
        ia, ib_true = pairs[pair_idx]
        query_fname = names_a[ia]
        true_rank = int(ranks[pair_idx])

        # Query column
        ax = axs[row, 0]
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        thumb = thumb_a.get(query_fname)
        if thumb is not None:
            ax.imshow(np.asarray(thumb))
        title = f"Query ({label_a})" if row == 0 else "Query"
        ax.set_title(title, fontsize=12, pad=4)
        ax.set_ylabel(f"true rank: {true_rank}/{n_db}",
                      fontsize=10, rotation=0, ha="right", va="center",
                      labelpad=42)

        # Neighbour columns
        for col in range(n_neighbors):
            ax = axs[row, col + 1]
            ax.set_facecolor("white")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            nb_idx_in_b = int(topk[pair_idx, col])
            score = float(topk_scores[pair_idx, col])
            nb_fname = names_b[nb_idx_in_b]
            thumb = thumb_b.get(nb_fname)
            if thumb is not None:
                ax.imshow(np.asarray(thumb))

            is_match = nb_idx_in_b == ib_true
            if is_match:
                ax.set_title(f"#{col+1}  {score_label}={score:.3f}  MATCH",
                             fontsize=11, color="#1a7f37", fontweight="bold", pad=4)
                # Green frame around the matching neighbour
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                w = xmax - xmin
                h = ymin - ymax  # inverted axis
                rect = Rectangle((xmin, ymax), w, h,
                                 linewidth=3, edgecolor="#1a7f37",
                                 facecolor="none", zorder=10)
                ax.add_patch(rect)
            else:
                ax.set_title(f"#{col+1}  {score_label}={score:.3f}",
                             fontsize=11, pad=4)
            if row == 0 and col == 0:
                # add a row-zero suptitle hint via the column-1 axis title
                pass

    fig.suptitle(
        f"Cross-modal KNN: {label_a} query → top-{n_neighbors} in {label_b} "
        f"(n_db={n_db}, metric={metric})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving figure to {output_path} …")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=transparent_fig)
    plt.close(fig)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Cross-modal KNN visualisation: RGB query -> top-K in RGB+UV.",
    )
    parser.add_argument("--embeddings_a", type=str, required=True,
                        help="CSV of embeddings for queries (e.g. RGB). First column = filename.")
    parser.add_argument("--embeddings_b", type=str, required=True,
                        help="CSV of embeddings for the candidate gallery (e.g. RGB+UV).")
    parser.add_argument("--data_dir_a", type=str, required=True,
                        help="Directory of images for set A (queries).")
    parser.add_argument("--data_dir_b", type=str, default=None,
                        help="Directory of images for set B (gallery). Defaults to data_dir_a if omitted.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the PNG.")
    parser.add_argument("--n_neighbors", type=int, default=5,
                        help="Number of nearest neighbours from B per query (default: 5).")
    parser.add_argument("--num_examples", type=int, default=6,
                        help="Number of query rows in the figure (default: 6).")
    parser.add_argument("--thumb_size", type=int, default=200,
                        help="Max thumbnail size in pixels (default: 200).")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"],
                        help="Distance metric for retrieval (default: cosine).")
    parser.add_argument("--id_regex", type=str, default=None,
                        help="Optional regex to extract a specimen ID from filenames "
                             "(bypasses the stem + prefix matching).")
    parser.add_argument("--query_names", type=str, nargs="+", default=None,
                        help="Specific A filenames (or stems) to use as queries. "
                             "Overrides --filter and --num_examples (caps to num_examples).")
    parser.add_argument("--filter", type=str, default="all",
                        choices=["all", "success", "failure"],
                        help="Restrict random query sampling to specimens whose true match was "
                             "retrieved at rank 1 (success), not at rank 1 (failure), or any (all).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for query sampling (default: 42).")
    parser.add_argument("--invert_mask", action="store_true", default=False,
                        help="Invert mask semantics (treat 0 as foreground).")
    parser.add_argument("--norm_foreground_only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Normalise each RGB band using only foreground pixels (default: True).")
    parser.add_argument("--transparent_fig", action="store_true", default=False,
                        help="Save the figure canvas with a transparent background.")
    parser.add_argument("--label_a", type=str, default="RGB",
                        help="Display label for set A (default: RGB).")
    parser.add_argument("--label_b", type=str, default="RGB+UV",
                        help="Display label for set B (default: RGB+UV).")
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Loading embeddings A from {args.embeddings_a} …")
    emb_a, names_a = load_embeddings_from_csv(args.embeddings_a)
    print(f"  -> {emb_a.shape[0]} samples, {emb_a.shape[1]} dimensions")

    print(f"Loading embeddings B from {args.embeddings_b} …")
    emb_b, names_b = load_embeddings_from_csv(args.embeddings_b)
    print(f"  -> {emb_b.shape[0]} samples, {emb_b.shape[1]} dimensions")

    if emb_a.shape[1] != emb_b.shape[1]:
        print(f"ERROR: embedding dimensions differ ({emb_a.shape[1]} vs {emb_b.shape[1]}).")
        sys.exit(1)

    plot_cross_modal_knn(
        emb_a=emb_a, names_a=names_a,
        emb_b=emb_b, names_b=names_b,
        data_dir_a=args.data_dir_a,
        data_dir_b=args.data_dir_b or args.data_dir_a,
        output_path=args.output_path,
        n_neighbors=args.n_neighbors,
        num_examples=args.num_examples,
        thumb_size=args.thumb_size,
        metric=args.metric,
        id_regex=args.id_regex,
        query_names=args.query_names,
        filter_mode=args.filter,
        random_state=args.random_state,
        invert_mask=args.invert_mask,
        norm_foreground_only=args.norm_foreground_only,
        transparent_fig=args.transparent_fig,
        label_a=args.label_a,
        label_b=args.label_b,
    )


if __name__ == "__main__":
    main()
