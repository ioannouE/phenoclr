"""Compute pairwise Mantel correlation between embedding distance matrices.

Python port of the Mantel-test section in ``simclr/simclr_embedding_correlation.R``.

For each pair of embedding CSVs, specimens are aligned by filename (with optional
``--id_regex`` and the same prefix-matching heuristic used in
``visualisations/paired_retrieval_test.py``), pairwise distance matrices are built,
max-normalised, and compared with a Mantel test (Pearson *r* + permutation *p*).

Examples
--------
Two embedding sets (e.g. RGB vs UV), cosine distance (default):

    python mantel_embedding_correlation.py \\
        --embeddings_a rgb_embeddings.csv \\
        --embeddings_b uv_embeddings.csv \\
        --output_dir results/mantel

All pairwise comparisons among three modalities:

    python mantel_embedding_correlation.py \\
        --embeddings rgb.csv uv.csv rgbuv.csv \\
        --labels RGB UV RGBUV \\
        --output_dir results/mantel

Mirror the original R script (Euclidean on set A, cosine on set B):

    python mantel_embedding_correlation.py \\
        --embeddings_a features_a.csv \\
        --embeddings_b simclr_embeddings.csv \\
        --metric_a euclidean \\
        --metric_b cosine \\
        --output_dir results/mantel
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Loading and specimen pairing
# ---------------------------------------------------------------------------

def load_embeddings_from_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
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


def _stem(name: str) -> str:
    s = str(name).strip().replace("\\", "/")
    return os.path.splitext(os.path.basename(s))[0]


def _extract_id(name: str, pattern: re.Pattern | None) -> str | None:
    if pattern is None:
        return _stem(name)
    m = pattern.search(str(name))
    if m is None:
        return None
    if m.groups():
        return m.group(1)
    return m.group(0)


def _prefix_match(stem_a: str, b_index: dict[str, int]) -> int | None:
    if stem_a in b_index:
        return b_index[stem_a]

    forward_prefix = stem_a + "_"
    forward = [k for k in b_index if k.startswith(forward_prefix)]
    backward = [k for k in b_index if stem_a.startswith(k + "_")]

    candidates = forward + backward
    if not candidates:
        return None
    if len(candidates) == 1:
        return b_index[candidates[0]]

    vis_pat = re.compile(re.escape(stem_a) + r"_Vis_G(\d+)$")
    vis_hits = []
    for k in candidates:
        m = vis_pat.fullmatch(k)
        if m:
            vis_hits.append((int(m.group(1)), k))
    if vis_hits:
        vis_hits.sort(key=lambda t: t[0])
        return b_index[vis_hits[0][1]]

    candidates.sort()
    return b_index[candidates[0]]


def build_pairs(
    names_a: np.ndarray,
    names_b: np.ndarray,
    id_regex: str | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[str, int, int]]]:
    """Pair specimens between two sets by filename."""
    pattern = re.compile(id_regex) if id_regex else None

    ids_a = [_extract_id(n, pattern) for n in names_a]
    ids_b = [_extract_id(n, pattern) for n in names_b]

    b_index: dict[str, int] = {}
    duplicate_b = []
    for j, k in enumerate(ids_b):
        if k is None:
            continue
        if k in b_index:
            duplicate_b.append((k, b_index[k], j))
            continue
        b_index[k] = j

    pairs: list[tuple[int, int]] = []
    unmatched_a: list[int] = []
    for i, k in enumerate(ids_a):
        if k is None:
            unmatched_a.append(i)
            continue
        if pattern is None:
            j = _prefix_match(k, b_index)
        else:
            j = b_index.get(k)
        if j is None:
            unmatched_a.append(i)
        else:
            pairs.append((i, j))

    return pairs, unmatched_a, duplicate_b


def align_embeddings(
    emb_a: np.ndarray,
    names_a: np.ndarray,
    emb_b: np.ndarray,
    names_b: np.ndarray,
    id_regex: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return aligned embedding arrays for specimens shared between A and B."""
    pairs, unmatched_a, _ = build_pairs(names_a, names_b, id_regex=id_regex)
    if not pairs:
        raise ValueError(
            "No specimens could be paired between the two embedding sets. "
            "Check filenames or pass --id_regex."
        )
    if unmatched_a:
        print(f"  Warning: {len(unmatched_a)} specimens in set A have no match in set B.")

    idx_a = np.array([p[0] for p in pairs], dtype=np.int64)
    idx_b = np.array([p[1] for p in pairs], dtype=np.int64)
    return emb_a[idx_a], emb_b[idx_b], len(pairs)


# ---------------------------------------------------------------------------
# Distance matrices and Mantel test
# ---------------------------------------------------------------------------

def compute_distance_matrix(embeddings: np.ndarray, metric: str) -> np.ndarray:
    """Build a square pairwise distance matrix."""
    if metric == "cosine":
        normalized = normalize(embeddings)
        distances = pdist(normalized, metric="cosine")
    elif metric == "euclidean":
        distances = pdist(embeddings, metric="euclidean")
    else:
        raise ValueError(f"Unsupported distance metric: {metric!r}. Use 'cosine' or 'euclidean'.")
    return squareform(distances)


def normalize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """Max-normalise a distance matrix (matches the R script)."""
    max_val = float(np.max(distance_matrix))
    if max_val == 0.0:
        return distance_matrix.copy()
    return distance_matrix / max_val


def _upper_triangle_values(distance_matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(distance_matrix.shape[0], k=1)
    return distance_matrix[i, j]


def mantel_test(
    distance_a: np.ndarray,
    distance_b: np.ndarray,
    permutations: int = 999,
    seed: int = 42,
) -> tuple[float, float]:
    """Mantel test: Pearson correlation of distance matrices + permutation p-value."""
    a_vals = _upper_triangle_values(distance_a)
    b_vals = _upper_triangle_values(distance_b)
    r_obs, _ = pearsonr(a_vals, b_vals)

    n = distance_a.shape[0]
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        b_perm = distance_b[np.ix_(perm, perm)]
        r_perm, _ = pearsonr(a_vals, _upper_triangle_values(b_perm))
        if r_perm >= r_obs:
            extreme += 1

    p_value = (extreme + 1) / (permutations + 1)
    return float(r_obs), float(p_value)


def run_mantel_pair(
    path_a: str,
    path_b: str,
    label_a: str,
    label_b: str,
    metric_a: str,
    metric_b: str,
    permutations: int,
    seed: int,
    id_regex: str | None,
) -> dict:
    """Load, align, and compare two embedding sets with a Mantel test."""
    print(f"\n[{label_a} vs {label_b}]")
    print(f"  Loading A: {path_a}")
    emb_a, names_a = load_embeddings_from_csv(path_a)
    print(f"  Loading B: {path_b}")
    emb_b, names_b = load_embeddings_from_csv(path_b)
    print(f"  Raw sizes: A={emb_a.shape[0]}, B={emb_b.shape[0]}")

    print("  Pairing specimens ...")
    aligned_a, aligned_b, n_paired = align_embeddings(
        emb_a, names_a, emb_b, names_b, id_regex=id_regex
    )
    print(f"  Paired specimens used for Mantel: {n_paired}")

    dist_a = normalize_distance_matrix(compute_distance_matrix(aligned_a, metric_a))
    dist_b = normalize_distance_matrix(compute_distance_matrix(aligned_b, metric_b))

    r_val, p_val = mantel_test(dist_a, dist_b, permutations=permutations, seed=seed)
    print(f"  Mantel r = {r_val:.6f}, p = {p_val:.6f} ({permutations} permutations)")

    return {
        "set_a": label_a,
        "set_b": label_b,
        "path_a": path_a,
        "path_b": path_b,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "n_paired": n_paired,
        "mantel_r": r_val,
        "p_value": p_val,
        "permutations": permutations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise Mantel correlation between embedding distance matrices.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        nargs="+",
        default=None,
        help="Two or more embedding CSV paths; all unique pairs are compared.",
    )
    parser.add_argument(
        "--embeddings_a",
        type=str,
        default=None,
        help="First embedding CSV (alternative to --embeddings).",
    )
    parser.add_argument(
        "--embeddings_b",
        type=str,
        default=None,
        help="Second embedding CSV (alternative to --embeddings).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Display labels for each --embeddings path (same order).",
    )
    parser.add_argument(
        "--label_a",
        type=str,
        default=None,
        help="Display label for set A when using --embeddings_a/--embeddings_b.",
    )
    parser.add_argument(
        "--label_b",
        type=str,
        default=None,
        help="Display label for set B when using --embeddings_a/--embeddings_b.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for mantel_summary.csv (optional; results always printed).",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for all sets unless --metric_a/--metric_b are set.",
    )
    parser.add_argument(
        "--metric_a",
        type=str,
        default=None,
        choices=["cosine", "euclidean"],
        help="Distance metric for set A / first path (overrides --distance_metric).",
    )
    parser.add_argument(
        "--metric_b",
        type=str,
        default=None,
        choices=["cosine", "euclidean"],
        help="Distance metric for set B / second path when comparing exactly two sets.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Number of permutations for the Mantel test (default: 999, as in R).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Mantel permutations.",
    )
    parser.add_argument(
        "--id_regex",
        type=str,
        default=None,
        help="Optional regex to extract specimen IDs from filenames.",
    )
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    """Return list of (path_a, path_b, label_a, label_b) pairs to compare."""
    if args.embeddings and (args.embeddings_a or args.embeddings_b):
        print("ERROR: use either --embeddings or --embeddings_a/--embeddings_b, not both.")
        sys.exit(1)

    if args.embeddings:
        paths = args.embeddings
        if len(paths) < 2:
            print("ERROR: --embeddings requires at least two CSV paths.")
            sys.exit(1)
        if args.labels and len(args.labels) != len(paths):
            print("ERROR: --labels must have the same length as --embeddings.")
            sys.exit(1)
        labels = args.labels or [_default_label(p) for p in paths]
        return [
            (paths[i], paths[j], labels[i], labels[j])
            for i, j in itertools.combinations(range(len(paths)), 2)
        ]

    if args.embeddings_a and args.embeddings_b:
        label_a = args.label_a or _default_label(args.embeddings_a)
        label_b = args.label_b or _default_label(args.embeddings_b)
        return [(args.embeddings_a, args.embeddings_b, label_a, label_b)]

    print(
        "ERROR: provide either --embeddings (2+ paths) or "
        "--embeddings_a and --embeddings_b."
    )
    sys.exit(1)


def main() -> None:
    args = parse_args()
    pairs = _resolve_inputs(args)

    default_metric = args.distance_metric
    results = []
    for i, (path_a, path_b, label_a, label_b) in enumerate(pairs):
        metric_a = args.metric_a or default_metric
        if len(pairs) == 1 and args.metric_b is not None:
            metric_b = args.metric_b
        else:
            metric_b = args.metric_b or default_metric

        # Offset seed per pair so repeated runs stay reproducible but pairs differ.
        seed = args.seed + i
        row = run_mantel_pair(
            path_a=path_a,
            path_b=path_b,
            label_a=label_a,
            label_b=label_b,
            metric_a=metric_a,
            metric_b=metric_b,
            permutations=args.permutations,
            seed=seed,
            id_regex=args.id_regex,
        )
        results.append(row)

    summary = pd.DataFrame(results)
    print("\nSummary")
    print(summary[["set_a", "set_b", "n_paired", "mantel_r", "p_value"]].to_string(index=False))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "mantel_summary.csv")
        summary.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
