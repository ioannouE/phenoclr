"""Paired RGB vs RGB+UV retrieval test.

For each specimen, use its embedding from set A (e.g. RGB) as a query and check
whether the paired embedding of the *same specimen* in set B (e.g. RGB+UV or
UV-only) is retrieved in the top-k of all set B candidates. The test is run in
both directions (A->B and B->A) because they can differ.

Metrics reported per direction and per metric (cosine + L2):
  - Recall@1, Recall@5, Recall@10
  - Median rank
  - Mean rank
  - Mean Reciprocal Rank (MRR)

This directly tests whether two modalities preserve specimen identity in a
shared embedding space.

Specimen pairing strategy (in order):
  1. Exact filename-stem match between the two CSVs.
  2. Prefix match: one stem is the other with an extra ``_<suffix>`` (handles
     e.g. RGB previews named ``..._Vis_G167935`` while multispectral TIFFs use
     just the shared prefix). This mirrors the matching used in
     ``plot_embeddings_multispectral.py``.
  3. Optional ``--id_regex`` to extract a custom specimen ID from each filename
     (first capture group is used if any, otherwise the whole match).

Both embedding sets must share the same dimensionality (e.g. same projection
head output size), since the test computes cross-set distances directly.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings_from_csv(path: str):
    """Load embeddings from CSV. Returns (embeddings, filenames)."""
    df = pd.read_csv(path)
    first = df.columns[0].lower()
    if first in ("filename", "file", "name", "image"):
        filenames = df.iloc[:, 0].astype(str).values
        embeddings = df.iloc[:, 1:].select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    else:
        print(f"WARNING: first column of {path} ('{df.columns[0]}') does not look like a "
              f"filename column; using row index as identifier.")
        filenames = np.array([str(i) for i in range(len(df))])
        embeddings = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    return embeddings, filenames


# ---------------------------------------------------------------------------
# Specimen pairing
# ---------------------------------------------------------------------------

def _stem(name: str) -> str:
    s = str(name).strip().replace("\\", "/")
    return os.path.splitext(os.path.basename(s))[0]


def _extract_id(name: str, pattern: re.Pattern | None) -> str | None:
    """Extract a specimen identifier from a filename.

    If *pattern* is given, returns its first capture group (or whole match if
    no groups). Otherwise returns the filename stem (basename without extension).
    """
    if pattern is None:
        return _stem(name)
    m = pattern.search(str(name))
    if m is None:
        return None
    if m.groups():
        return m.group(1)
    return m.group(0)


def _prefix_match(stem_a: str, b_index: dict[str, int]) -> int | None:
    """Resolve a stem against a {stem: idx} index using a bidirectional prefix rule.

    Returns the index of the matched stem in B (or ``None`` if no unambiguous
    match exists). When the A stem has multiple ``..._Vis_G<id>`` extensions in
    B (RGB preview naming), the smallest G id is preferred (matches the heuristic
    in ``plot_embeddings_multispectral.py``).
    """
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


def build_pairs(names_a, names_b, id_regex: str | None = None):
    """Pair up specimens between two sets by filename.

    Returns:
        pairs: list of (idx_a, idx_b) tuples for matched specimens.
        unmatched_a: list of idx_a values that could not be paired.
        duplicate_b: list of (stem, idx_kept, idx_dropped) for B entries with
            duplicate stems (only the first occurrence is kept in the index).
    """
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

    pairs = []
    unmatched_a = []
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


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _cosine_sim_matrix(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    return normalize(q.astype(np.float32)) @ normalize(db.astype(np.float32)).T


def _euclidean_dist_matrix(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    # ||q-d||^2 = ||q||^2 + ||d||^2 - 2 q.d  (keep float32 for memory)
    q = q.astype(np.float32)
    db = db.astype(np.float32)
    qn = np.sum(q * q, axis=1, keepdims=True)            # (Q, 1)
    dn = np.sum(db * db, axis=1, keepdims=True).T        # (1, D)
    cross = q @ db.T                                     # (Q, D)
    sq = qn + dn - 2.0 * cross
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq, dtype=np.float32)


def compute_ranks(scores: np.ndarray, target_indices: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Rank of each query's paired target among all candidates.

    rank == 1 means the paired specimen is the top-1 nearest neighbour.
    Ties are broken pessimistically (we count strictly-better candidates), which
    is the common convention in image retrieval.
    """
    n_q = scores.shape[0]
    ranks = np.empty(n_q, dtype=np.int64)
    target_scores = scores[np.arange(n_q), target_indices]  # (Q,)
    if higher_is_better:
        better = scores > target_scores[:, None]
    else:
        better = scores < target_scores[:, None]
    ranks = 1 + better.sum(axis=1)
    return ranks


def retrieval_metrics(ranks: np.ndarray, ks=(1, 5, 10)) -> dict:
    metrics = {f"recall@{k}": float(np.mean(ranks <= k)) for k in ks}
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    metrics["n_queries"] = int(len(ranks))
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rank_histogram(rank_records: list[dict], output_path: str, n_db: int):
    """Plot a histogram of ranks per (direction, metric) combination.

    Each record is {"label": str, "ranks": np.ndarray}.
    """
    import matplotlib.pyplot as plt

    n = len(rank_records)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    # log-spaced bins so we see the head of the distribution clearly
    bins = np.geomspace(1, max(n_db, 2), num=30)
    for ax, rec in zip(axes, rank_records):
        ranks = rec["ranks"]
        ax.hist(ranks, bins=bins, color="#1f77b4", edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("rank of paired specimen (log)")
        ax.set_title(rec["label"], fontsize=10)
        med = float(np.median(ranks))
        ax.axvline(med, color="#d62728", linestyle="--", linewidth=1,
                   label=f"median = {med:.0f}")
        ax.axvline(1, color="#2ca02c", linestyle=":", linewidth=1, label="rank = 1")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("# queries")
    fig.suptitle("Paired retrieval — rank of true match", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Rank histogram saved to {output_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_direction(
    name: str,
    emb_q: np.ndarray,
    emb_db: np.ndarray,
    pair_q: np.ndarray,
    pair_db: np.ndarray,
    metrics: list[str],
    ks: tuple[int, ...],
):
    """Run retrieval for one direction and return (rows, rank_records).

    *rows* is a list of dicts for the summary CSV; *rank_records* is a list of
    dicts for the per-query CSV and the histogram plot.
    """
    rows = []
    rank_records = []

    emb_q_pairs = emb_q[pair_q]  # only paired queries have a ground-truth target

    for metric in metrics:
        if metric == "cosine":
            scores = _cosine_sim_matrix(emb_q_pairs, emb_db)
            higher_is_better = True
        elif metric == "l2":
            scores = _euclidean_dist_matrix(emb_q_pairs, emb_db)
            higher_is_better = False
        else:
            raise ValueError(f"Unknown metric: {metric}")

        ranks = compute_ranks(scores, pair_db, higher_is_better=higher_is_better)

        m = retrieval_metrics(ranks, ks=ks)
        label = f"{name} | {metric}"
        print(f"\n[{label}]  n={m['n_queries']}, n_db={emb_db.shape[0]}")
        for k in ks:
            print(f"  Recall@{k:<2d}: {m[f'recall@{k}']:.4f}")
        print(f"  Median rank: {m['median_rank']:.1f}")
        print(f"  Mean rank:   {m['mean_rank']:.2f}")
        print(f"  MRR:         {m['mrr']:.4f}")

        row = {"direction": name, "metric": metric, "n_db": int(emb_db.shape[0]), **m}
        rows.append(row)
        rank_records.append({
            "label": label,
            "direction": name,
            "metric": metric,
            "ranks": ranks,
            "query_idx": pair_q,
            "target_idx": pair_db,
        })

    return rows, rank_records


def get_args():
    parser = argparse.ArgumentParser(
        description="Paired RGB vs RGB+UV retrieval test (Recall@k, median rank, MRR).",
    )
    parser.add_argument("--embeddings_a", type=str, required=True,
                        help="CSV of embeddings for set A (e.g. RGB). First column = filename.")
    parser.add_argument("--embeddings_b", type=str, required=True,
                        help="CSV of embeddings for set B (e.g. RGB+UV or UV). First column = filename.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save summary, per-query ranks, and histogram. "
                             "If omitted, results are only printed.")
    parser.add_argument("--metrics", type=str, nargs="+", default=["cosine", "l2"],
                        choices=["cosine", "l2"],
                        help="Distance metric(s) to use (default: cosine and l2).")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10],
                        help="Cutoffs for Recall@k (default: 1 5 10).")
    parser.add_argument("--id_regex", type=str, default=None,
                        help="Optional regex to extract a custom specimen ID from filenames. "
                             "If set, the filename-stem and prefix-fallback logic is bypassed.")
    parser.add_argument("--direction", type=str, default="both",
                        choices=["a_to_b", "b_to_a", "both"],
                        help="Which retrieval direction(s) to evaluate (default: both).")
    parser.add_argument("--label_a", type=str, default=None,
                        help="Display label for set A (default: filename of embeddings_a).")
    parser.add_argument("--label_b", type=str, default=None,
                        help="Display label for set B (default: filename of embeddings_b).")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Save a rank histogram plot to output_dir.")
    return parser.parse_args()


def main(args):
    print(f"Loading embeddings A from {args.embeddings_a} ...")
    emb_a, names_a = load_embeddings_from_csv(args.embeddings_a)
    print(f"  -> {emb_a.shape[0]} samples, {emb_a.shape[1]} dimensions")

    print(f"Loading embeddings B from {args.embeddings_b} ...")
    emb_b, names_b = load_embeddings_from_csv(args.embeddings_b)
    print(f"  -> {emb_b.shape[0]} samples, {emb_b.shape[1]} dimensions")

    if emb_a.shape[1] != emb_b.shape[1]:
        print(f"ERROR: embedding dimensions differ ({emb_a.shape[1]} vs {emb_b.shape[1]}). "
              "Cross-set distances are not comparable.")
        sys.exit(1)

    label_a = args.label_a or os.path.splitext(os.path.basename(args.embeddings_a))[0]
    label_b = args.label_b or os.path.splitext(os.path.basename(args.embeddings_b))[0]

    print("\nPairing specimens by filename ...")
    pairs, unmatched_a, dupes_b = build_pairs(names_a, names_b, id_regex=args.id_regex)
    print(f"  Paired:                {len(pairs)} / {len(names_a)} A-side specimens")
    print(f"  Unpaired in A:         {len(unmatched_a)}")
    print(f"  Duplicate stems in B:  {len(dupes_b)} "
          "(first occurrence kept as the canonical target)")

    if not pairs:
        print("ERROR: no specimens could be paired between the two sets. "
              "Check filenames or pass --id_regex.")
        sys.exit(1)

    if unmatched_a:
        preview = [str(names_a[i]) for i in unmatched_a[:5]]
        print(f"  Example unpaired A filenames: {preview}"
              + (" ..." if len(unmatched_a) > 5 else ""))

    pair_q = np.array([p[0] for p in pairs], dtype=np.int64)  # indices into A
    pair_db = np.array([p[1] for p in pairs], dtype=np.int64) # indices into B

    directions = []
    if args.direction in ("a_to_b", "both"):
        directions.append((f"{label_a} -> {label_b}", emb_a, emb_b, pair_q, pair_db))
    if args.direction in ("b_to_a", "both"):
        directions.append((f"{label_b} -> {label_a}", emb_b, emb_a, pair_db, pair_q))

    all_rows = []
    all_rank_records = []
    for name, q, db, pq, pd_ in directions:
        rows, rec = run_direction(name, q, db, pq, pd_, metrics=args.metrics, ks=tuple(args.ks))
        all_rows.extend(rows)
        all_rank_records.extend(rec)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        summary_path = os.path.join(args.output_dir, "paired_retrieval_summary.csv")
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        per_query_rows = []
        for rec in all_rank_records:
            for qi, ti, r in zip(rec["query_idx"], rec["target_idx"], rec["ranks"]):
                if rec["direction"].startswith(label_a):
                    q_name, t_name = names_a[qi], names_b[ti]
                else:
                    q_name, t_name = names_b[qi], names_a[ti]
                per_query_rows.append({
                    "direction": rec["direction"],
                    "metric": rec["metric"],
                    "query": q_name,
                    "target": t_name,
                    "rank": int(r),
                    "reciprocal_rank": 1.0 / float(r),
                })
        per_query_path = os.path.join(args.output_dir, "paired_retrieval_per_query.csv")
        pd.DataFrame(per_query_rows).to_csv(per_query_path, index=False)
        print(f"Per-query ranks saved to {per_query_path}")

        if args.plot and all_rank_records:
            n_db = max(emb_a.shape[0], emb_b.shape[0])
            plot_rank_histogram(all_rank_records,
                                os.path.join(args.output_dir, "paired_retrieval_ranks.png"),
                                n_db=n_db)

    print("\nDone.")


if __name__ == "__main__":
    main(get_args())
