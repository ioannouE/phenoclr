"""
Combined bootstrap evaluation: embedding-space metrics (silhouette / variance
explained) AND perceptual-judgement metrics (BAPPS 2AFC / JND) with matching
statistical error bars (mean +/- std, percentile CI) for both.

This script is a superset of `eval_silhouette_variance_bootstrap.py`: the
embedding-space logic is identical (same point-estimate + subsampling
bootstrap procedure). It additionally runs the PhenoCLR / LPIPS forward passes
on the BAPPS 2AFC and JND datasets *once* to collect per-example scores, then
bootstraps those pre-computed per-example arrays (resampling with
replacement) to get statistical errors for:
  - BAPPS 2AFC accuracy (per subset, e.g. 'val/color', and overall = mean
    across subsets)
  - BAPPS JND ROC-AUC and Acc@best-threshold (per subset, e.g. 'val/cnn',
    'val/traditional', and overall = mean across subsets)

Because the per-example scores/labels only need to be computed once (a
single forward pass per subset), bootstrapping over them thousands of times
is cheap (pure numpy), unlike re-running the network per replicate.

Usage
-----
python eval_full_bootstrap.py \\
    --embeddings /path/to/embeddings.csv --output results/ \\
    --n-bootstrap 1000 --ci 0.95 \\
    --run-perceptual \\
    --bapps-2afc-root /path/to/PerceptualSimilarity/dataset/2afc \\
    --bapps-jnd-root /path/to/PerceptualSimilarity/dataset/jnd \\
    --eval-simclr --simclr-checkpoint /path/to/ckpt.ckpt \\
    --simclr-module train.simclr_birdcolour_RGB \\
    --eval-lpips --lpips-net alex
"""

import os
import sys
import json
import argparse
import warnings

# Silence noisy, unactionable deprecation warnings from third-party libraries
# (torchvision's `T.ToTensor()` deprecation notice, LPIPS' `torch.load`
# `weights_only` FutureWarning, etc.) that otherwise clutter the bootstrap logs.
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# --- Make perception_eval importable regardless of CWD ---
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SIMCLR_DIR = os.path.dirname(EVAL_DIR)  # simclr/
PERCEPTION_EVAL_DIR = os.path.join(SIMCLR_DIR, "perception_eval")
for p in (SIMCLR_DIR, PERCEPTION_EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# Embedding-space metrics: silhouette score + variance explained
# (identical logic to eval_silhouette_variance_bootstrap.py)
# =============================================================================

def calculate_silhouette_metrics(embeddings, labels, metric='euclidean'):
    """Calculate silhouette score directly using sklearn."""
    return float(silhouette_score(X=embeddings, labels=labels, metric=metric))


def extract_metadata_from_filename(filename):
    """
    Extract species and sex from filename, supporting multiple formats.
    Format 1: Genus_species_id_SEX... (e.g., 'Anthochaera_chrysoptera_1_M_12345')
    Format 2: SpeciesName_ID1_ID2 (e.g., 'Abraxas_fulvobasalis_1D_040917')
    """
    base_name = os.path.basename(filename)
    parts = base_name.split('_')

    if len(parts) >= 4 and parts[3] in ['M', 'F', 'U']:
        species = f"{parts[0]}_{parts[1]}"
        sex = parts[3]
        return species, sex

    if len(parts) > 2:
        species = "_".join(parts[:-2])
        sex = None
        return species, sex

    return base_name, None


def calculate_variance_explained(embeddings, labels):
    """Calculate percentage of variance explained by labels."""
    overall_mean = embeddings.mean(axis=0)
    total_var = np.sum(((embeddings - overall_mean) ** 2).sum(axis=0))

    unique_labels = np.unique(labels)
    between_var = 0

    for label in unique_labels:
        mask = labels == label
        group_size = np.sum(mask)
        if group_size > 0:
            group_mean = embeddings[mask].mean(axis=0)
            between_var += group_size * np.sum((group_mean - overall_mean) ** 2)

    return float(between_var / total_var * 100) if total_var > 0 else 0.0


def filter_classes(labels, min_samples=2):
    """Filter out classes with fewer than min_samples samples."""
    unique_labels = np.unique(labels)
    valid_classes = [label for label in unique_labels if np.sum(labels == label) >= min_samples]
    mask = np.isin(labels, valid_classes)
    return mask, valid_classes


def compute_point_estimates(embeddings, species_labels, sex_labels, class_labels, min_samples_per_class=2):
    """Compute silhouette + variance-explained metrics on a single (sub)sample, no resampling."""
    mask, valid_classes = filter_classes(class_labels, min_samples_per_class)
    embeddings_filtered = embeddings[mask]
    class_labels_filtered = class_labels[mask]
    normalized_embeddings = normalize(embeddings_filtered, norm='l2')

    silhouette_results = {}
    for metric in ['euclidean', 'cosine']:
        if len(np.unique(class_labels_filtered)) < 2:
            silhouette_results[f'{metric}_distance'] = np.nan
            continue
        silhouette_results[f'{metric}_distance'] = calculate_silhouette_metrics(
            normalized_embeddings, class_labels_filtered, metric=metric
        )

    variance_results = {
        'species_variance': calculate_variance_explained(embeddings, species_labels),
        'sex_variance': calculate_variance_explained(embeddings, sex_labels),
        'class_variance': calculate_variance_explained(embeddings, class_labels)
    }

    return silhouette_results, variance_results, len(valid_classes), len(embeddings_filtered)


def bootstrap_replicate_embeddings(embeddings, species_labels, sex_labels, class_labels, seed,
                                    min_samples_per_class=2, replacement=False, subsample_frac=0.8):
    """
    Resample the dataset and recompute embedding metrics on the replicate.

    Note: classic bootstrap (replacement=True) draws rows with replacement, which
    creates exact-duplicate embeddings within a replicate. For distance-based metrics
    like silhouette score, a duplicated point has a same-class neighbor at distance 0,
    which mechanically inflates the score and biases the bootstrap mean upward relative
    to the true (no-duplicates) point estimate. Subsampling without replacement
    (replacement=False) avoids this bias while still producing a resampling-based
    variance/CI estimate.
    """
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    if replacement:
        indices = rng.choice(n, size=n, replace=True)
    else:
        sample_size = max(2, int(round(n * subsample_frac)))
        indices = rng.choice(n, size=sample_size, replace=False)

    silhouette_results, variance_results, _, _ = compute_point_estimates(
        embeddings[indices], species_labels[indices], sex_labels[indices], class_labels[indices],
        min_samples_per_class=min_samples_per_class
    )
    return silhouette_results, variance_results


def summarize_bootstrap(values, ci=0.95):
    """Mean, std, and percentile confidence interval for a list of bootstrap values (NaNs dropped)."""
    values = np.array([v for v in values if not np.isnan(v)])
    alpha = (1.0 - ci) / 2.0
    if len(values) == 0:
        return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n_valid': 0}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'ci_lower': float(np.percentile(values, 100 * alpha)),
        'ci_upper': float(np.percentile(values, 100 * (1 - alpha))),
        'n_valid': int(len(values))
    }


def run_embedding_bootstrap(embeddings_csv, num_samples, random_seed, n_bootstrap, ci,
                             replacement, subsample_frac):
    """Full embedding-space analysis: point estimates + bootstrap summary. Mirrors
    eval_silhouette_variance_bootstrap.analyze_embeddings but returns results instead of
    writing files directly (writing is handled centrally at the end of this script)."""
    min_samples_per_class = 2

    print("\n[Embeddings] Loading embeddings...")
    df = pd.read_csv(embeddings_csv)
    embeddings = df.iloc[:, 1:].values
    filenames = df.iloc[:, 0].values

    if num_samples is not None and num_samples < len(embeddings):
        print(f"[Embeddings] Sampling {num_samples} embeddings...")
        np.random.seed(random_seed)
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings = embeddings[indices]
        filenames = filenames[indices]
    else:
        print(f"[Embeddings] Using all {len(embeddings)} samples...")

    print("[Embeddings] Extracting metadata...")
    species_labels, sex_labels, class_labels = [], [], []
    for filename in tqdm(filenames):
        species, sex = extract_metadata_from_filename(filename)
        species_labels.append(species)
        sex_labels.append(sex if sex else 'U')
        class_labels.append(f"{species}_{sex}" if sex else species)

    species_labels = np.array(species_labels)
    sex_labels = np.array(sex_labels)
    class_labels = np.array(class_labels)

    print("\n[Embeddings] Dataset statistics:")
    print(f"Total samples: {len(embeddings)}")
    print(f"Total unique species: {len(np.unique(species_labels))}")
    print(f"Total unique species_sex classes: {len(np.unique(class_labels))}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\n[Embeddings] Calculating point estimates on full dataset...")
    point_silhouette, point_variance, num_valid_classes, num_filtered = compute_point_estimates(
        embeddings, species_labels, sex_labels, class_labels, min_samples_per_class
    )
    for metric, score in point_silhouette.items():
        print(f"Point estimate silhouette ({metric}): {score:.4f}")
    for name, score in point_variance.items():
        print(f"Point estimate {name}: {score:.1f}%")

    resample_desc = "with replacement (full size)" if replacement else f"without replacement ({subsample_frac:.0%} subsample)"
    print(f"\n[Embeddings] Running {n_bootstrap} bootstrap replicates, sampling {resample_desc}...")
    boot_silhouette = {'euclidean_distance': [], 'cosine_distance': []}
    boot_variance = {'species_variance': [], 'sex_variance': [], 'class_variance': []}

    for i in tqdm(range(n_bootstrap)):
        seed = random_seed + i
        sil, var = bootstrap_replicate_embeddings(
            embeddings, species_labels, sex_labels, class_labels, seed, min_samples_per_class,
            replacement=replacement, subsample_frac=subsample_frac
        )
        for k, v in sil.items():
            boot_silhouette[k].append(v)
        for k, v in var.items():
            boot_variance[k].append(v)

    silhouette_summary = {k: summarize_bootstrap(v, ci=ci) for k, v in boot_silhouette.items()}
    variance_summary = {k: summarize_bootstrap(v, ci=ci) for k, v in boot_variance.items()}

    print(f"\n[Embeddings] Bootstrap results ({n_bootstrap} resamples, {int(ci * 100)}% CI):")
    for metric, summary in silhouette_summary.items():
        print(f"Silhouette ({metric}): {summary['mean']:.4f} +/- {summary['std']:.4f} "
              f"[{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]")
    for name, summary in variance_summary.items():
        print(f"{name}: {summary['mean']:.1f}% +/- {summary['std']:.1f}% "
              f"[{summary['ci_lower']:.1f}%, {summary['ci_upper']:.1f}%]")

    metadata = {
        'num_samples': len(embeddings),
        'num_species': len(np.unique(species_labels)),
        'num_sexes': len(np.unique(sex_labels)),
        'num_classes': len(np.unique(class_labels)),
        'num_valid_classes': num_valid_classes,
        'min_samples_per_class': min_samples_per_class,
        'samples_after_filtering': num_filtered,
        'embedding_dim': int(embeddings.shape[1]),
        'random_seed': random_seed,
        'sampled_data': num_samples is not None,
        'n_bootstrap': n_bootstrap,
        'confidence_level': ci,
        'resample_with_replacement': replacement,
        'subsample_frac': subsample_frac if not replacement else None
    }

    raw_rows = []
    for i in range(n_bootstrap):
        for metric, values in boot_silhouette.items():
            raw_rows.append({'iteration': i, 'metric_type': 'silhouette', 'metric': metric, 'value': values[i]})
        for name, values in boot_variance.items():
            raw_rows.append({'iteration': i, 'metric_type': 'variance_explained', 'metric': name, 'value': values[i]})

    return {
        'metadata': metadata,
        'point_estimate': {'silhouette_analysis': point_silhouette, 'variance_explained': point_variance},
        'bootstrap': {'silhouette_analysis': silhouette_summary, 'variance_explained': variance_summary},
        'raw_rows': raw_rows,
    }


# =============================================================================
# Perceptual metrics: BAPPS 2AFC + JND, bootstrapped over pre-computed
# per-example scores (single forward pass, cheap numpy resampling after).
# =============================================================================

def _derive_seed(base_seed, *ints):
    """Deterministic (reproducible across processes/runs) integer seed derived from
    a base seed plus a handful of small integer indices. Avoids Python's built-in
    hash() for strings, which is randomized per-process (PYTHONHASHSEED) and would
    silently break reproducibility of --random-seed across runs."""
    seed = base_seed
    for i, v in enumerate(ints):
        seed = (seed + (i + 1) * 1_000_003 * (v + 1)) % (2**32 - 1)
    return seed


def bootstrap_means_vectorized(raw_scores, n_bootstrap, seed):
    """Resample per-example fractional 2AFC scores with replacement for ALL bootstrap
    replicates in one batched numpy call (draws an (n_bootstrap, n) index matrix at
    once instead of looping in Python), returning an array of per-replicate means."""
    n = len(raw_scores)
    if n == 0:
        return np.full(n_bootstrap, np.nan)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=(n_bootstrap, n), replace=True)
    return raw_scores[idx].mean(axis=1)


def bootstrap_jnd_vectorized(scores, y_true, n_bootstrap, seed):
    """Resample (score, label) pairs with replacement for ALL bootstrap replicates at
    once; returns (aucs, accs) arrays of length n_bootstrap.

    ROC-AUC is fully vectorized via a rank trick (argsort-of-argsort). This skips
    explicit tie-averaging, but bootstrap resampling only ever creates ties between
    exact duplicates of the *same* original example (real-valued distances essentially
    never collide otherwise) which always share the same label -- and same-label ties
    don't affect the AUC formula regardless of how ranks are broken within the tie.
    So this matches the tie-averaged AUC used for the point estimate.

    Acc@best-threshold still requires a per-replicate O(n log n) search, so it is
    looped over replicates -- but the RandomState / index-generation overhead that
    used to dominate is eliminated since indices are drawn once for all replicates.
    """
    from eval_bapps_jnd import best_accuracy_threshold  # local import, see sys.path setup

    n = len(scores)
    if n == 0:
        return np.full(n_bootstrap, np.nan), np.full(n_bootstrap, np.nan)

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=(n_bootstrap, n), replace=True)
    scores_b = scores[idx]  # (B, n)
    y_b = y_true[idx].astype(np.float64)  # (B, n)

    order = np.argsort(scores_b, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(n_bootstrap)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]

    n_pos = y_b.sum(axis=1)
    n_neg = n - n_pos
    sum_pos_ranks = (ranks * y_b).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        aucs = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    aucs = np.where((n_pos == 0) | (n_neg == 0), np.nan, aucs)

    accs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        _, accs[i] = best_accuracy_threshold(y_b[i].astype(np.int32), scores_b[i])

    return aucs, accs


def run_2afc_bootstrap(args, device, simclr_model, lpips_model, n_bootstrap, ci, random_seed):
    """Point estimate + bootstrap for BAPPS 2AFC, per subset and overall (mean across subsets)."""
    from eval_bapps_2afc import evaluate_simclr_2afc, evaluate_lpips_2afc

    subsets = args.bapps_2afc_subsets
    per_subset = {}  # subset -> {'lpips': raw_array, 'simclr': raw_array}
    point = {}  # subset -> {'lpips': acc, 'simclr': acc}

    for subset in subsets:
        subset_dir = os.path.join(args.bapps_2afc_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"[2AFC][WARN] Skipping missing subset: {subset_dir}")
            continue
        per_subset[subset] = {}
        point[subset] = {}

        if args.eval_lpips:
            acc, raw = evaluate_lpips_2afc(
                subset_dir, device=device, batch_size=args.batch_size, num_workers=args.num_workers,
                lpips_net=args.lpips_net, lpips_size=args.lpips_size, return_raw=True,
                lpips_model=lpips_model,
            )
            per_subset[subset]['lpips'] = raw
            point[subset]['lpips'] = acc
            print(f"[2AFC] LPIPS[{args.lpips_net}] {subset:>16}: {acc*100:.2f}% (n={len(raw)})")

        if args.eval_simclr and simclr_model is not None:
            acc, raw = evaluate_simclr_2afc(
                subset_dir, model=simclr_model, device=device, input_size=args.simclr_input_size,
                batch_size=args.batch_size, num_workers=args.num_workers, distance=args.simclr_distance,
                normalize=(not args.no_simclr_normalize), return_raw=True,
            )
            tag = f"simclr_{args.simclr_distance}{'_unnorm' if args.no_simclr_normalize else ''}"
            per_subset[subset]['simclr'] = raw
            point[subset]['simclr'] = acc
            print(f"[2AFC] PhenoCLR[{args.simclr_distance}] {subset:>16}: {acc*100:.2f}% (n={len(raw)})")

    methods = sorted({m for d in per_subset.values() for m in d.keys()})
    valid_subsets = list(per_subset.keys())

    point_overall = {
        m: float(np.mean([point[s][m] for s in valid_subsets if m in point[s]]))
        for m in methods
    }

    print(f"\n[2AFC] Running {n_bootstrap} bootstrap replicates over pre-computed scores "
          f"(vectorized, no re-inference)...")
    boot_per_subset = {s: {} for s in valid_subsets}
    for s_i, s in enumerate(tqdm(valid_subsets)):
        for m_i, m in enumerate(sorted(per_subset[s].keys())):
            raw = per_subset[s][m]
            seed = _derive_seed(random_seed, s_i, m_i)
            boot_per_subset[s][m] = bootstrap_means_vectorized(raw, n_bootstrap, seed)

    # Overall (per replicate) = mean across subsets, same semantics as the point estimate
    boot_overall = {
        m: np.mean([boot_per_subset[s][m] for s in valid_subsets if m in boot_per_subset[s]], axis=0)
        for m in methods
    }

    summary_per_subset = {
        s: {m: summarize_bootstrap(v, ci=ci) for m, v in mdict.items()}
        for s, mdict in boot_per_subset.items()
    }
    summary_overall = {m: summarize_bootstrap(v, ci=ci) for m, v in boot_overall.items()}

    for s in valid_subsets:
        for m, summ in summary_per_subset[s].items():
            print(f"[2AFC] {s} [{m}]: {summ['mean']*100:.2f}% +/- {summ['std']*100:.2f}% "
                  f"[{summ['ci_lower']*100:.2f}%, {summ['ci_upper']*100:.2f}%]")
    for m, summ in summary_overall.items():
        print(f"[2AFC] OVERALL [{m}]: {summ['mean']*100:.2f}% +/- {summ['std']*100:.2f}% "
              f"[{summ['ci_lower']*100:.2f}%, {summ['ci_upper']*100:.2f}%]")

    raw_rows = []
    for s in valid_subsets:
        for m, values in boot_per_subset[s].items():
            for i, v in enumerate(values):
                raw_rows.append({'iteration': i, 'metric_type': '2afc', 'subset': s, 'method': m, 'metric': 'accuracy', 'value': v})
    for m, values in boot_overall.items():
        for i, v in enumerate(values):
            raw_rows.append({'iteration': i, 'metric_type': '2afc', 'subset': 'OVERALL', 'method': m, 'metric': 'accuracy', 'value': v})

    return {
        'subsets_evaluated': valid_subsets,
        'point_estimate': {'per_subset': point, 'overall': point_overall},
        'bootstrap': {'per_subset': summary_per_subset, 'overall': summary_overall},
        'raw_rows': raw_rows,
    }


def run_jnd_bootstrap(args, device, simclr_model, lpips_model, n_bootstrap, ci, random_seed):
    """Point estimate + bootstrap for BAPPS JND (ROC-AUC, Acc@best-thresh), per subset and overall."""
    from eval_bapps_jnd import evaluate_simclr_jnd, evaluate_lpips_jnd, _auto_discover_subsets

    subsets = args.bapps_jnd_subsets
    if subsets is None:
        subsets = _auto_discover_subsets(args.bapps_jnd_root)

    per_subset = {}  # subset -> {'lpips': (scores, y_true), 'simclr': (scores, y_true)}
    point = {}  # subset -> {'lpips': {'roc_auc':..,'acc_best_t':..}, 'simclr': {...}}

    for subset in subsets:
        subset_dir = os.path.join(args.bapps_jnd_root, subset)
        if subset == ".":
            subset_dir = args.bapps_jnd_root
        if not os.path.isdir(subset_dir):
            print(f"[JND][WARN] Skipping missing subset: {subset_dir}")
            continue
        per_subset[subset] = {}
        point[subset] = {}

        if args.eval_lpips:
            m, scores, y_true = evaluate_lpips_jnd(
                subset_dir, device=device, batch_size=args.batch_size, num_workers=args.num_workers,
                lpips_net=args.lpips_net, lpips_size=args.lpips_size,
                positive_label=args.jnd_positive_label, judge_threshold=args.jnd_judge_threshold,
                p0_dir=args.jnd_p0_dir, p1_dir=args.jnd_p1_dir, same_dir=args.jnd_same_dir,
                same_exts=args.jnd_same_exts, return_raw=True,
                lpips_model=lpips_model,
            )
            per_subset[subset]['lpips'] = (scores, y_true)
            point[subset]['lpips'] = m
            print(f"[JND] LPIPS[{args.lpips_net}] {subset:>16}: AUC={m['roc_auc']*100:.2f}%, "
                  f"Acc@best={m['acc_best_t']*100:.2f}% (n={len(scores)})")

        if args.eval_simclr and simclr_model is not None:
            m, scores, y_true = evaluate_simclr_jnd(
                subset_dir, model=simclr_model, device=device, input_size=args.simclr_input_size,
                batch_size=args.batch_size, num_workers=args.num_workers, distance=args.simclr_distance,
                normalize=(not args.no_simclr_normalize),
                positive_label=args.jnd_positive_label, judge_threshold=args.jnd_judge_threshold,
                p0_dir=args.jnd_p0_dir, p1_dir=args.jnd_p1_dir, same_dir=args.jnd_same_dir,
                same_exts=args.jnd_same_exts, return_raw=True,
            )
            tag = f"simclr_{args.simclr_distance}{'_unnorm' if args.no_simclr_normalize else ''}"
            per_subset[subset]['simclr'] = (scores, y_true)
            point[subset]['simclr'] = m
            print(f"[JND] PhenoCLR[{args.simclr_distance}] {subset:>16}: AUC={m['roc_auc']*100:.2f}%, "
                  f"Acc@best={m['acc_best_t']*100:.2f}% (n={len(scores)})")

    methods = sorted({meth for d in per_subset.values() for meth in d.keys()})
    valid_subsets = list(per_subset.keys())

    point_overall = {}
    for meth in methods:
        aucs = [point[s][meth]['roc_auc'] for s in valid_subsets if meth in point[s]]
        accs = [point[s][meth]['acc_best_t'] for s in valid_subsets if meth in point[s]]
        point_overall[meth] = {'roc_auc': float(np.mean(aucs)) if aucs else np.nan,
                                'acc_best_t': float(np.mean(accs)) if accs else np.nan}

    print(f"\n[JND] Running {n_bootstrap} bootstrap replicates over pre-computed scores "
          f"(vectorized, no re-inference)...")
    boot_per_subset = {s: {} for s in valid_subsets}
    for s_i, s in enumerate(tqdm(valid_subsets)):
        for meth_i, meth in enumerate(sorted(per_subset[s].keys())):
            scores, y_true = per_subset[s][meth]
            seed = _derive_seed(random_seed, s_i, meth_i)
            aucs, accs = bootstrap_jnd_vectorized(scores, y_true, n_bootstrap, seed)
            boot_per_subset[s][meth] = {'roc_auc': aucs, 'acc_best_t': accs}

    boot_overall = {
        meth: {
            'roc_auc': np.mean([boot_per_subset[s][meth]['roc_auc'] for s in valid_subsets if meth in boot_per_subset[s]], axis=0),
            'acc_best_t': np.mean([boot_per_subset[s][meth]['acc_best_t'] for s in valid_subsets if meth in boot_per_subset[s]], axis=0),
        }
        for meth in methods
    }

    summary_per_subset = {
        s: {meth: {k: summarize_bootstrap(v, ci=ci) for k, v in mdict.items()} for meth, mdict in d.items()}
        for s, d in boot_per_subset.items()
    }
    summary_overall = {meth: {k: summarize_bootstrap(v, ci=ci) for k, v in d.items()} for meth, d in boot_overall.items()}

    for s in valid_subsets:
        for meth, mdict in summary_per_subset[s].items():
            auc_s, acc_s = mdict['roc_auc'], mdict['acc_best_t']
            print(f"[JND] {s} [{meth}]: AUC={auc_s['mean']*100:.2f}% +/- {auc_s['std']*100:.2f}% "
                  f"[{auc_s['ci_lower']*100:.2f}%, {auc_s['ci_upper']*100:.2f}%]")
    for meth, mdict in summary_overall.items():
        auc_s = mdict['roc_auc']
        print(f"[JND] OVERALL [{meth}]: AUC={auc_s['mean']*100:.2f}% +/- {auc_s['std']*100:.2f}% "
              f"[{auc_s['ci_lower']*100:.2f}%, {auc_s['ci_upper']*100:.2f}%]")

    raw_rows = []
    for s in valid_subsets:
        for meth, mdict in boot_per_subset[s].items():
            for metric, values in mdict.items():
                for i, v in enumerate(values):
                    raw_rows.append({'iteration': i, 'metric_type': 'jnd', 'subset': s, 'method': meth, 'metric': metric, 'value': v})
    for meth, mdict in boot_overall.items():
        for metric, values in mdict.items():
            for i, v in enumerate(values):
                raw_rows.append({'iteration': i, 'metric_type': 'jnd', 'subset': 'OVERALL', 'method': meth, 'metric': metric, 'value': v})

    return {
        'subsets_evaluated': valid_subsets,
        'point_estimate': {'per_subset': point, 'overall': point_overall},
        'bootstrap': {'per_subset': summary_per_subset, 'overall': summary_overall},
        'raw_rows': raw_rows,
    }


# =============================================================================
# Orchestration
# =============================================================================

def build_manuscript_report(ci, n_bootstrap, embedding_results, afc_results, jnd_results):
    """Compact mean +/- std [CI] strings for the metrics used in the manuscript table."""
    ci_pct = int(round(ci * 100))
    report = {'n_bootstrap': n_bootstrap, 'ci_percent': ci_pct}

    if embedding_results is not None:
        sil = embedding_results['bootstrap']['silhouette_analysis']
        var = embedding_results['bootstrap']['variance_explained']
        report['silhouette'] = {
            metric: f"{s['mean']:.4f} +/- {s['std']:.4f} [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
            for metric, s in sil.items()
        }
        report['variance_explained'] = {
            name: f"{s['mean']:.1f}% +/- {s['std']:.1f}% [{s['ci_lower']:.1f}%, {s['ci_upper']:.1f}%]"
            for name, s in var.items()
        }

    if afc_results is not None:
        report['bapps_2afc'] = {
            'overall': {
                m: f"{s['mean']*100:.2f}% +/- {s['std']*100:.2f}% [{s['ci_lower']*100:.2f}%, {s['ci_upper']*100:.2f}%]"
                for m, s in afc_results['bootstrap']['overall'].items()
            },
            'per_subset': {
                subset: {
                    m: f"{s['mean']*100:.2f}% +/- {s['std']*100:.2f}% [{s['ci_lower']*100:.2f}%, {s['ci_upper']*100:.2f}%]"
                    for m, s in mdict.items()
                }
                for subset, mdict in afc_results['bootstrap']['per_subset'].items()
            }
        }

    if jnd_results is not None:
        def fmt_jnd(mdict):
            return {
                meth: {
                    'roc_auc': f"{d['roc_auc']['mean']*100:.2f}% +/- {d['roc_auc']['std']*100:.2f}% "
                               f"[{d['roc_auc']['ci_lower']*100:.2f}%, {d['roc_auc']['ci_upper']*100:.2f}%]",
                    'acc_best_t': f"{d['acc_best_t']['mean']*100:.2f}% +/- {d['acc_best_t']['std']*100:.2f}% "
                                  f"[{d['acc_best_t']['ci_lower']*100:.2f}%, {d['acc_best_t']['ci_upper']*100:.2f}%]",
                }
                for meth, d in mdict.items()
            }

        report['bapps_jnd'] = {
            'overall': fmt_jnd(jnd_results['bootstrap']['overall']),
            'per_subset': {s: fmt_jnd(mdict) for s, mdict in jnd_results['bootstrap']['per_subset'].items()},
        }

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap-evaluate embedding-space metrics (silhouette / variance explained) '
                    'and BAPPS perceptual metrics (2AFC / JND) with matching statistical errors.'
    )

    # --- Shared bootstrap settings ---
    parser.add_argument('--output', required=True, help='Directory to save results')
    parser.add_argument('--random-seed', type=int, default=42, help='Base random seed for reproducibility')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap resamples')
    parser.add_argument('--ci', type=float, default=0.95, help='Confidence interval level (e.g. 0.95 for 95%% CI)')

    # --- Embedding-space (silhouette / variance) settings ---
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip the silhouette/variance analysis')
    parser.add_argument('--embeddings', help='Path to CSV file containing embeddings')
    parser.add_argument('--num-samples', type=int, help='Number of random samples to analyze (default: use all)')
    parser.add_argument('--with-replacement', action='store_true',
                         help='Resample embeddings with replacement (classic bootstrap). WARNING: for '
                              'distance-based metrics like silhouette score this creates duplicate points '
                              'that inflate the score. Default is subsampling without replacement instead.')
    parser.add_argument('--subsample-frac', type=float, default=0.8,
                         help='Fraction of samples to draw (without replacement) per replicate for the '
                              'embedding-space bootstrap when --with-replacement is not set.')

    # --- Perceptual (BAPPS) settings ---
    parser.add_argument('--run-perceptual', action='store_true', help='Run BAPPS 2AFC/JND bootstrap evaluation')
    parser.add_argument('--n-bootstrap-perceptual', type=int, default=None,
                         help='Override --n-bootstrap for the (cheap, numpy-only) perceptual bootstrap. '
                              'Defaults to --n-bootstrap.')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'])

    parser.add_argument('--eval-lpips', action='store_true', help='Evaluate LPIPS baseline')
    parser.add_argument('--lpips-net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'])
    parser.add_argument('--lpips-size', type=int, default=256)

    parser.add_argument('--eval-simclr', action='store_true', help='Evaluate PhenoCLR embeddings')
    parser.add_argument('--simclr-checkpoint', type=str, default=None)
    parser.add_argument('--simclr-config', type=str, default=None)
    parser.add_argument('--simclr-module', type=str, default='train.simclr_birdcolour_multispectral')
    parser.add_argument('--simclr-input-size', type=int, default=224)
    parser.add_argument('--simclr-distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--no-simclr-normalize', action='store_true')

    parser.add_argument('--bapps-2afc-root', type=str, default=None,
                         help='Path to BAPPS 2AFC root, e.g. .../PerceptualSimilarity/dataset/2afc')
    parser.add_argument('--bapps-2afc-subsets', nargs='+', default=[
        'val/traditional', 'val/cnn', 'val/superres', 'val/deblur', 'val/color', 'val/frameinterp'
    ])

    parser.add_argument('--bapps-jnd-root', type=str, default=None,
                         help='Path to BAPPS JND root, e.g. .../PerceptualSimilarity/dataset/jnd')
    parser.add_argument('--bapps-jnd-subsets', nargs='+', default=None,
                         help='Subsets to evaluate (relative to --bapps-jnd-root). If omitted, auto-discover.')

    parser.add_argument('--jnd-positive-label', type=str, default='different', choices=['different', 'same'])
    parser.add_argument('--jnd-judge-threshold', type=float, default=0.5)
    parser.add_argument('--jnd-p0-dir', type=str, default='p0')
    parser.add_argument('--jnd-p1-dir', type=str, default='p1')
    parser.add_argument('--jnd-same-dir', type=str, default='same')
    parser.add_argument('--jnd-same-exts', type=str, nargs='+', default=['npy'])

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    embedding_results = None
    if not args.skip_embeddings:
        if not args.embeddings:
            parser.error('--embeddings is required unless --skip-embeddings is set')
        embedding_results = run_embedding_bootstrap(
            args.embeddings, args.num_samples, args.random_seed, args.n_bootstrap, args.ci,
            args.with_replacement, args.subsample_frac
        )

    afc_results = None
    jnd_results = None
    if args.run_perceptual:
        from eval_bapps_jnd import get_device, load_simclr_model, build_lpips_model

        device = get_device(args.device)
        print(f"\n[Perceptual] Using device: {device}")

        # Load each model exactly once and reuse it across every 2AFC/JND subset,
        # instead of re-loading checkpoint/network weights from disk per subset.
        simclr_model = None
        if args.eval_simclr:
            assert args.simclr_checkpoint is not None, '--simclr-checkpoint is required when --eval-simclr is set'
            simclr_model = load_simclr_model(
                args.simclr_checkpoint, args.simclr_config, device, module_path=args.simclr_module
            )

        lpips_model = None
        if args.eval_lpips:
            lpips_model = build_lpips_model(args.lpips_net, device)

        n_boot_perceptual = args.n_bootstrap_perceptual or args.n_bootstrap

        if args.bapps_2afc_root:
            afc_results = run_2afc_bootstrap(
                args, device, simclr_model, lpips_model, n_boot_perceptual, args.ci, args.random_seed
            )
        else:
            print('[2AFC] Skipped: no --bapps-2afc-root provided')

        if args.bapps_jnd_root:
            jnd_results = run_jnd_bootstrap(
                args, device, simclr_model, lpips_model, n_boot_perceptual, args.ci, args.random_seed
            )
        else:
            print('[JND] Skipped: no --bapps-jnd-root provided')

    # --- Save full summary JSON ---
    summary = {
        'settings': {
            'n_bootstrap': args.n_bootstrap,
            'n_bootstrap_perceptual': args.n_bootstrap_perceptual or args.n_bootstrap,
            'confidence_level': args.ci,
            'random_seed': args.random_seed,
        }
    }
    if embedding_results is not None:
        summary['embeddings'] = {
            'metadata': embedding_results['metadata'],
            'point_estimate': embedding_results['point_estimate'],
            'bootstrap': embedding_results['bootstrap'],
        }
    if afc_results is not None:
        summary['bapps_2afc'] = {
            'subsets_evaluated': afc_results['subsets_evaluated'],
            'point_estimate': afc_results['point_estimate'],
            'bootstrap': afc_results['bootstrap'],
        }
    if jnd_results is not None:
        summary['bapps_jnd'] = {
            'subsets_evaluated': jnd_results['subsets_evaluated'],
            'point_estimate': jnd_results['point_estimate'],
            'bootstrap': jnd_results['bootstrap'],
        }

    summary_file = os.path.join(args.output, 'full_metrics_bootstrap_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary results saved to {summary_file}")

    # --- Manuscript-ready compact report ---
    report = build_manuscript_report(args.ci, args.n_bootstrap, embedding_results, afc_results, jnd_results)
    report_file = os.path.join(args.output, 'full_metrics_bootstrap_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Manuscript-ready report saved to {report_file}")

    # --- Raw per-iteration values, all metrics combined for custom stats/tables ---
    raw_rows = []
    if embedding_results is not None:
        raw_rows.extend(embedding_results['raw_rows'])
    if afc_results is not None:
        raw_rows.extend(afc_results['raw_rows'])
    if jnd_results is not None:
        raw_rows.extend(jnd_results['raw_rows'])

    raw_df = pd.DataFrame(raw_rows)
    raw_file = os.path.join(args.output, 'full_metrics_bootstrap_raw.csv')
    raw_df.to_csv(raw_file, index=False)
    print(f"Raw bootstrap values saved to {raw_file}")


if __name__ == '__main__':
    main()
