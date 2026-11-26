import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import os
import json
import argparse
from tqdm import tqdm

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

    # Try Format 1: Genus_species_id_SEX...
    if len(parts) >= 4 and parts[3] in ['M', 'F', 'U']:
        species = f"{parts[0]}_{parts[1]}"
        sex = parts[3]
        return species, sex

    # Try Format 2: SpeciesName_ID1_ID2
    if len(parts) > 2:
        species = "_".join(parts[:-2])
        # No reliable sex info in this format
        sex = None
        return species, sex

    # Fallback for unexpected formats
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
    valid_classes = []

    for label in unique_labels:
        if np.sum(labels == label) >= min_samples:
            valid_classes.append(label)

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

def bootstrap_replicate(embeddings, species_labels, sex_labels, class_labels, seed,
                         min_samples_per_class=2, replacement=False, subsample_frac=0.8):
    """
    Resample the dataset and recompute metrics on the replicate.

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
        embeddings[indices],
        species_labels[indices],
        sex_labels[indices],
        class_labels[indices],
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

def analyze_embeddings(embeddings_csv, output_dir, num_samples=None, random_seed=42,
                        n_bootstrap=1000, ci=0.95, replacement=False, subsample_frac=0.8):
    """Analyze PhenoCLR embeddings and calculate metrics with bootstrap-based statistical variance."""

    min_samples_per_class = 2

    # Load embeddings
    print("\nLoading embeddings...")
    df = pd.read_csv(embeddings_csv)
    embeddings = df.iloc[:, 1:].values  # First column is filename
    filenames = df.iloc[:, 0].values

    if num_samples is not None and num_samples < len(embeddings):
        print(f"\nSampling {num_samples} embeddings...")
        np.random.seed(random_seed)
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings = embeddings[indices]
        filenames = filenames[indices]
    else:
        print(f"\nUsing all {len(embeddings)} samples...")

    # Extract metadata
    print("\nExtracting metadata...")
    species_labels, sex_labels, class_labels = [], [], []
    for filename in tqdm(filenames):
        species, sex = extract_metadata_from_filename(filename)
        species_labels.append(species)
        sex_labels.append(sex if sex else 'U')
        class_labels.append(f"{species}_{sex}" if sex else species)

    species_labels = np.array(species_labels)
    sex_labels = np.array(sex_labels)
    class_labels = np.array(class_labels)

    print("\nDataset statistics:")
    print(f"Total samples: {len(embeddings)}")
    print(f"Total unique species: {len(np.unique(species_labels))}")
    print(f"Total unique species_sex classes: {len(np.unique(class_labels))}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Point estimate on the full (unresampled) dataset
    print("\nCalculating point estimates on full dataset...")
    point_silhouette, point_variance, num_valid_classes, num_filtered = compute_point_estimates(
        embeddings, species_labels, sex_labels, class_labels, min_samples_per_class
    )
    for metric, score in point_silhouette.items():
        print(f"Point estimate silhouette ({metric}): {score:.4f}")
    for name, score in point_variance.items():
        print(f"Point estimate {name}: {score:.1f}%")

    # Bootstrap replicates
    resample_desc = "with replacement (full size)" if replacement else f"without replacement ({subsample_frac:.0%} subsample)"
    print(f"\nRunning {n_bootstrap} bootstrap replicates, sampling {resample_desc}...")
    boot_silhouette = {'euclidean_distance': [], 'cosine_distance': []}
    boot_variance = {'species_variance': [], 'sex_variance': [], 'class_variance': []}

    for i in tqdm(range(n_bootstrap)):
        seed = random_seed + i
        sil, var = bootstrap_replicate(
            embeddings, species_labels, sex_labels, class_labels, seed, min_samples_per_class,
            replacement=replacement, subsample_frac=subsample_frac
        )
        for k, v in sil.items():
            boot_silhouette[k].append(v)
        for k, v in var.items():
            boot_variance[k].append(v)

    # Summarize bootstrap distributions
    silhouette_summary = {k: summarize_bootstrap(v, ci=ci) for k, v in boot_silhouette.items()}
    variance_summary = {k: summarize_bootstrap(v, ci=ci) for k, v in boot_variance.items()}

    print(f"\nBootstrap results ({n_bootstrap} resamples, {int(ci * 100)}% CI):")
    for metric, summary in silhouette_summary.items():
        print(f"Silhouette ({metric}): {summary['mean']:.4f} +/- {summary['std']:.4f} "
              f"[{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]")
    for name, summary in variance_summary.items():
        print(f"{name}: {summary['mean']:.1f}% +/- {summary['std']:.1f}% "
              f"[{summary['ci_lower']:.1f}%, {summary['ci_upper']:.1f}%]")

    # Save summary JSON
    results = {
        'metadata': {
            'num_samples': len(embeddings),
            'num_species': len(np.unique(species_labels)),
            'num_sexes': len(np.unique(sex_labels)),
            'num_classes': len(np.unique(class_labels)),
            'num_valid_classes': num_valid_classes,
            'min_samples_per_class': min_samples_per_class,
            'samples_after_filtering': num_filtered,
            'embedding_dim': embeddings.shape[1],
            'random_seed': random_seed,
            'sampled_data': num_samples is not None,
            'n_bootstrap': n_bootstrap,
            'confidence_level': ci,
            'resample_with_replacement': replacement,
            'subsample_frac': subsample_frac if not replacement else None
        },
        'point_estimate': {
            'silhouette_analysis': point_silhouette,
            'variance_explained': point_variance
        },
        'bootstrap': {
            'silhouette_analysis': silhouette_summary,
            'variance_explained': variance_summary
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'embedding_metrics_bootstrap_summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSummary results saved to {output_file}")

    # Save a compact, manuscript-ready JSON with just mean +/- std [CI] strings
    ci_pct = int(round(ci * 100))
    report_results = {
        'n_bootstrap': n_bootstrap,
        'ci_percent': ci_pct,
        'silhouette': {
            metric: f"{s['mean']:.4f} +/- {s['std']:.4f} [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
            for metric, s in silhouette_summary.items()
        },
        'variance_explained': {
            name: f"{s['mean']:.1f}% +/- {s['std']:.1f}% [{s['ci_lower']:.1f}%, {s['ci_upper']:.1f}%]"
            for name, s in variance_summary.items()
        }
    }
    report_file = os.path.join(output_dir, 'embedding_metrics_bootstrap_report.json')
    with open(report_file, 'w') as f:
        json.dump(report_results, f, indent=4)
    print(f"Manuscript-ready report saved to {report_file}")

    # Save raw per-iteration values (useful for manuscript tables / custom stats)
    raw_rows = []
    for i in range(n_bootstrap):
        for metric, values in boot_silhouette.items():
            raw_rows.append({'iteration': i, 'metric_type': 'silhouette', 'metric': metric, 'value': values[i]})
        for name, values in boot_variance.items():
            raw_rows.append({'iteration': i, 'metric_type': 'variance_explained', 'metric': name, 'value': values[i]})

    raw_df = pd.DataFrame(raw_rows)
    raw_file = os.path.join(output_dir, 'embedding_metrics_bootstrap_raw.csv')
    raw_df.to_csv(raw_file, index=False)
    print(f"Raw bootstrap values saved to {raw_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze PhenoCLR embeddings with bootstrap statistical variance')
    parser.add_argument('--embeddings', help='Path to CSV file containing embeddings')
    parser.add_argument('--output', help='Directory to save results')
    parser.add_argument('--num-samples', type=int, help='Number of random samples to analyze. If not specified, uses all samples.')
    parser.add_argument('--random-seed', type=int, default=42, help='Base random seed for reproducibility')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap resamples')
    parser.add_argument('--ci', type=float, default=0.95, help='Confidence interval level (e.g. 0.95 for 95%% CI)')
    parser.add_argument('--with-replacement', action='store_true',
                         help='Resample with replacement (classic bootstrap). WARNING: for distance-based '
                              'metrics like silhouette score this creates duplicate points that inflate '
                              'the score. Default is subsampling without replacement instead.')
    parser.add_argument('--subsample-frac', type=float, default=0.8,
                         help='Fraction of samples to draw (without replacement) per replicate when '
                              '--with-replacement is not set. Ignored if --with-replacement is set.')

    args = parser.parse_args()
    analyze_embeddings(
        args.embeddings,
        args.output,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        replacement=args.with_replacement,
        subsample_frac=args.subsample_frac
    )
