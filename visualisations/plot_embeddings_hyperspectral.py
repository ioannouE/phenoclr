import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.offsetbox as offsetbox
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Make sure we can import read_iml_hyp from simclr/lepidoptera
LEPI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lepidoptera')
if LEPI_DIR not in sys.path:
    sys.path.append(LEPI_DIR)

from read_iml_hyp import read_iml_hyp  # noqa: E402


def load_embeddings_from_csv(embeddings_path):
    """Load embeddings from CSV file.
    If the first column looks like filenames, return (embeddings, filenames), else (embeddings, None).
    """
    df = pd.read_csv(embeddings_path)
    if df.columns[0].lower() in ['filename', 'file', 'name', 'image']:
        filenames = df.iloc[:, 0].values
        embeddings = df.iloc[:, 1:].values
    else:
        filenames = None
        embeddings = df.values
    return embeddings, filenames


def bil_matches_header(base_path):
    """Return True if the .bil file size matches the dimensions declared in its .bil.hdr.

    Several cubes in the dataset are truncated (the .bil is much smaller than
    lines*samples*bands*2 bytes). The reader silently fabricates dimensions for
    those, producing scrambled, striped thumbnails. We use this check to skip them.
    """
    hdr_path = base_path + '.bil.hdr'
    bil_path = base_path + '.bil'
    try:
        lines = samples = bands = None
        with open(hdr_path, 'r') as f:
            for line in f:
                stripped = line.strip().lower()
                if '=' not in line:
                    continue
                if stripped.startswith('lines'):
                    lines = int(line.split('=')[1].strip())
                elif stripped.startswith('samples'):
                    samples = int(line.split('=')[1].strip())
                elif stripped.startswith('bands'):
                    bands = int(line.split('=')[1].strip())
        if lines is None or samples is None or bands is None:
            return False
        expected_bytes = lines * samples * bands * 2  # uint16
        return os.path.getsize(bil_path) == expected_bytes
    except Exception:
        return False


def find_hyperspectral_base(data_dir, filename_base):
    """Find the base path to a hyperspectral .bil file that has a matching .bil.hdr.
    - filename_base: may include an extension; we strip it and search for exact matches.
    Returns: base_path (without extension) or None.
    """
    base = os.path.splitext(filename_base)[0]

    # Try exact path
    direct = os.path.join(data_dir, base)
    if os.path.exists(direct + '.bil') and os.path.exists(direct + '.bil.hdr'):
        return direct

    # Recursive exact filename search
    patterns = [
        os.path.join(data_dir, '**', base + '.bil'),
        os.path.join(data_dir, '**', base + '.BIL'),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        for m in matches:
            b = m[:-4]  # strip .bil
            if os.path.exists(b + '.bil.hdr'):
                return b

    # If nothing exact, give up (avoid fuzzy matches that can mismatch)
    return None


def select_rgb_indices(wavelengths, targets=(650, 550, 450)):
    """Select band indices closest to the given wavelength targets (in nm)."""
    indices = []
    for t in targets:
        idx = int(np.argmin(np.abs(wavelengths - t)))
        indices.append(idx)
    return indices


def downsample_if_large(image_hw_c, max_dim=1000):
    """Quick stride downsample if H or W exceeds max_dim to save memory/time."""
    H, W, C = image_hw_c.shape
    if H <= max_dim and W <= max_dim:
        return image_hw_c
    h_factor = max(1, H // max_dim)
    w_factor = max(1, W // max_dim)
    factor = max(h_factor, w_factor)
    return image_hw_c[::factor, ::factor, :]


def load_hyperspectral_rgb(base_path,
                           wavelength_targets=(650, 550, 450),
                           transparent_bg=False,
                           alpha_threshold=0.03,
                           norm_foreground_only=False,
                           downsample_max_dim=1000):
    """Load hyperspectral BIL (via base_path) and convert to RGB or RGBA array.
    - Uses wavelengths to pick closest bands to wavelength_targets; falls back to first 3 bands if not available.
    - transparent_bg: if True, compute alpha from intensity threshold over RGB.
    - alpha_threshold: threshold in [0,1] applied to normalized intensity to define foreground.
    - norm_foreground_only: normalize R,G,B using only foreground pixels to avoid washed-out backgrounds.
    - downsample_max_dim: stride-downsample large images before thumbnailing.
    Returns: np.uint8 array (H, W, 3) or (H, W, 4) if transparent_bg.
    """
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)

    # Read cube
    image, wavelengths, scan, _ = read_iml_hyp(directory, filename)  # image: (H, W, C)
    image = downsample_if_large(image, max_dim=downsample_max_dim)

    H, W, C = image.shape

    # Choose RGB bands
    if wavelengths is not None and len(wavelengths) == C:
        rgb_idxs = select_rgb_indices(wavelengths, wavelength_targets)
    else:
        rgb_idxs = [0, 1, 2] if C >= 3 else list(range(min(3, C)))
        if len(rgb_idxs) < 3:
            print(f"Warning: {base_path} has only {C} bands; repeating last band to make RGB")
            while len(rgb_idxs) < 3:
                rgb_idxs.append(rgb_idxs[-1])

    rgb_f = np.stack([image[:, :, i].astype(np.float32) for i in rgb_idxs], axis=-1)  # (H, W, 3)

    # Build a rough foreground mask from raw intensity (before normalization)
    fg_mask = None
    if transparent_bg or norm_foreground_only:
        intensity = np.mean(rgb_f, axis=2)
        # Normalize intensity to [0,1] to apply threshold consistently per image
        i_min = float(np.min(intensity))
        i_max = float(np.max(intensity))
        if i_max > i_min:
            intensity_n = (intensity - i_min) / (i_max - i_min)
        else:
            intensity_n = np.zeros_like(intensity, dtype=np.float32)
        fg_mask = intensity_n > float(alpha_threshold)

    # Per-band min/max, optionally using only foreground
    norm = np.zeros_like(rgb_f, dtype=np.float32)
    for c in range(3):
        band = rgb_f[:, :, c]
        if norm_foreground_only and (fg_mask is not None) and np.any(fg_mask):
            vals = band[fg_mask]
            bmin = float(np.min(vals))
            bmax = float(np.max(vals))
        else:
            bmin = float(np.min(band))
            bmax = float(np.max(band))
        if bmax > bmin:
            norm[:, :, c] = (band - bmin) / (bmax - bmin)
        else:
            norm[:, :, c] = 0.0

    norm = np.clip(norm, 0.0, 1.0)
    rgb_u8 = (norm * 255).astype(np.uint8)

    if transparent_bg and (fg_mask is not None):
        alpha = np.where(fg_mask, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb_u8, alpha])
        return rgba
    else:
        return rgb_u8


def get_hyperspectral_thumbnail(args):
    """Load and resize a hyperspectral image to create an RGB or RGBA thumbnail.
    args: (base_path, size, transparent_bg, alpha_threshold, norm_foreground_only, wavelength_targets, downsample_max_dim)
    """
    (base_path, size, transparent_bg, alpha_threshold, norm_foreground_only, wavelength_targets, downsample_max_dim) = args

    if not (os.path.exists(base_path + '.bil') and os.path.exists(base_path + '.bil.hdr')):
        return None

    try:
        arr = load_hyperspectral_rgb(
            base_path,
            wavelength_targets=wavelength_targets,
            transparent_bg=transparent_bg,
            alpha_threshold=alpha_threshold,
            norm_foreground_only=norm_foreground_only,
            downsample_max_dim=downsample_max_dim,
        )
        if arr is None or arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return None
        mode = 'RGBA' if arr.shape[2] == 4 else 'RGB'
        img = Image.fromarray(arr, mode=mode)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error creating thumbnail for {base_path}: {e}")
        return None


def plot_embeddings_with_hyperspectral_images(embeddings, filenames, data_dir, output_path, method='umap',
                                              thumb_size=100, max_images=1000, random_state=42, grid_scale=1.0,
                                              transparent_bg=False, alpha_threshold=0.03, norm_foreground_only=False,
                                              wavelength_targets=(650, 550, 450), fig_transparent=False,
                                              downsample_max_dim=1000):
    """
    Create and save a 2D visualization using RGB/RGBA thumbnails converted from hyperspectral BIL images.
    """
    print(f"Reducing dimensionality using {method.upper()}...")

    # Scale the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Reduce dimensionality
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings_scaled)
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings_scaled)

    # Random sampling if too many images
    if len(filenames) > max_images:
        print(f"Sampling {max_images} images from {len(filenames)} total images...")
        indices = np.random.choice(len(filenames), max_images, replace=False)
        embedding_2d = embedding_2d[indices]
        filenames = filenames[indices]

    # Find base paths
    print("Finding hyperspectral base paths (.bil + .bil.hdr)...")
    base_paths = []
    valid_indices = []
    skipped_missing = 0
    skipped_mismatch = 0
    for i, fname in enumerate(tqdm(filenames, desc="Finding files")):
        base = find_hyperspectral_base(data_dir, fname)
        if base is None:
            print(f"Skipping {fname} - .bil/.bil.hdr not found")
            skipped_missing += 1
            continue
        if not bil_matches_header(base):
            print(f"Skipping {fname} - .bil size does not match header dimensions (truncated/corrupt cube)")
            skipped_mismatch += 1
            continue
        base_paths.append((base, (thumb_size, thumb_size), transparent_bg, alpha_threshold, norm_foreground_only, wavelength_targets, downsample_max_dim))
        valid_indices.append(i)

    if skipped_missing or skipped_mismatch:
        print(f"Skipped {skipped_missing} missing and {skipped_mismatch} size-mismatched cubes")

    # Update arrays to only keep found images
    if not base_paths:
        print("Error: No valid hyperspectral files found!")
        return

    embedding_2d = embedding_2d[valid_indices]
    filenames = [filenames[i] for i in valid_indices]

    # Load thumbnails in parallel
    print("Loading and converting hyperspectral images to thumbnails...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        thumbnails = list(tqdm(
            executor.map(get_hyperspectral_thumbnail, base_paths),
            total=len(base_paths),
            desc="Loading images"
        ))

    # Filter failed loads
    final_valid_indices = [i for i, thumb in enumerate(thumbnails) if thumb is not None]
    thumbnails = [thumbnails[i] for i in final_valid_indices]
    embedding_2d = embedding_2d[final_valid_indices]

    if len(thumbnails) == 0:
        print("Error: No thumbnails could be created!")
        return

    print(f"Successfully created {len(thumbnails)} thumbnails")
    print("Creating visualization...")

    # Create the figure
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)

    # Calculate figure bounds
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()

    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Create grid for overlap detection
    grid_size = (thumb_size / 100) * grid_scale
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)
    occupied_grid = np.zeros((len(x_grid) - 1, len(y_grid) - 1), dtype=bool)

    # Plot thumbnails with grid-based overlap detection
    print("Placing images on plot...")
    for idx, (x0, y0) in enumerate(tqdm(embedding_2d, desc="Plotting images")):
        x_idx = np.searchsorted(x_grid, x0) - 1
        y_idx = np.searchsorted(y_grid, y0) - 1

        if 0 <= x_idx < occupied_grid.shape[0] and 0 <= y_idx < occupied_grid.shape[1]:
            if occupied_grid[x_idx, y_idx]:
                continue
            occupied_grid[x_idx, y_idx] = True

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(thumbnails[idx], zoom=0.5),
                (x0, y0),
                frameon=False,
                pad=0.0,
            )
            ax.add_artist(imagebox)

    plt.title(f'2D {method.upper()} projection with hyperspectral thumbnails (RGB from wavelengths)')
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    plt.grid(True, alpha=0.3)

    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False if not fig_transparent else True)
    plt.close()
    print("Saved successfully!")


def parse_wavelength_targets(s):
    try:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        vals = tuple(int(float(p)) for p in parts)
        if len(vals) != 3:
            raise ValueError
        return vals
    except Exception:
        raise ValueError("--wavelength_targets must be a comma-separated list of 3 numbers, e.g., '650,550,450'")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Generate UMAP/t-SNE visualizations with hyperspectral RGB(A) thumbnails')
    parser.add_argument('--embeddings_csv', type=str, required=True,
                        help='Path to the CSV file containing embeddings (first column should be filenames)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the hyperspectral .bil/.bil.hdr files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the visualizations')
    parser.add_argument('--method', type=str, choices=['umap', 'tsne', 'both'], default='both',
                        help='Dimensionality reduction method to use')
    parser.add_argument('--thumb_size', type=int, default=100,
                        help='Size of thumbnail images in pixels')
    parser.add_argument('--max_images', type=int, default=600,
                        help='Maximum number of images to plot')
    parser.add_argument('--grid_scale', type=float, default=1.0,
                        help='Scale factor for grid size (lower = more dense images)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--transparent_bg', action='store_true', default=False,
                        help='Make thumbnails RGBA using an intensity threshold for alpha')
    parser.add_argument('--alpha_threshold', type=float, default=0.03,
                        help='Threshold in [0,1] on normalized intensity to define foreground for alpha mask')
    parser.add_argument('--norm_foreground_only', action='store_true', default=False,
                        help='Normalize RGB using only foreground pixels (requires --transparent_bg)')
    parser.add_argument('--wavelength_targets', type=parse_wavelength_targets, default=(650, 550, 450),
                        help="Comma-separated wavelengths for R,G,B (e.g., '650,550,450'; typical RGB mapping)")
    parser.add_argument('--fig_transparent', action='store_true', default=False,
                        help='Save the figure canvas with a transparent background')
    parser.add_argument('--downsample_max_dim', type=int, default=1000,
                        help='Stride-downsample large images to this max dimension to save memory')
    return parser.parse_args()


def main(args):
    print(f"Loading embeddings from {args.embeddings_csv}...")
    embeddings, filenames = load_embeddings_from_csv(args.embeddings_csv)
    print(f"Loaded {len(embeddings)} embeddings")

    if filenames is None:
        print("Error: No filenames found in embeddings CSV. Please ensure the first column contains filenames.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method in ['umap', 'both']:
        out_path = os.path.join(args.output_dir, 'umap_hyperspectral_visualization.png')
        plot_embeddings_with_hyperspectral_images(
            embeddings, filenames, args.data_dir, out_path,
            method='umap', thumb_size=args.thumb_size, max_images=args.max_images,
            grid_scale=args.grid_scale, random_state=args.random_state,
            transparent_bg=args.transparent_bg, alpha_threshold=args.alpha_threshold,
            norm_foreground_only=args.norm_foreground_only, wavelength_targets=args.wavelength_targets,
            fig_transparent=args.fig_transparent, downsample_max_dim=args.downsample_max_dim,
        )

    if args.method in ['tsne', 'both']:
        out_path = os.path.join(args.output_dir, 'tsne_hyperspectral_visualization.png')
        plot_embeddings_with_hyperspectral_images(
            embeddings, filenames, args.data_dir, out_path,
            method='tsne', thumb_size=args.thumb_size, max_images=args.max_images,
            grid_scale=args.grid_scale, random_state=args.random_state,
            transparent_bg=args.transparent_bg, alpha_threshold=args.alpha_threshold,
            norm_foreground_only=args.norm_foreground_only, wavelength_targets=args.wavelength_targets,
            fig_transparent=args.fig_transparent, downsample_max_dim=args.downsample_max_dim,
        )


if __name__ == '__main__':
    args = get_args()
    main(args)
