import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.offsetbox as offsetbox
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import rasterio
try:
    import tifffile as tiff
except Exception:
    tiff = None
import glob
import re

# Extensions used when indexing data_dir and when resolving CSV filenames to paths.
# CSV rows may list .tif while previews on disk are .png (or vice versa); lookup is by stem.
_RASTER_EXTENSIONS = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
)

def _ext_key(path):
    return os.path.splitext(path)[1].lower()


def _csv_path_stem(filename):
    """Stem used for matching: basename only (CSV paths may include directories), normalized slashes."""
    s = str(filename).strip().replace("\\", "/")
    return os.path.splitext(os.path.basename(s))[0]


def _find_stem_prefix_match(stem, index):
    """If exact stem is missing, match index keys that extend the stem (e.g. CSV ..._Back vs disk ..._Back_Vis_G167935)."""
    if stem in index:
        return index[stem]
    prefix = stem + "_"
    matches = [k for k in index if k.startswith(prefix)]
    if not matches:
        return None
    if len(matches) == 1:
        return index[matches[0]]

    # RGB previews often use ..._Back_Vis_G<id> (see Segmented-Aves-Back-RGB-PNG-* exports).
    vis_pat = re.compile(re.escape(stem) + r"_Vis_G(\d+)$")
    vis_matches = []
    for k in matches:
        m = vis_pat.fullmatch(k)
        if m:
            vis_matches.append((int(m.group(1)), k))
    if vis_matches:
        vis_matches.sort(key=lambda t: t[0])
        chosen = vis_matches[0][1]
        if len(vis_matches) > 1:
            print(
                f"Warning: Multiple _Vis_G thumbnails for stem '{stem}' "
                f"(e.g. G={vis_matches[0][0]}, G={vis_matches[1][0]}, …) — using {chosen}"
            )
        return index[chosen]

    matches.sort()
    print(
        f"Warning: Ambiguous thumbnail stem for '{stem}': "
        f"{matches[:5]}{'...' if len(matches) > 5 else ''} — using {matches[0]}"
    )
    return index[matches[0]]


def _collect_rasters_under(top, ext_set):
    """Walk one subtree and return {stem: path} (first path wins per stem). Uses filesystem order (not sorted)."""
    out = {}
    top = os.path.abspath(top)
    for dirpath, _, filenames in os.walk(top, followlinks=False):
        for fname in filenames:
            stem, ext = os.path.splitext(fname)
            if ext.lower() in ext_set:
                out.setdefault(stem, os.path.join(dirpath, fname))
    return out


def _build_index_lower(index):
    index_lower = {}
    for stem, path in index.items():
        low = stem.lower()
        if low in index_lower:
            if index_lower[low] != path:
                index_lower[low] = None
        else:
            index_lower[low] = path
    return {k: v for k, v in index_lower.items() if v is not None}


def build_multispectral_index(data_dir, index_workers=-1):
    """Build stem -> path index and a case-insensitive helper (only unambiguous lowercase stems).

    index_workers:
      * 0 — single-threaded ``os.walk`` of the whole tree (legacy, deterministic traversal order).
      * -1 (default) — collect files in the top-level directory with ``scandir``; if there are
        multiple immediate subdirectories, walk each subtree in parallel (often faster on NFS).
      * N>0 — use at most N threads for those parallel subtree walks.

    Indexing does not sort filenames; order follows the OS directory listing.
    """
    ext_set = {e.lower() for e in _RASTER_EXTENSIONS}
    data_dir = os.path.abspath(data_dir)

    if index_workers == 0:
        index = _collect_rasters_under(data_dir, ext_set)
        return index, _build_index_lower(index)

    index = {}
    try:
        with os.scandir(data_dir) as it:
            entries = list(it)
    except OSError as e:
        print(f"Error reading {data_dir}: {e}")
        return {}, {}

    subdirs = []
    for e in entries:
        try:
            if e.is_file(follow_symlinks=False):
                stem, ext = os.path.splitext(e.name)
                if ext.lower() in ext_set:
                    index.setdefault(stem, e.path)
            elif e.is_dir(follow_symlinks=False):
                subdirs.append(e.path)
        except OSError:
            continue

    if not subdirs:
        return index, _build_index_lower(index)

    if index_workers < 0:
        index_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
    n_workers = max(1, min(index_workers, len(subdirs)))

    if n_workers == 1 or len(subdirs) == 1:
        for sd in subdirs:
            part = _collect_rasters_under(sd, ext_set)
            for k, v in part.items():
                index.setdefault(k, v)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_collect_rasters_under, sd, ext_set) for sd in subdirs]
            for fut in as_completed(futs):
                part = fut.result()
                for k, v in part.items():
                    index.setdefault(k, v)

    return index, _build_index_lower(index)


def load_embeddings_from_csv(embeddings_path):
    """Load embeddings from CSV file"""
    df = pd.read_csv(embeddings_path)
    
    # Separate filenames (if present) from embeddings
    if df.columns[0].lower() in ['filename', 'file', 'name', 'image']:
        filenames = df.iloc[:, 0].values
        embeddings = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        filenames = None
        embeddings = df.to_numpy(dtype=np.float32)
    
    return embeddings, filenames


def _load_standard_raster_rgb(filepath, transparent_bg=False):
    """Load a typical RGB/RGBA image (PNG, JPEG, WebP, …) for thumbnail display."""
    try:
        with Image.open(filepath) as im:
            if transparent_bg:
                return np.asarray(im.convert("RGBA"))
            # convert("RGB") would put black behind transparent pixels; match RGBA-on-white like plot_embeddings_images.
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[-1])
                return np.asarray(bg)
            return np.asarray(im.convert("RGB"))
    except Exception as e:
        print(f"Error loading raster image {filepath}: {e}")
        return None


def load_multispectral_image_rgb(filepath, transparent_bg=False, invert_mask=False, norm_foreground_only=False):
    """Load a multispectral TIFF or a standard RGB/RGBA preview (PNG, etc.) and return uint8 HxWx3 or HxWx4.

    Non-TIFF files are loaded with PIL; ``invert_mask`` / ``norm_foreground_only`` apply only to multispectral TIFFs.

    For TIFFs: rasterio first, then tifffile. If transparent_bg and band 7 exists, use it as alpha (non-zero -> visible).
    If invert_mask, treat zeros as foreground. If norm_foreground_only and mask exists, normalize using foreground only.
    """
    ext = _ext_key(filepath)
    if ext not in (".tif", ".tiff"):
        return _load_standard_raster_rgb(filepath, transparent_bg=transparent_bg)

    # First attempt with rasterio
    try:
        with rasterio.open(filepath) as src:
            if src.count >= 7:
                spectral_data = src.read([1, 2, 3]).astype(np.float32)  # (3, H, W)
                mask = src.read(7).astype(np.float32)  # (H, W)
            elif src.count >= 3:
                spectral_data = src.read([1, 2, 3]).astype(np.float32)
                mask = None
            else:
                print(f"Error: Image {filepath} has fewer than 3 bands ({src.count})")
                return None

            mask_bool = None
            if mask is not None:
                mask_bool = mask > 0
                if invert_mask:
                    mask_bool = ~mask_bool

            # If we want transparency and mask exists, don't zero out background; use mask as alpha instead
            if (mask_bool is not None) and (not transparent_bg):
                spectral_data = np.where(mask_bool[None, ...], spectral_data, 0.0)

            # Normalize each band independently to [0, 1]
            normalized_data = np.zeros_like(spectral_data, dtype=np.float32)
            for i in range(3):
                band = spectral_data[i]
                if (mask_bool is not None) and norm_foreground_only:
                    fg = band[mask_bool]
                    if fg.size > 0:
                        band_min = float(np.min(fg))
                        band_max = float(np.max(fg))
                    else:
                        band_min = float(np.min(band))
                        band_max = float(np.max(band))
                else:
                    band_min = float(np.min(band))
                    band_max = float(np.max(band))
                if band_max > band_min:
                    normalized_data[i] = (band - band_min) / (band_max - band_min)
                else:
                    normalized_data[i] = 0.0

            normalized_data = np.clip(normalized_data, 0.0, 1.0)
            rgb = (np.transpose(normalized_data, (1, 2, 0)) * 255).astype(np.uint8)  # (H, W, 3)

            if transparent_bg and (mask_bool is not None):
                alpha = np.where(mask_bool, 255, 0).astype(np.uint8)
                rgba = np.dstack([rgb, alpha])  # (H, W, 4)
                return rgba
            if mask_bool is not None:
                rgb = np.where(mask_bool[..., None], rgb, 255).astype(np.uint8)
            return rgb

    except Exception as e_rio:
        print(f"Rasterio failed on {filepath}: {e_rio}. Trying tifffile...")

    # Fallback: use tifffile which supports many-channel TIFFs
    if tiff is None:
        print("tifffile is not available. Please: pip install tifffile imagecodecs")
        return None

    try:
        arr = tiff.imread(filepath)  # shape could be (H, W, C) or (C, H, W)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)  # grayscale -> RGB
        if arr.ndim != 3:
            print(f"Unsupported TIFF array shape for {filepath}: {arr.shape}")
            return None

        # Determine channel axis heuristically
        if arr.shape[0] <= 10 and arr.shape[1] > 32 and arr.shape[2] > 32:
            # channels-first (C, H, W)
            C, H, W = arr.shape
            spectral_data = arr[:3, :, :].astype(np.float32)
            mask = arr[6, :, :].astype(np.float32) if C >= 7 else None
        else:
            # channels-last (H, W, C)
            H, W, C = arr.shape
            spectral_data = np.moveaxis(arr[..., :3], -1, 0).astype(np.float32)  # (3, H, W)
            mask = arr[..., 6].astype(np.float32) if C >= 7 else None

        mask_bool = None
        if mask is not None:
            mask_bool = mask > 0
            if invert_mask:
                mask_bool = ~mask_bool

        if (mask_bool is not None) and (not transparent_bg):
            spectral_data = np.where(mask_bool[None, ...], spectral_data, 0.0)

        normalized_data = np.zeros_like(spectral_data, dtype=np.float32)
        for i in range(3):
            band = spectral_data[i]
            if (mask_bool is not None) and norm_foreground_only:
                fg = band[mask_bool]
                if fg.size > 0:
                    band_min = float(np.min(fg))
                    band_max = float(np.max(fg))
                else:
                    band_min = float(np.min(band))
                    band_max = float(np.max(band))
            else:
                band_min = float(np.min(band))
                band_max = float(np.max(band))
            if band_max > band_min:
                normalized_data[i] = (band - band_min) / (band_max - band_min)
            else:
                normalized_data[i] = 0.0

        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        rgb = (np.transpose(normalized_data, (1, 2, 0)) * 255).astype(np.uint8)

        if transparent_bg and (mask_bool is not None):
            if mask.ndim == 3:
                mask_bool = mask_bool.squeeze()
            alpha = np.where(mask_bool, 255, 0).astype(np.uint8)
            rgba = np.dstack([rgb, alpha])
            return rgba
        if mask_bool is not None:
            if mask.ndim == 3:
                m = mask_bool.squeeze()
            else:
                m = mask_bool
            rgb = np.where(m[..., None], rgb, 255).astype(np.uint8)
        return rgb

    except Exception as e_tiff:
        print(f"Error loading multispectral image {filepath} with tifffile: {e_tiff}")
        return None

def get_multispectral_thumbnail(args):
    """Load and resize a multispectral image to create an RGB or RGBA thumbnail"""
    path, size, transparent_bg, invert_mask, norm_foreground_only = args
    
    # Convert .tif filename to full path if needed
    if not os.path.exists(path):
        return None
    
    array = load_multispectral_image_rgb(
        path,
        transparent_bg=transparent_bg,
        invert_mask=invert_mask,
        norm_foreground_only=norm_foreground_only,
    )
    if array is None:
        return None
    
    try:
        if len(array.shape) != 3 or array.shape[2] not in (3, 4):
            print(f"Error: Invalid array shape {array.shape} for {path}. Expected (H, W, 3) or (H, W, 4).")
            return None
        
        mode = 'RGBA' if array.shape[2] == 4 else 'RGB'
        img = Image.fromarray(array, mode=mode)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error creating thumbnail from array for {path}: {e}")
        return None

def find_multispectral_image(data_dir, filename, index=None, index_lower=None):
    """Find the full path for filename using stem match; disk file may use a different extension than CSV."""
    base_name = _csv_path_stem(filename)

    if index is not None:
        path = index.get(base_name)
        if path is None and index_lower is not None:
            path = index_lower.get(base_name.lower())
        if path is None:
            path = _find_stem_prefix_match(base_name, index)
        if path is not None:
            return path
        print(f"Warning: Could not find multispectral image for {filename}")
        return None

    # Fallback: try each known extension under data_dir (used only if no index is provided)
    for ext in _RASTER_EXTENSIONS:
        full_path = os.path.join(data_dir, base_name + ext)
        if os.path.exists(full_path):
            return full_path
    for ext in _RASTER_EXTENSIONS:
        pattern = os.path.join(data_dir, "**", base_name + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    for ext in _RASTER_EXTENSIONS:
        pattern = os.path.join(data_dir, "**", base_name + "_*" + ext)
        matches = glob.glob(pattern, recursive=True)
        if len(matches) == 1:
            return matches[0]

    print(f"Warning: Could not find multispectral image for {filename}")
    return None

def plot_embeddings_with_multispectral_images(embeddings, filenames, data_dir, output_path, method='umap', 
                                             thumb_size=100, max_images=1000, random_state=42, grid_scale=1.0,
                                             transparent_bg=False, fig_transparent=False,
                                             invert_mask=False, norm_foreground_only=False,
                                             image_index=None, image_index_lower=None, index_workers=-1):
    """
    Create and save a 2D visualization using RGB (or RGBA) thumbnails converted from multispectral bird images
    
    Parameters:
        embeddings: numpy array of embeddings
        filenames: list of filenames corresponding to embeddings
        data_dir: directory containing multispectral TIFFs and/or RGB previews (PNG, JPEG, …); matched by stem
        output_path: path to save the visualization
        method: dimensionality reduction method ('umap' or 'tsne')
        thumb_size: size of thumbnail images in pixels
        max_images: maximum number of images to plot
        random_state: random seed for reproducibility
        grid_scale: controls the density of images (lower values = more dense, higher = more sparse)
        transparent_bg: if True and mask band exists, use it as alpha for thumbnails
        fig_transparent: if True, save the figure with transparent background (decoupled from thumbnail alpha)
        invert_mask: if True, invert mask semantics (treat 0 as foreground)
        norm_foreground_only: if True, normalize channels using only foreground pixels
        image_index: optional pre-built stem -> path map (CLI passes this once for UMAP and t-SNE)
        image_index_lower: optional companion map from build_multispectral_index for case-insensitive lookup
        index_workers: passed to build_multispectral_index only when image_index is None
    """
    if image_index is None:
        print("Indexing multispectral image directory...")
        image_index, image_index_lower = build_multispectral_index(data_dir, index_workers=index_workers)
        print(f"Indexed {len(image_index)} raster files under {data_dir}")
    elif image_index_lower is None:
        image_index_lower = _build_index_lower(image_index)

    # Sample BEFORE dimensionality reduction so UMAP/t-SNE only run on max_images rows
    if len(filenames) > max_images:
        print(f"Sampling {max_images} embeddings from {len(filenames)} total before reduction...")
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(filenames), max_images, replace=False)
        embeddings = embeddings[indices]
        filenames = filenames[indices] if isinstance(filenames, np.ndarray) else np.asarray(filenames)[indices]

    print(f"Reducing dimensionality using {method.upper()} on {len(embeddings)} samples...")

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

    print("Finding multispectral image paths...")
    image_paths = []
    valid_indices = []
    ext_mismatch_count = 0

    for i, fname in enumerate(tqdm(filenames, desc="Finding images")):
        full_path = find_multispectral_image(
            data_dir, fname, index=image_index, index_lower=image_index_lower
        )
        if full_path:
            if _ext_key(str(fname)) != _ext_key(full_path):
                ext_mismatch_count += 1
            image_paths.append((full_path, (thumb_size, thumb_size), transparent_bg, invert_mask, norm_foreground_only))
            valid_indices.append(i)
        else:
            print(f"Skipping {fname} - file not found")

    if ext_mismatch_count:
        print(
            f"Note: {ext_mismatch_count} row(s) used a file whose extension differs from the CSV "
            f"(e.g. CSV lists .tif but data_dir has .png); lookup is by filename stem."
        )
    
    # Update embeddings and filenames to only include found images
    embedding_2d = embedding_2d[valid_indices]
    filenames = [filenames[i] for i in valid_indices]
    
    if len(image_paths) == 0:
        print("Error: No valid multispectral images found!")
        return
    
    print(f"Found {len(image_paths)} valid multispectral images")
    
    # Load images in parallel
    print("Loading and converting multispectral images to RGB thumbnails...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        thumbnails = list(tqdm(
            executor.map(get_multispectral_thumbnail, image_paths),
            total=len(image_paths),
            desc="Loading images"
        ))
    
    # Filter out None values (failed loads) and their corresponding embeddings
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
    grid_size = (thumb_size / 100) * grid_scale  # Adjustable grid size
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)
    occupied_grid = np.zeros((len(x_grid)-1, len(y_grid)-1), dtype=bool)
    
    # Plot thumbnails with grid-based overlap detection
    print("Placing images on plot...")
    for idx, (x0, y0) in enumerate(tqdm(embedding_2d, desc="Plotting images")):
        # Convert coordinates to grid indices
        x_idx = np.searchsorted(x_grid, x0) - 1
        y_idx = np.searchsorted(y_grid, y0) - 1
        
        # Check if grid cell is occupied
        if x_idx >= 0 and y_idx >= 0 and x_idx < occupied_grid.shape[0] and y_idx < occupied_grid.shape[1]:
            if occupied_grid[x_idx, y_idx]:
                continue
            
            # Mark grid cell as occupied
            occupied_grid[x_idx, y_idx] = True
            
            # Add thumbnail to plot
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(thumbnails[idx], zoom=0.5),
                (x0, y0),
                frameon=False,
                pad=0.0,
            )
            ax.add_artist(imagebox)
    
    plt.title(f'2D {method.upper()} projection with multispectral bird thumbnails (RGB conversion)')
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    # plt.grid(True, alpha=0.3)  # Optional: add a light grid
    
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=fig_transparent)
    plt.close()
    print("Saved successfully!")

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Generate UMAP and t-SNE visualizations with multispectral bird thumbnails converted to RGB')
    parser.add_argument('--embeddings_csv', type=str, required=True,
                      help='Path to the CSV file containing embeddings (first column should be filenames)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing multispectral TIFFs and/or preview rasters (PNG, JPEG, …); matched by stem to CSV names')
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
    parser.add_argument('--transparent_bg', action='store_true', default=True,
                      help='Use mask (band 7) as alpha for transparent thumbnails (RGBA)')
    parser.add_argument('--transparent_fig', action='store_true', default=False,
                      help='Save the figure canvas with a transparent background (default False)')
    parser.add_argument('--invert_mask', action='store_true', default=False,
                      help='Invert mask semantics (treat 0 as foreground, >0 as background)')
    parser.add_argument('--norm_foreground_only', action='store_true', default=False,
                      help='Normalize each RGB band using only foreground pixels (requires mask)')
    parser.add_argument('--index-workers', type=int, default=-1,
                      help='Raster index: 0 = single-threaded full-tree os.walk; -1 (default) = parallel '
                           'top-level subtree walks when several subfolders exist (no sorting; order is OS-defined).')
    return parser.parse_args()

def main(args):
    print(f"Loading embeddings from {args.embeddings_csv}...")
    embeddings, filenames = load_embeddings_from_csv(args.embeddings_csv)
    print(f"Loaded {len(embeddings)} embeddings")
    
    if filenames is None:
        print("Error: No filenames found in embeddings CSV. Please ensure the first column contains filenames.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Indexing multispectral image directory...")
    image_index, image_index_lower = build_multispectral_index(
        args.data_dir, index_workers=args.index_workers
    )
    print(f"Indexed {len(image_index)} raster files under {args.data_dir}")

    # Generate and save UMAP visualization
    if args.method in ['umap', 'both']:
        umap_output_path = os.path.join(args.output_dir, 'umap_multispectral_visualization.png')
        plot_embeddings_with_multispectral_images(
            embeddings, filenames, args.data_dir, umap_output_path,
            method='umap', thumb_size=args.thumb_size, max_images=args.max_images,
            grid_scale=args.grid_scale, random_state=args.random_state,
            transparent_bg=args.transparent_bg, fig_transparent=args.transparent_fig,
            invert_mask=args.invert_mask, norm_foreground_only=args.norm_foreground_only,
            image_index=image_index, image_index_lower=image_index_lower,
        )
    
    # Generate and save t-SNE visualization
    if args.method in ['tsne', 'both']:
        tsne_output_path = os.path.join(args.output_dir, 'tsne_multispectral_visualization.png')
        plot_embeddings_with_multispectral_images(
            embeddings, filenames, args.data_dir, tsne_output_path,
            method='tsne', thumb_size=args.thumb_size, max_images=args.max_images,
            grid_scale=args.grid_scale, random_state=args.random_state,
            transparent_bg=args.transparent_bg, fig_transparent=args.transparent_fig,
            invert_mask=args.invert_mask, norm_foreground_only=args.norm_foreground_only,
            image_index=image_index, image_index_lower=image_index_lower,
        )

if __name__ == "__main__":
    args = get_args()
    main(args)
