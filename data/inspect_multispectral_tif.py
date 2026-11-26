#!/usr/bin/env python3
"""
inspect_multispectral_tif.py

Inspect a multispectral .tif to discover any embedded band/wavelength
metadata, and print structural info (band count, dtype, per-band stats)
to help sanity-check band-order assumptions used elsewhere in the pipeline
(e.g. simclr/data/multispectral_data.py).

Usage:
    python inspect_multispectral_tif.py /path/to/specimen.tif
    python inspect_multispectral_tif.py /path/to/dir_of_tifs/ --sample 5
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio


def inspect_file(path: str, show_stats: bool = True) -> None:
    print(f"\n{'=' * 70}")
    print(f"File: {path}")
    print(f"{'=' * 70}")

    with rasterio.open(path) as src:
        print(f"Driver:        {src.driver}")
        print(f"Band count:    {src.count}")
        print(f"Size (W x H):  {src.width} x {src.height}")
        print(f"Dtype:         {src.dtypes}")
        print(f"CRS:           {src.crs}")

        # Dataset-level tags (GDAL metadata, e.g. TIFFTAG_* or custom keys)
        ds_tags = src.tags()
        print(f"\nDataset-level tags ({len(ds_tags)}):")
        if ds_tags:
            for k, v in ds_tags.items():
                print(f"  {k}: {v}")
        else:
            print("  (none)")

        # Band descriptions (sometimes used to store e.g. 'Red 650nm')
        print(f"\nBand descriptions:")
        descs = src.descriptions
        if any(descs):
            for i, d in enumerate(descs):
                print(f"  Band {i}: {d!r}")
        else:
            print("  (none set)")

        # Per-band tags and color interpretation
        print(f"\nPer-band tags / color interpretation:")
        for i in range(1, src.count + 1):
            band_tags = src.tags(i)
            colorinterp = src.colorinterp[i - 1] if src.colorinterp else None
            print(f"  Band {i}: colorinterp={colorinterp}, tags={band_tags or '(none)'}")

        # Look for any wavelength-ish keys, case-insensitively, in all tags
        wavelength_hits = []
        for i in range(0, src.count + 1):  # 0 = dataset-level
            tags = ds_tags if i == 0 else src.tags(i)
            for k, v in tags.items():
                if any(term in k.lower() for term in
                       ("wavelength", "wvl", "nm", "band_name", "spectral")):
                    wavelength_hits.append((i, k, v))

        print(f"\nPossible wavelength-related tags found: {len(wavelength_hits)}")
        for band_idx, k, v in wavelength_hits:
            label = "dataset" if band_idx == 0 else f"band {band_idx}"
            print(f"  [{label}] {k} = {v}")

        if not wavelength_hits:
            print("  None found. Wavelength info is NOT embedded in this file's "
                  "metadata — it must come from the capture/calibration protocol "
                  "used upstream (camera + filter specs), not from the pixel data.")

        # Basic per-band pixel statistics — useful for confirming which bands
        # are visible vs UV (UV channels are often much dimmer / noisier if the
        # image wasn't captured with a converted+filtered camera correctly).
        if show_stats:
            print(f"\nPer-band pixel statistics:")
            data = src.read()
            for i in range(data.shape[0]):
                band = data[i].astype(np.float64)
                nz = band[band > 0]
                print(f"  Band {i}: min={band.min():.1f} max={band.max():.1f} "
                      f"mean={band.mean():.2f} nonzero_frac={nz.size / band.size:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to a .tif file or a directory of .tif files")
    p.add_argument("--sample", type=int, default=1,
                    help="If path is a directory, inspect this many files (default 1)")
    p.add_argument("--no-stats", action="store_true", help="Skip pixel statistics")
    args = p.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", "*.tif*"), recursive=True))
        if not files:
            print(f"No .tif files found under {args.path}", file=sys.stderr)
            sys.exit(1)
        for f in files[: args.sample]:
            inspect_file(f, show_stats=not args.no_stats)
    else:
        inspect_file(args.path, show_stats=not args.no_stats)


if __name__ == "__main__":
    main()