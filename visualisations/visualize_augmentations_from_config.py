"""
Plot each Kornia augmentation from a SimCLR-style YAML config (RGB) applied in
isolation to one image. Matches the augmentation keys used in
simclr/train/simclr_birdcolour_RGB.py.

Layout: **Original** (full resolution, no resize or pad), then one example of
the training **spatial** step (random resized crop or resize+pad), then each
further enabled augmentation applied after a fresh random spatial step. This
mirrors SimCLR view generation: every transform call samples its own crop before
the remaining stochastic augmentations are sampled.

Apple Silicon: ``auto`` does not use MPS (Kornia geometry uses float64, which MPS
lacks); use ``--device cpu`` to skip the stderr notice. ImageNet-style
``normalize`` is omitted from the grid so panels stay in displayable RGB.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import yaml

# Panel titles (Original, crop, each augmentation)
TITLE_FONTSIZE = 16
FIG_TITLE_FONTSIZE = 16


def resolve_device(preference: str) -> torch.device:
    """
    Select device for running Kornia augmentations.

    ``auto`` prefers CUDA, otherwise CPU. Apple MPS is **not** used in ``auto`` mode
    because Kornia geometric ops (e.g. RandomResizedCrop) solve linear systems in
    float64, which MPS does not support.
    """
    pref = preference.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preference}")


def load_image_bchw(path: str) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil, dtype=np.uint8)
    if arr.size == 0:
        raise FileNotFoundError(f"Could not read image: {path}")
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(torch.float32)


def tensor_to_imshow(t: torch.Tensor) -> np.ndarray:
    """BCHW -> HWRGB float [0,1] numpy."""
    x = t.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)


def _f(x) -> float:
    """YAML often emits ints; Kornia then infers Long dtype and samplers break."""
    return float(x)


def build_spatial_crop(
    transform_cfg: dict, input_size: int
) -> Tuple[str, nn.Module]:
    """
    First stage of the training pipeline: crop or resize+pad to model input size.
    """
    cfg = transform_cfg
    strategy = cfg["augmentation_strategy"]

    if strategy == "crop":
        rrc = cfg.get("random_resize_crop") or {}
        if rrc.get("enabled"):
            return (
                "random_resize_crop",
                K.RandomResizedCrop(
                    size=tuple(int(x) for x in rrc["size"]),
                    scale=(float(rrc["scale"][0]), float(rrc["scale"][1])),
                    p=1.0,
                ),
            )
        return (
            "resize_and_pad",
            nn.Sequential(
                K.Resize(size=input_size, side="long"),
                K.PadTo(size=(input_size, input_size)),
            ),
        )

    if strategy == "resize_and_pad":
        return (
            "resize_and_pad",
            nn.Sequential(
                K.Resize(size=input_size, side="long"),
                K.PadTo(size=(input_size, input_size)),
            ),
        )

    raise ValueError(f"Unknown augmentation_strategy: {strategy}")


def build_post_spatial_augmentations(
    transform_cfg: dict,
) -> List[Tuple[str, nn.Module]]:
    """Augmentations after the spatial crop, in training-pipeline order."""
    cfg = transform_cfg
    out: List[Tuple[str, nn.Module]] = []

    checks = [
         (
            "grayscale",
            lambda: cfg["grayscale"]["enabled"],
            lambda: K.RandomGrayscale(p=1.0),
        ),
        (
            "horizontal_flip",
            lambda: cfg["horizontal_flip"]["enabled"],
            lambda: K.RandomHorizontalFlip(p=1.0),
        ),
        (
            "vertical_flip",
            lambda: cfg["vertical_flip"]["enabled"],
            lambda: K.RandomVerticalFlip(p=1.0),
        ),
        (
            "rotate",
            lambda: cfg["rotate"]["enabled"],
            lambda: K.RandomRotation(
                degrees=_f(cfg["rotate"]["limit"]),
                p=1.0,
            ),
        ),
        (
            "color_jitter",
            lambda: cfg["color_jitter"]["enabled"],
            lambda: K.ColorJitter(
                brightness=_f(cfg["color_jitter"]["brightness"]),
                contrast=_f(cfg["color_jitter"]["contrast"]),
                saturation=_f(cfg["color_jitter"]["saturation"]),
                hue=_f(cfg["color_jitter"]["hue"]),
                p=1.0,
            ),
        ),
        (
            "gaussian_blur",
            lambda: cfg["gaussian_blur"]["enabled"],
            lambda: K.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(
                    tuple(float(x) for x in cfg["gaussian_blur"]["sigma_limit"])
                    if isinstance(cfg["gaussian_blur"]["sigma_limit"], (list, tuple))
                    else float(cfg["gaussian_blur"]["sigma_limit"])
                ),
                p=1.0,
            ),
        ),
        (
            "gaussian_noise",
            lambda: cfg["gaussian_noise"]["enabled"],
            lambda: K.RandomGaussianNoise(
                mean=_f(cfg["gaussian_noise"]["mean_range"][0]),
                std=_f(cfg["gaussian_noise"]["std_range"][1]),
                p=1.0,
            ),
        ),
        (
            "thin_plate_spline",
            lambda: cfg.get("thin_plate_spline", {}).get("enabled", False),
            lambda: K.RandomThinPlateSpline(
                scale=_f(cfg["thin_plate_spline"]["scale"]),
                p=1.0,
            ),
        ),
        (
            "random_perspective",
            lambda: cfg.get("random_perspective", {}).get("enabled", False),
            lambda: K.RandomPerspective(
                distortion_scale=_f(cfg["random_perspective"]["distortion_scale"]),
                p=1.0,
            ),
        ),
        (
            "random_sharpness",
            lambda: cfg.get("random_sharpness", {}).get("enabled", False),
            lambda: K.RandomSharpness(
                sharpness=_f(cfg["random_sharpness"]["sharpness"]),
                p=1.0,
            ),
        ),
        (
            "random_posterize",
            lambda: cfg.get("random_posterize", {}).get("enabled", False),
            lambda: K.RandomPosterize(
                bits=cfg["random_posterize"]["bits"],
                p=1.0,
            ),
        ),
        (
            "random_erasing",
            lambda: cfg.get("random_erasing", {}).get("enabled", False),
            lambda: K.RandomErasing(
                scale=(float(cfg["random_erasing"]["scale"][0]), float(cfg["random_erasing"]["scale"][1])),
                p=1.0,
            ),
        ),
    ]

    for name, enabled_fn, build_fn in checks:
        try:
            if enabled_fn():
                out.append((name, build_fn()))
        except KeyError:
            continue

    return out


def format_title(name: str) -> str:
    return name.replace("_", " ").title()


def run(
    config_path: str,
    image_path: str,
    output_path: str,
    seed: int,
    device_pref: str,
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    aug = config.get("augmentations") or {}
    transforms = aug.get("transforms")
    if transforms is None:
        raise ValueError(
            "Config must contain augmentations.transforms (Kornia / simclr_birdcolour_RGB.yaml style)."
        )
    input_size = int(aug["input_size"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if (
        device_pref == "auto"
        and not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        print(
            "Using CPU for augmentations (Kornia + MPS: float64 ops unsupported on MPS).",
            file=sys.stderr,
        )

    device = resolve_device(device_pref)
    x_full = load_image_bchw(image_path).to(device=device, dtype=torch.float32)

    spatial_name, spatial_mod = build_spatial_crop(transforms, input_size)
    spatial_mod = spatial_mod.to(device=device, dtype=torch.float32)
    spatial_mod.eval()

    with torch.no_grad():
        cropped = spatial_mod(x_full.clone())

    rest_layers = build_post_spatial_augmentations(transforms)

    ncol = 2 + len(rest_layers)
    fig_w = max(14, 3.0 * ncol)
    fig, axes = plt.subplots(1, ncol, figsize=(fig_w, 4.2))
    if ncol == 1:
        axes = [axes]

    axes[0].imshow(tensor_to_imshow(x_full.cpu()))
    axes[0].set_title("Original", fontsize=TITLE_FONTSIZE)
    axes[0].axis("off")

    axes[1].imshow(tensor_to_imshow(cropped.cpu()))
    axes[1].set_title(format_title(spatial_name), fontsize=TITLE_FONTSIZE)
    axes[1].axis("off")

    for ax, (name, module) in zip(axes[2:], rest_layers):
        # Kornia: `.to(device)` leaves dtype=None → make_samplers uses Long for
        # Uniform(0,1) bounds; always pair device with float32.
        module = module.to(device=device, dtype=torch.float32)
        module.eval()

        with torch.no_grad():
            try:
                # Match SimCLR's per-view sampling: each panel gets a fresh
                # random crop before the selected augmentation is applied.
                inp = spatial_mod(x_full.clone()).contiguous().to(dtype=torch.float32)
                y = module(inp)
            except Exception as e:
                raise RuntimeError(
                    f'Failed applying augmentation "{name}". '
                    "Check config fields and image size."
                ) from e

        ax.imshow(tensor_to_imshow(y.cpu()))
        ax.set_title(format_title(name), fontsize=TITLE_FONTSIZE)
        ax.axis("off")

    fig.suptitle(os.path.basename(config_path), fontsize=FIG_TITLE_FONTSIZE, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize each enabled RGB Kornia augmentation from a YAML config.",
    )
    p.add_argument("--image", required=True, help="Path to input .png (.jpg works too)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--output",
        default=None,
        help="Output figure path (.png). Default: ./augmentation_panels_<stem>.png",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help='Where to run Kornia (default: auto = CUDA if available, else CPU; MPS is skipped because Kornia geometry uses float64). Use "cpu" to silence the MPS notice.',
    )
    args = p.parse_args()

    out = args.output
    if out is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out = os.path.abspath(f"augmentation_panels_{stem}.png")

    run(args.config, args.image, out, args.seed, args.device)


if __name__ == "__main__":
    main()
