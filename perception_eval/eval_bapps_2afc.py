import os
import sys
import argparse
import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ensure local imports from this scripts directory work regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMCLR_DIR = os.path.dirname(SCRIPT_DIR)  # go up from perception_eval/ to simclr/
if SIMCLR_DIR not in sys.path:
    sys.path.insert(0, SIMCLR_DIR)

# Optional: ImageNet normalization constants
try:
    from lightly.transforms import utils as lightly_utils
    IMAGENET_MEAN = lightly_utils.IMAGENET_NORMALIZE["mean"]
    IMAGENET_STD = lightly_utils.IMAGENET_NORMALIZE["std"]
except Exception:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

import torchvision.transforms as T
import yaml

# LPIPS (pip install lpips)
try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


class BAPPS2AFCDataset(Dataset):
    """BAPPS 2AFC triplet dataset (ref, p0, p1, judge).

    Expects directory structure like:
      subset_dir/
        ref/
        p0/
        p1/
        judge/
    Files are sorted and matched by index (as in LPIPS repo).
    """

    def __init__(self, subset_dir: str, transform_lpips: Optional[T.Compose] = None,
                 transform_simclr: Optional[T.Compose] = None):
        self.subset_dir = subset_dir
        self.dir_ref = os.path.join(subset_dir, "ref")
        self.dir_p0 = os.path.join(subset_dir, "p0")
        self.dir_p1 = os.path.join(subset_dir, "p1")
        self.dir_judge = os.path.join(subset_dir, "judge")

        def sorted_files(d: str, exts: Tuple[str, ...]) -> List[str]:
            paths = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(d, f"*.{ext}")))
            return sorted(paths)

        self.ref_paths = sorted_files(self.dir_ref, ("png", "jpg", "jpeg", "bmp"))
        self.p0_paths = sorted_files(self.dir_p0, ("png", "jpg", "jpeg", "bmp"))
        self.p1_paths = sorted_files(self.dir_p1, ("png", "jpg", "jpeg", "bmp"))
        self.judge_paths = sorted_files(self.dir_judge, ("npy",))

        n = len(self.ref_paths)
        assert len(self.p0_paths) == n and len(self.p1_paths) == n and len(self.judge_paths) == n, \
            f"Mismatched counts in {subset_dir}"

        self.transform_lpips = transform_lpips
        self.transform_simclr = transform_simclr

    def __len__(self) -> int:
        return len(self.ref_paths)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        ref = self._load_image(self.ref_paths[idx])
        p0 = self._load_image(self.p0_paths[idx])
        p1 = self._load_image(self.p1_paths[idx])
        judge = float(np.load(self.judge_paths[idx]).astype(np.float32).reshape(-1)[0])  # fraction in [0,1]

        sample = {
            "judge": judge,
        }

        if self.transform_lpips is not None:
            sample["ref_lpips"] = self.transform_lpips(ref)
            sample["p0_lpips"] = self.transform_lpips(p0)
            sample["p1_lpips"] = self.transform_lpips(p1)

        if self.transform_simclr is not None:
            sample["ref_simclr"] = self.transform_simclr(ref)
            sample["p0_simclr"] = self.transform_simclr(p0)
            sample["p1_simclr"] = self.transform_simclr(p1)

        return sample


def get_device(prefer: Optional[str] = None) -> torch.device:
    # prefer can be None, 'cpu', 'cuda', 'mps'
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _import_simclr_class(module_path: str):
    """Dynamically import SimCLRModel from the given dotted module path.

    Example module_path values:
      - "train.simclr_birdcolour_RGB"
      - "train.simclr_birdcolour_multispectral"
      - "train.simclr_birdcolour_kornia02"
    """
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, "SimCLRModel", None)
    if cls is None:
        raise ImportError(f"Module '{module_path}' has no 'SimCLRModel' class")
    return cls


def _infer_backbone_from_state_dict(state_dict: dict) -> Optional[str]:
    """Peek at state_dict keys to decide whether weights belong to a ViT or ResNet."""
    for key in state_dict:
        if key.startswith("backbone.encoder.layers.encoder_layer_"):
            return "vit"
        if key.startswith("backbone.layer1."):
            return "resnet"
    return None


def load_simclr_model(ckpt_path: str, config_path: Optional[str], device: torch.device,
                      module_path: str = "train.simclr_birdcolour_multispectral"):
    """Load a SimCLR checkpoint from any training script.

    Parameters
    ----------
    ckpt_path : str
        Path to the Lightning ``.ckpt`` file.
    config_path : str or None
        Optional path to an external config YAML/JSON.  When *None* the config
        embedded in the checkpoint's ``hyper_parameters`` is used instead.
    device : torch.device
        Target device.
    module_path : str
        Dotted import path to the training module that defines ``SimCLRModel``
        (e.g. ``"train.simclr_birdcolour_RGB"``).
    """
    SimCLRClass = _import_simclr_class(module_path)
    print(f"[INFO] Using SimCLRModel from '{module_path}'")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    actual_backbone = _infer_backbone_from_state_dict(state_dict)

    hparams = ckpt.get("hyper_parameters", {})
    config = hparams.get("config", None)

    if config is None and config_path is not None:
        import json
        with open(config_path, "r") as f:
            full_cfg = json.load(f) if config_path.endswith(".json") else yaml.safe_load(f)
        config = full_cfg.get("model", full_cfg)

    if config is None:
        raise RuntimeError(
            "No config found in checkpoint hyper_parameters and no --simclr-config provided. "
            "Either use a checkpoint that embeds its config or supply --simclr-config."
        )

    if actual_backbone == "vit" and config.get("backbone", "").startswith("resnet"):
        print(f"[INFO] Checkpoint weights are ViT but config says '{config['backbone']}' — overriding to 'vit'")
        config["backbone"] = "vit"
    elif actual_backbone == "resnet" and not config.get("backbone", "").startswith("resnet"):
        print(f"[INFO] Checkpoint weights are ResNet but config says '{config['backbone']}' — overriding to 'resnet'")
        config["backbone"] = "resnet"

    model = SimCLRClass.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    model = model.to(device)
    return model


def simclr_feature_extractor(model, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Return L2-normalized backbone features, handling both tensor and dict outputs."""
    with torch.no_grad():
        feats = model.backbone(x)
        if isinstance(feats, dict):
            if 'x_norm_patchtokens' in feats:
                feats = feats['x_norm_patchtokens'].mean(dim=1)
            elif 'x_norm_clstoken' in feats:
                feats = feats['x_norm_clstoken']
            else:
                for key in ('x_cls', 'features', 'pooler_output'):
                    if key in feats:
                        feats = feats[key]
                        break
                else:
                    tensors = [v for v in feats.values() if isinstance(v, torch.Tensor)]
                    if tensors:
                        feats = tensors[0]
                    else:
                        raise RuntimeError(f"Cannot extract features from backbone dict: {list(feats.keys())}")
        feats = feats.flatten(1)
        if normalize:
            feats = F.normalize(feats, dim=1)
    return feats


def _collate_simclr(batch):
    refs = torch.stack([b["ref_simclr"] for b in batch])
    p0s = torch.stack([b["p0_simclr"] for b in batch])
    p1s = torch.stack([b["p1_simclr"] for b in batch])
    judges = torch.tensor([b["judge"] for b in batch], dtype=torch.float32)
    return refs, p0s, p1s, judges


def _collate_lpips(batch):
    refs = torch.stack([b["ref_lpips"] for b in batch])
    p0s = torch.stack([b["p0_lpips"] for b in batch])
    p1s = torch.stack([b["p1_lpips"] for b in batch])
    judges = torch.tensor([b["judge"] for b in batch], dtype=torch.float32)
    return refs, p0s, p1s, judges


def evaluate_simclr_2afc(subset_dir: str, model, device: torch.device,
                          input_size: int, batch_size: int, num_workers: int,
                          distance: str = "cosine", normalize: bool = True,
                          return_raw: bool = False):
    t_simclr = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds = BAPPS2AFCDataset(subset_dir, transform_simclr=t_simclr)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    collate_fn=_collate_simclr)

    all_scores: List[np.ndarray] = []
    for refs, p0s, p1s, judges in dl:
        refs = refs.to(device)
        p0s = p0s.to(device)
        p1s = p1s.to(device)

        z_ref = simclr_feature_extractor(model, refs, normalize=normalize)
        z_p0 = simclr_feature_extractor(model, p0s, normalize=normalize)
        z_p1 = simclr_feature_extractor(model, p1s, normalize=normalize)

        # compute distances
        if distance == "cosine":
            d0 = 1.0 - F.cosine_similarity(z_ref, z_p0, dim=1)
            d1 = 1.0 - F.cosine_similarity(z_ref, z_p1, dim=1)
        elif distance == "euclidean":
            d0 = F.pairwise_distance(z_ref, z_p0, p=2)
            d1 = F.pairwise_distance(z_ref, z_p1, p=2)
        else:
            raise ValueError(f"Unknown distance: {distance}")

        choose_p1 = (d1 < d0).float().cpu()
        judges = judges.cpu()

        # fractional 2AFC score: if choose p1, score=judge; else score=1-judge
        scores = choose_p1 * judges + (1.0 - choose_p1) * (1.0 - judges)
        all_scores.append(scores.numpy())

    raw = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32)
    mean_score = float(raw.mean()) if raw.size > 0 else float("nan")
    if return_raw:
        return mean_score, raw
    return mean_score


def build_lpips_model(lpips_net: str, device: torch.device):
    """Instantiate an LPIPS network once, so it can be reused across multiple
    subsets/datasets instead of reloading weights from disk every time."""
    if lpips_lib is None:
        raise RuntimeError("lpips is not installed. Please `pip install lpips`. ")
    lpips_fn = lpips_lib.LPIPS(net=lpips_net).to(device)
    lpips_fn.eval()
    return lpips_fn


def evaluate_lpips_2afc(subset_dir: str, device: torch.device, batch_size: int, num_workers: int,
                         lpips_net: str = "alex", lpips_size: int = 256,
                         return_raw: bool = False, lpips_model=None):
    if lpips_lib is None:
        raise RuntimeError("lpips is not installed. Please `pip install lpips`. ")

    # LPIPS expects inputs in [0,1] if normalize=True is passed to forward
    t_ops = []
    if lpips_size and lpips_size > 0:
        t_ops.append(T.Resize((lpips_size, lpips_size), interpolation=T.InterpolationMode.BILINEAR))
    t_ops.append(T.ToTensor())  # [0,1]
    t_lpips = T.Compose(t_ops)

    ds = BAPPS2AFCDataset(subset_dir, transform_lpips=t_lpips)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    collate_fn=_collate_lpips)

    lpips_fn = lpips_model if lpips_model is not None else build_lpips_model(lpips_net, device)

    all_scores: List[np.ndarray] = []
    with torch.no_grad():
        for refs, p0s, p1s, judges in dl:
            refs = refs.to(device)
            p0s = p0s.to(device)
            p1s = p1s.to(device)

            d0 = lpips_fn(refs, p0s, normalize=True).view(-1)
            d1 = lpips_fn(refs, p1s, normalize=True).view(-1)

            choose_p1 = (d1 < d0).float().cpu()
            judges = judges.cpu()

            scores = choose_p1 * judges + (1.0 - choose_p1) * (1.0 - judges)
            all_scores.append(scores.numpy())

    raw = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32)
    mean_score = float(raw.mean()) if raw.size > 0 else float("nan")
    if return_raw:
        return mean_score, raw
    return mean_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS and SimCLR on BAPPS 2AFC")
    parser.add_argument("--bapps-root", type=str, required=True,
                        help="Path to BAPPS 2AFC root, e.g., /path/PerceptualSimilarity/dataset/2afc")
    parser.add_argument("--subsets", nargs="+", default=[
        "val/traditional", "val/cnn", "val/superres", "val/deblur", "val/color", "val/frameinterp"
    ], help="Subsets to evaluate (relative to --bapps-root)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Force device (default: auto)")

    # LPIPS options
    parser.add_argument("--eval-lpips", action="store_true", help="Evaluate LPIPS")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone")
    parser.add_argument("--lpips-size", type=int, default=256,
                        help="Resize HxW for LPIPS batching (e.g., 256).")

    # SimCLR options
    parser.add_argument("--eval-simclr", action="store_true", help="Evaluate SimCLR embeddings")
    parser.add_argument("--simclr-checkpoint", type=str, default=None, help="Path to SimCLR checkpoint")
    parser.add_argument("--simclr-config", type=str, default=None,
                        help="Path to SimCLR config YAML/JSON (optional; uses checkpoint's embedded config by default)")
    parser.add_argument("--simclr-module", type=str, default="train.simclr_birdcolour_multispectral",
                        help="Dotted module path that defines SimCLRModel, e.g. "
                             "'train.simclr_birdcolour_RGB' or 'train.simclr_birdcolour_multispectral'")
    parser.add_argument("--simclr-input-size", type=int, default=224, help="Resize for SimCLR backbone input")
    parser.add_argument("--simclr-distance", type=str, default="cosine", choices=["cosine", "euclidean"],
                        help="Distance metric for SimCLR features.")
    parser.add_argument("--no-simclr-normalize", action="store_true",
                        help="Disable L2-normalization of SimCLR features before distance.")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    results: Dict[str, Dict[str, float]] = {}

    # Optionally load SimCLR
    simclr_model = None
    if args.eval_simclr:
        assert args.simclr_checkpoint is not None, \
            "--simclr-checkpoint is required when --eval-simclr is set"
        simclr_model = load_simclr_model(
            args.simclr_checkpoint, args.simclr_config, device,
            module_path=args.simclr_module,
        )

    all_lpips = []
    all_simclr = []

    for subset in args.subsets:
        subset_dir = os.path.join(args.bapps_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"[WARN] Skipping missing subset: {subset_dir}")
            continue

        res = {}
        if args.eval_lpips:
            acc_lpips = evaluate_lpips_2afc(
                subset_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lpips_net=args.lpips_net,
                lpips_size=args.lpips_size,
            )
            res["lpips"] = acc_lpips
            all_lpips.append(acc_lpips)
            print(f"LPIPS[{args.lpips_net}]  {subset:>16}: {acc_lpips*100:.2f}%")

        if args.eval_simclr and simclr_model is not None:
            acc_simclr = evaluate_simclr_2afc(
                subset_dir,
                model=simclr_model,
                device=device,
                input_size=args.simclr_input_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                distance=args.simclr_distance,
                normalize=(not args.no_simclr_normalize),
            )
            tag = f"simclr_{args.simclr_distance}{'_unnorm' if args.no_simclr_normalize else ''}"
            res[tag] = acc_simclr
            all_simclr.append(acc_simclr)
            print(f"SimCLR[{args.simclr_distance}{', unnorm' if args.no_simclr_normalize else ''}] {subset:>16}: {acc_simclr*100:.2f}%")

        results[subset] = res

    # Overall averages
    if all_lpips:
        print(f"LPIPS[{args.lpips_net}]  OVERALL: {np.mean(all_lpips)*100:.2f}%")
    if all_simclr:
        print(f"SimCLR[{args.simclr_distance}{', unnorm' if args.no_simclr_normalize else ''}] OVERALL: {np.mean(all_simclr)*100:.2f}%")


if __name__ == "__main__":
    main()
