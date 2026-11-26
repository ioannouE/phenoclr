import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import datetime as dt
import json
import os
import re
import yaml
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import wandb
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform, utils

import kornia
import torchvision.transforms.v2 as v2

import tempfile


def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# DINOv3 helpers
# ---------------------------------------------------------------------------

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from a variety of checkpoint formats.

    Handles keys like 'state_dict', 'model', 'model_state_dict'. If both
    teacher/student branches exist (as in some DINO-style checkpoints), prefer
    'teacher'. Also strips common prefixes like 'module.', 'model.', 'backbone.',
    'encoder.', 'visual.'.
    """
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model', 'model_state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        raise ValueError('Unsupported checkpoint type; expected a dict or dict with state_dict')

    keys = list(sd.keys())

    if any(k.startswith('teacher.') for k in keys):
        sd = {k[len('teacher.'):]: v for k, v in sd.items() if k.startswith('teacher.')}
    elif any(k.startswith('student.') for k in keys):
        sd = {k[len('student.'):]: v for k, v in sd.items() if k.startswith('student.')}

    def strip_prefix(k: str) -> str:
        for p in ('module.', 'model.', 'backbone.', 'encoder.', 'visual.'):
            if k.startswith(p):
                return k[len(p):]
        return k

    sd = {strip_prefix(k): v for k, v in sd.items()}
    return sd


def guess_arch_from_weights(weights_path: str) -> Optional[str]:
    """Best-effort heuristic to guess DINOv3 ViT arch from a checkpoint."""
    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        sd = _extract_state_dict(ckpt)
        if 'pos_embed' in sd and sd['pos_embed'].ndim == 3:
            d = sd['pos_embed'].shape[-1]
        elif 'blocks.0.attn.qkv.weight' in sd and sd['blocks.0.attn.qkv.weight'].ndim == 2:
            d = sd['blocks.0.attn.qkv.weight'].shape[1]
        elif 'norm.weight' in sd and sd['norm.weight'].ndim == 1:
            d = sd['norm.weight'].shape[0]
        else:
            d = None
        if d is None:
            return None
        if d >= 4000:
            return 'dinov3_vit7b16'
        if d >= 1024:
            return 'dinov3_vitl16'
        if d >= 768:
            return 'dinov3_vitb16'
        return 'dinov3_vits16'
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def get_mean_lab_color(x):
    x = x * 255.0
    x = x.to(torch.uint8)

    mean_lab = []
    for img in x:
        img_np = img.permute(1, 2, 0).cpu().numpy()
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab_mean = lab.mean(axis=(0, 1)) / 255.0
        mean_lab.append(torch.tensor(lab_mean, dtype=torch.float32))

    return torch.stack(mean_lab).to(x.device)


# ---------------------------------------------------------------------------
# Data pipeline (Kornia-based)
# ---------------------------------------------------------------------------

class KorniaTransform:
    """Custom transform class that applies Kornia transforms twice for SimCLR"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)

        result1 = self.transform(img)
        result2 = self.transform(img)

        return result1.squeeze(0), result2.squeeze(0)


class SimCLRDataset(torch.utils.data.Dataset):
    """Custom dataset that uses Kornia transforms"""
    def __init__(self, input_dir, transform=None):
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform

        all_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(root, f))

        all_files.sort()

        self.files = all_files
        print(f"Dataset ready: {len(self.files)} images")

    def _is_valid_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            return img is not None and img.size > 0
        except:
            return False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        return img, img_path


def collate_fn(batch):
    aug_pairs, filenames = zip(*batch)
    aug1s, aug2s = zip(*aug_pairs)
    aug1s = torch.stack(aug1s)
    aug2s = torch.stack(aug2s)
    return (aug1s, aug2s), filenames


class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config.get("dataloader", {}).get("batch_size", 32)
        self.num_workers = config.get("dataloader", {}).get("num_workers", 4)
        self.data_dir = config["data"]["data_dir"]
        self.input_size = config["augmentations"]["input_size"]
        self.transform_config = config["augmentations"]["transforms"]

        self.train_transform = self._get_train_transforms()

    def _get_train_transforms(self):
        cfg = self.transform_config
        transforms = []

        if cfg["augmentation_strategy"] == "crop":
            if cfg["random_resize_crop"]["enabled"]:
                transforms.append(kornia.augmentation.RandomResizedCrop(
                    size=tuple(cfg["random_resize_crop"]["size"]),
                    scale=tuple(cfg["random_resize_crop"]["scale"]),
                    p=cfg["random_resize_crop"]["p"]
                ))
        elif cfg["augmentation_strategy"] == "resize_and_pad":
            print("Random resize crop is disabled, using fixed size resize and padding")
            transforms.append(
                kornia.augmentation.Resize(size=self.input_size, side='long')
            )
            transforms.append(
                kornia.augmentation.PadTo(size=(self.input_size, self.input_size))
            )

        if cfg["horizontal_flip"]["enabled"]:
            transforms.append(kornia.augmentation.RandomHorizontalFlip(
                p=cfg["horizontal_flip"]["p"]
            ))

        if cfg["vertical_flip"]["enabled"]:
            transforms.append(kornia.augmentation.RandomVerticalFlip(
                p=cfg["vertical_flip"]["p"]
            ))

        if cfg["rotate"]["enabled"]:
            transforms.append(kornia.augmentation.RandomRotation(
                degrees=cfg["rotate"]["limit"],
                p=cfg["rotate"]["p"]
            ))

        if cfg["grayscale"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGrayscale(
                p=cfg["grayscale"]["p"]
            ))

        if cfg["color_jitter"]["enabled"]:
            transforms.append(kornia.augmentation.ColorJitter(
                brightness=cfg["color_jitter"]["brightness"],
                contrast=cfg["color_jitter"]["contrast"],
                saturation=cfg["color_jitter"]["saturation"],
                hue=cfg["color_jitter"]["hue"],
                p=cfg["color_jitter"]["p"]
            ))

        if cfg["gaussian_blur"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=cfg["gaussian_blur"]["sigma_limit"],
                p=cfg["gaussian_blur"]["p"]
            ))

        if cfg["gaussian_noise"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGaussianNoise(
                mean=cfg["gaussian_noise"]["mean_range"][0],
                std=cfg["gaussian_noise"]["std_range"][1],
                p=cfg["gaussian_noise"]["p"]
            ))

        if "thin_plate_spline" in cfg:
            if cfg["thin_plate_spline"]["enabled"]:
                transforms.append(kornia.augmentation.RandomThinPlateSpline(
                    scale=cfg["thin_plate_spline"]["scale"],
                    p=cfg["thin_plate_spline"]["p"]
                ))

        if "random_perspective" in cfg:
            if cfg["random_perspective"]["enabled"]:
                transforms.append(kornia.augmentation.RandomPerspective(
                    distortion_scale=cfg["random_perspective"]["distortion_scale"],
                    p=cfg["random_perspective"]["p"]
                ))

        if "random_sharpness" in cfg:
            if cfg["random_sharpness"]["enabled"]:
                transforms.append(kornia.augmentation.RandomSharpness(
                    sharpness=cfg["random_sharpness"]["sharpness"],
                    p=cfg["random_sharpness"]["p"]
                ))

        if "random_posterize" in cfg:
            if cfg["random_posterize"]["enabled"]:
                transforms.append(kornia.augmentation.RandomPosterize(
                    bits=cfg["random_posterize"]["bits"],
                    p=cfg["random_posterize"]["p"]
                ))

        if "random_erasing" in cfg:
            if cfg["random_erasing"]["enabled"]:
                transforms.append(kornia.augmentation.RandomErasing(
                    scale=tuple(cfg["random_erasing"]["scale"]),
                    p=cfg["random_erasing"]["p"]
                ))

        transforms.append(kornia.augmentation.Normalize(
            mean=torch.tensor(cfg["normalize"]["mean"]),
            std=torch.tensor(cfg["normalize"]["std"])
        ))

        return KorniaTransform(nn.Sequential(*transforms))

    def setup(self, stage: str = "train"):
        if stage == 'fit':
            self.dataset_train = SimCLRDataset(
                self.data_dir,
                transform=self.train_transform
            )

        if stage == "predict":
            predict_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=utils.IMAGENET_NORMALIZE["mean"],
                    std=utils.IMAGENET_NORMALIZE["std"],
                ),
            ])
            self.dataset_predict = LightlyDataset(input_dir=self.data_dir,
                                                  transform=predict_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


# ---------------------------------------------------------------------------
# SimCLR model with selectable backbone: resnet | vit | dino
# ---------------------------------------------------------------------------

class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.T_max = config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = config.get('optimizer').get('lr', lr)
        self.temp = config.get('criterion').get('params', {}).get('temperature', 0.5)

        self.backbone_type = config.get("backbone", "dino").lower()
        self.backbone, self.hidden_dim = self._build_backbone()

        self.projection_head = SimCLRProjectionHead(self.hidden_dim, self.hidden_dim, 128)
        self.criterion = NTXentLoss(self.temp)

    # ----- backbone factory ------------------------------------------------

    def _build_backbone(self):
        if self.backbone_type == "resnet":
            return self._build_resnet()
        elif self.backbone_type == "vit":
            return self._build_vit()
        elif self.backbone_type == "dino":
            return self._build_dino()
        else:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_type}'. "
                "Choose from: resnet, vit, dino"
            )

    def _build_resnet(self):
        resnet_cfg = self.config.get("resnet", {})
        model_name = resnet_cfg.get("model", "resnet50")
        weights = resnet_cfg.get("weights", "IMAGENET1K_V1")
        print(f"Loading ResNet backbone: {model_name} (weights={weights})")
        base_model = torchvision.models.get_model(model_name, weights=weights)
        hidden_dim = base_model.fc.in_features
        backbone = nn.Sequential(*list(base_model.children())[:-1])
        return backbone, hidden_dim

    def _build_vit(self):
        vit_cfg = self.config.get("vit", {})
        model_name = vit_cfg.get("model", "vit_l_16")
        weights_name = vit_cfg.get("weights", "IMAGENET1K_V1")
        image_size = vit_cfg.get("image_size", 224)

        print(f"Loading ViT backbone: {model_name} (weights={weights_name})")

        if model_name == "vit_l_16":
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            weights = ViT_L_16_Weights.IMAGENET1K_V1
            base_model = vit_l_16(weights=weights, image_size=image_size)
            hidden_dim = 1024
        elif model_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            base_model = vit_b_16(weights=weights, image_size=image_size)
            hidden_dim = 768
        else:
            base_model = torchvision.models.get_model(model_name, weights=weights_name, image_size=image_size)
            dummy = torch.randn(1, 3, image_size, image_size)
            base_model.heads = nn.Identity()
            hidden_dim = base_model(dummy).shape[-1]

        base_model.heads = nn.Identity()
        print(f"ViT hidden dimension: {hidden_dim}")
        return base_model, hidden_dim

    def _build_dino(self):
        dino_cfg = self.config.get("dinov3", {})
        repo = dino_cfg.get("repo", "facebookresearch/dinov3")
        arch = dino_cfg.get("arch", None)
        weights_path = dino_cfg.get("weights_path", None)
        source = dino_cfg.get("source", "auto")

        if source is None or source == 'auto':
            source = 'local' if os.path.exists(repo) else 'github'

        if arch is None:
            print("Architecture not provided; attempting to detect from weights...")
            if weights_path and os.path.exists(weights_path):
                arch = guess_arch_from_weights(weights_path)
                if arch:
                    print(f"Detected architecture: {arch}")
                else:
                    arch = 'dinov3_vitl16'
                    print(f"Could not detect architecture. Using default: {arch}")
            else:
                arch = 'dinov3_vitl16'
                print(f"No weights path provided. Using default architecture: {arch}")

        print(f"Loading {arch} from {repo} (source={source})")

        if weights_path:
            model = torch.hub.load(repo, arch, source=source, weights=weights_path)
        else:
            model = torch.hub.load(repo, arch, source=source)

        if 'vit7b16' in arch:
            hidden_dim = 4096
        elif 'vitl16' in arch:
            hidden_dim = 1024
        elif 'vitb16' in arch:
            hidden_dim = 768
        elif 'vits16' in arch:
            hidden_dim = 384
        else:
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                if hasattr(model, 'forward_features'):
                    dummy_output = model.forward_features(dummy_input)
                else:
                    dummy_output = model(dummy_input)

                if isinstance(dummy_output, dict):
                    for key in ['x_norm_clstoken', 'x_cls', 'features', 'pooler_output']:
                        if key in dummy_output:
                            hidden_dim = dummy_output[key].shape[-1]
                            break
                    else:
                        hidden_dim = 1024
                else:
                    hidden_dim = dummy_output.shape[-1]
            except:
                hidden_dim = 1024

        print(f"DINO hidden dimension: {hidden_dim}")
        return model, hidden_dim

    # ----- forward ---------------------------------------------------------

    def forward(self, x):
        if self.backbone_type == "dino":
            features = self.backbone(x)
            if isinstance(features, dict) and 'x_norm_patchtokens' in features:
                h = features['x_norm_patchtokens'].mean(dim=1)
            elif isinstance(features, dict) and 'x_norm_clstoken' in features:
                h = features['x_norm_clstoken']
            else:
                h = features
                if h.dim() > 2:
                    h = h.flatten(start_dim=1)
        else:
            h = self.backbone(x).flatten(start_dim=1)

        z = self.projection_head(h)
        return z

    # ----- training --------------------------------------------------------

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss_simclr = self.criterion(z0, z1)

        color_consistency = 0
        if color_consistency > 0:
            color0 = get_mean_lab_color(x0)
            color1 = get_mean_lab_color(x1)
            loss_colour = self.mse_loss(color0, color1)
            loss = loss_simclr + color_consistency * loss_colour
        else:
            loss = loss_simclr
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optimizer_type = self.config.get("optimizer", {}).get("type", "adamw").lower()

        if optimizer_type == "sgd":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=5e-4
            )
        elif optimizer_type == "adam":
            optim = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=5e-4
            )
        elif optimizer_type == "adamw":
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=0.01
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}. Choose from: sgd, adam, adamw")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max)
        return [optim], [scheduler]


# ---------------------------------------------------------------------------
# Embedding generation (backbone-aware)
# ---------------------------------------------------------------------------

def generate_embeddings(model, dataloader):
    from tqdm import tqdm

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in tqdm(dataloader, desc="Generating embeddings", unit="batch"):
            img = img.to(model.device)

            if model.backbone_type == "dino":
                features = model.backbone(img)
                if isinstance(features, dict):
                    if 'x_norm_patchtokens' in features:
                        emb = features['x_norm_patchtokens'].mean(dim=1)
                    elif 'x_norm_clstoken' in features:
                        emb = features['x_norm_clstoken']
                    else:
                        emb = None
                        for key in ['x_cls', 'features', 'pooler_output']:
                            if key in features:
                                emb = features[key]
                                break
                        if emb is None:
                            tensor_values = [v for v in features.values() if isinstance(v, torch.Tensor)]
                            if tensor_values:
                                emb = tensor_values[0]
                            else:
                                raise RuntimeError(
                                    f"Could not extract features from dict output: {list(features.keys())}"
                                )
                elif isinstance(features, (tuple, list)):
                    emb = features[0]
                else:
                    emb = features
                if emb.dim() > 2:
                    emb = emb.flatten(start_dim=1)
            else:
                emb = model.backbone(img).flatten(start_dim=1)

            embeddings.append(emb.cpu())
            filenames.extend(fnames)
            torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def get_image_as_np_array(filename: str):
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, data_dir, n_neighbors=3, num_examples=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    fig, axs = plt.subplots(num_examples, n_neighbors)
    for plot_y_offset, idx in enumerate(samples_idx):
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            ax = axs[plot_y_offset, plot_x_offset]
            fname = os.path.join(data_dir, filenames[neighbor_idx])
            ax.imshow(get_image_as_np_array(fname))
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            ax.axis("off")


def visualize_augmentations(datamodule, num_samples=6, num_augmentations=6):
    datamodule.setup(stage="fit")

    dataset = datamodule.dataset_train
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, num_augmentations + 1,
                            figsize=(3*(num_augmentations + 1), 3*num_samples))
    fig.suptitle("Augmentation Examples", fontsize=16)

    for i, idx in enumerate(indices):
        img_path = dataset.files[idx]
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        for j in range(num_augmentations):
            aug1, aug2 = dataset[idx][0]
            aug_img = aug1.permute(1, 2, 0).numpy()
            aug_img = aug_img * np.array(datamodule.transform_config["normalize"]["std"]) + \
                     np.array(datamodule.transform_config["normalize"]["mean"])
            aug_img = np.clip(aug_img, 0, 1)

            axes[i, j+1].imshow(aug_img)
            axes[i, j+1].set_title(f'Aug {j+1}')
            axes[i, j+1].axis('off')

    plt.tight_layout()

    os.makedirs('augmentation_examples', exist_ok=True)

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'augmentation_examples/augmentations_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration file",
        default="./config.yaml",
    )
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize augmentations and exit')
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    return parser.parse_args()


def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    pl.seed_everything(config["seed"])

    data_dir = config["data"]["data_dir"]

    config_filename = os.path.splitext(os.path.basename(args.config))[0]
    backbone_slug = str(config["model"].get("backbone", "vit")).replace(os.sep, "_").replace("/", "_")

    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec='minutes')
    version = f"{config_filename}_{backbone_slug}_{''.join(re.split(r'[-:]', date_str))}"

    logger = TensorBoardLogger(
        config["output"]["out_dir"],
        name=config["model"]["name"],
        version=version,
    )

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.wandb:
        os.environ["WANDB_CONSOLE"] = "off"
        wandb.init(
            name=f"simclr_{config['model'].get('backbone', 'vit')}_{timestamp}",
            project="simclr",
            dir=os.path.join(config["output"]["out_dir"], "wandb"),
        )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename='{epoch:02d}-{train_loss_ssl:.4f}',
            save_top_k=3,
            monitor='train_loss_ssl',
            mode='min',
            save_last=True,
        ),
    ]

    if args.visualize:
        print("Generating augmentation visualization...")
        datamodule = SimCLRDataModule(config)
        visualize_augmentations(datamodule)
        print("Visualization saved in augmentation_examples/")
        return

    dm = SimCLRDataModule(config)
    model = SimCLRModel(config["model"])

    trainer_config = config.get("train", {})
    default_trainer_config = {
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": 1,
        "log_every_n_steps": 10,
    }
    for key, value in default_trainer_config.items():
        if key not in trainer_config:
            trainer_config[key] = value

    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)

    if args.wandb and wandb.run is not None:
        wandb.config.update(config)

    with open(os.path.join(logger.log_dir, "config.json"), 'w') as f:
        json.dump(config, f)

    dm.setup(stage="predict")
    model.eval()

    print("Generating embeddings...")
    embeddings, filenames = generate_embeddings(model, dm.predict_dataloader())

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f'x{i+1}' for i in range(len(df_embeddings.columns))]
    df_filenames = pd.DataFrame(filenames, columns=['filename'])
    df = pd.concat([df_filenames, df_embeddings], axis=1)
    output_filename = f"embeddings_{version}.csv"
    embeddings_path = os.path.join(logger.log_dir, output_filename)
    df.to_csv(
        embeddings_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        float_format='%.15f')

    plot_knn_examples(embeddings, filenames, data_dir)

    plt.savefig(os.path.join(logger.log_dir, "knn_examples.png"))

    if args.wandb and wandb.run is not None:
        wandb.log({"knn_examples": wandb.Image(os.path.join(logger.log_dir, "knn_examples.png"))})
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)
