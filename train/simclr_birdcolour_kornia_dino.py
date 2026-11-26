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
from lightning.pytorch.loggers import TensorBoardLogger

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform, utils

import kornia
import torchvision.transforms.v2 as v2

import tempfile
# Set temporary directory on same filesystem as checkpoint destination
# tmpdir = "/shared/cooney_lab/Shared/Eleftherios-Ioannou/tmp"
# os.environ["TMPDIR"] = tmpdir
# os.environ["FSSPEC_TEMP"] = tmpdir
# tempfile.tempdir = tmpdir

def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from a variety of checkpoint formats.

    Handles keys like 'state_dict', 'model', 'model_state_dict'. If both teacher/student
    branches exist (as in some DINO-style checkpoints), prefer 'teacher'. Also strips
    common prefixes like 'module.', 'model.', 'backbone.', 'encoder.', 'visual.'.
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

    # Prefer teacher branch if present, else student
    if any(k.startswith('teacher.') for k in keys):
        sd = {k[len('teacher.'):]: v for k, v in sd.items() if k.startswith('teacher.')}
    elif any(k.startswith('student.') for k in keys):
        sd = {k[len('student.'):]: v for k, v in sd.items() if k.startswith('student.')}

    # Strip common wrappers
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
        # Try keys in order of reliability
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

def get_mean_lab_color(x):
    # x: (B, 3, H, W) in range [0, 1] (assumed)
    x = x * 255.0
    x = x.to(torch.uint8)

    mean_lab = []
    for img in x:
        img_np = img.permute(1, 2, 0).cpu().numpy()
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab_mean = lab.mean(axis=(0, 1)) / 255.0  # Normalize to [0, 1]
        mean_lab.append(torch.tensor(lab_mean, dtype=torch.float32))

    return torch.stack(mean_lab).to(x.device)  # Shape: (B, 3)

class KorniaTransform:
    """Custom transform class that applies Kornia transforms twice for SimCLR"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        """Apply the transform twice to get two augmented versions"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Convert to torch tensor and change to CHW format
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)  # Add batch dimension for Kornia
        
        # Apply transforms twice to get two augmented versions
        result1 = self.transform(img)
        result2 = self.transform(img)
        
        return result1.squeeze(0), result2.squeeze(0)

class SimCLRDataset(torch.utils.data.Dataset):
    """Custom dataset that uses Kornia transforms"""
    def __init__(self, input_dir, transform=None):
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform
        
        # Get all image files
        all_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(root, f))
        
        all_files.sort()  # Ensure consistent ordering
        
        # Filter out corrupted/unreadable images
        print(f"Validating {len(all_files)} images...")
        self.files = []
        corrupted_count = 0
        
        for img_path in all_files:
            if self._is_valid_image(img_path):
                self.files.append(img_path)
            else:
                corrupted_count += 1
                print(f"Skipping corrupted image: {img_path}")
        
        print(f"Dataset ready: {len(self.files)} valid images, {corrupted_count} corrupted images skipped")
    
    def _is_valid_image(self, img_path):
        """Check if image can be loaded successfully"""
        try:
            img = cv2.imread(img_path)
            return img is not None and img.size > 0
        except:
            return False
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        
        # Read image using cv2 - should always work since we pre-validated
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        return img, img_path

def collate_fn(batch):
    """Custom collate function to handle tuple of tensors"""
    # Separate the augmented pairs and filenames
    aug_pairs, filenames = zip(*batch)
    # Each aug_pair is already a tuple of (aug1, aug2)
    aug1s, aug2s = zip(*aug_pairs)
    # Stack the tensors
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
        
        # Create augmentation pipeline
        self.train_transform = self._get_train_transforms()
    
    def _get_train_transforms(self):
        """Create training augmentation pipeline using Kornia"""
        cfg = self.transform_config
        transforms = []
        
        if cfg["augmentation_strategy"] == "crop":
            # Random Resize Crop
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
            # Pad the shorter side to make the image square
            transforms.append(
                kornia.augmentation.PadTo(size=(self.input_size, self.input_size))
            )

        # Horizontal Flip
        if cfg["horizontal_flip"]["enabled"]:
            transforms.append(kornia.augmentation.RandomHorizontalFlip(
                p=cfg["horizontal_flip"]["p"]
            ))
        
        # Vertical Flip
        if cfg["vertical_flip"]["enabled"]:
            transforms.append(kornia.augmentation.RandomVerticalFlip(
                p=cfg["vertical_flip"]["p"]
            ))
        
        # Rotate
        if cfg["rotate"]["enabled"]:
            transforms.append(kornia.augmentation.RandomRotation(
                degrees=cfg["rotate"]["limit"],
                p=cfg["rotate"]["p"]
            ))
        
        # Random Gray Scale
        if cfg["grayscale"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGrayscale(
                p=cfg["grayscale"]["p"]
            ))
        
        # Color Jitter
        if cfg["color_jitter"]["enabled"]:
            transforms.append(kornia.augmentation.ColorJitter(
                brightness=cfg["color_jitter"]["brightness"],
                contrast=cfg["color_jitter"]["contrast"],
                saturation=cfg["color_jitter"]["saturation"],
                hue=cfg["color_jitter"]["hue"],
                p=cfg["color_jitter"]["p"]
            ))
        
        # Gaussian Blur
        if cfg["gaussian_blur"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=cfg["gaussian_blur"]["sigma_limit"],
                p=cfg["gaussian_blur"]["p"]
            ))
        
        # Gaussian Noise
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

        # Always add normalization at the end
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
            # Transformation for embedding the dataset after training
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

class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.T_max = config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = config.get('optimizer').get('lr', lr)
        self.temp = config.get('criterion').get('params', {}).get('temperature', 0.5)

        # DinoV3 model configuration
        self.dinov3_config = config.get("dinov3", {})
        self.repo = self.dinov3_config.get("repo", "facebookresearch/dinov3")
        self.arch = self.dinov3_config.get("arch", None)
        self.weights_path = self.dinov3_config.get("weights_path", None)
        self.source = self.dinov3_config.get("source", "auto")

        # Load DinoV3 backbone
        self.backbone, self.hidden_dim = self._load_dinov3_backbone()
        
        self.projection_head = SimCLRProjectionHead(self.hidden_dim, self.hidden_dim, 128)
        self.criterion = NTXentLoss(self.temp)
        # self.mse_loss = torch.nn.MSELoss()

    def _load_dinov3_backbone(self):
        """Load DinoV3 model using torch.hub"""
        device = get_device()
        
        if self.source is None or self.source == 'auto':
            source = 'local' if os.path.exists(self.repo) else 'github'
        else:
            source = self.source

        if self.arch is None:
            print("Architecture not provided; attempting to detect from weights...")
            if self.weights_path and os.path.exists(self.weights_path):
                arch = guess_arch_from_weights(self.weights_path)
                if arch:
                    print(f"Detected architecture: {arch}")
                else:
                    arch = 'dinov3_vitl16'  # Default to vitl16 as requested
                    print(f"Could not detect architecture. Using default: {arch}")
            else:
                arch = 'dinov3_vitl16'
                print(f"No weights path provided. Using default architecture: {arch}")
        else:
            arch = self.arch

        print(f"Loading {arch} from {self.repo} (source={source})")
        
        # Load model via torch.hub
        if self.weights_path:
            model = torch.hub.load(self.repo, arch, source=source, weights=self.weights_path)
        else:
            model = torch.hub.load(self.repo, arch, source=source)

        # Determine hidden dimension based on architecture
        if 'vit7b16' in arch:
            hidden_dim = 4096
        elif 'vitl16' in arch:
            hidden_dim = 1024
        elif 'vitb16' in arch:
            hidden_dim = 768
        elif 'vits16' in arch:
            hidden_dim = 384
        else:
            # Fallback: try to infer from model
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                if hasattr(model, 'forward_features'):
                    dummy_output = model.forward_features(dummy_input)
                else:
                    dummy_output = model(dummy_input)
                
                if isinstance(dummy_output, dict):
                    # Try common keys for features
                    for key in ['x_norm_clstoken', 'x_cls', 'features', 'pooler_output']:
                        if key in dummy_output:
                            hidden_dim = dummy_output[key].shape[-1]
                            break
                    else:
                        hidden_dim = 1024  # Default fallback
                else:
                    hidden_dim = dummy_output.shape[-1]
            except:
                hidden_dim = 1024  # Default fallback
        
        print(f"Using hidden dimension: {hidden_dim}")
        return model, hidden_dim

    # def forward(self, x):
    #     # Extract features from DinoV3
    #     h = self.backbone(x)
        
    #     # Ensure proper shape (flatten if needed)
    #     if h.dim() > 2:
    #         h = h.flatten(start_dim=1)
        
    #     z = self.projection_head(h)
    #     return z

    def forward(self, x):
        # Extract features from DinoV3
        features = self.backbone(x)
        
        # Use patch tokens with mean pooling for spatial information
        if isinstance(features, dict) and 'x_norm_patchtokens' in features:
            h = features['x_norm_patchtokens'].mean(dim=1)  # (B, num_patches, D) -> (B, D)
        elif isinstance(features, dict) and 'x_norm_clstoken' in features:
            h = features['x_norm_clstoken']  # Fallback to CLS token
        else:
            h = features
            if h.dim() > 2:
                h = h.flatten(start_dim=1)
        
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss_simclr = self.criterion(z0, z1)

        # Colour consistency loss (L2 distance)
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
        # Get optimizer type from config, default to AdamW for self-supervised learning
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
                weight_decay=0.01  # AdamW typically uses higher weight decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}. Choose from: sgd, adam, adamw")
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max)
        return [optim], [scheduler]

def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """
    from tqdm import tqdm

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img,_, fnames in tqdm(dataloader, desc="Generating embeddings", unit="batch"):
            img = img.to(model.device)
            
            # Extract features from DinoV3 backbone
            features = model.backbone(img)
            
            # Handle different output formats (same logic as in forward method)
            if isinstance(features, dict):
                if 'x_norm_patchtokens' in features:
                    emb = features['x_norm_patchtokens'].mean(dim=1)  # Mean pool patch tokens
                elif 'x_norm_clstoken' in features:
                    emb = features['x_norm_clstoken']  # Fallback
                else:
                    # Try other common keys
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
                            raise RuntimeError(f"Could not extract features from dict output: {list(features.keys())}")
            elif isinstance(features, (tuple, list)):
                emb = features[0]
            else:
                emb = features
            
            # Ensure proper shape
            if emb.dim() > 2:
                emb = emb.flatten(start_dim=1)
            
            # Move to CPU immediately to free GPU memory
            embeddings.append(emb.cpu())
            filenames.extend(fnames)
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)

def plot_knn_examples(embeddings, filenames, data_dir, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    fig, axs = plt.subplots(num_examples, n_neighbors)
    # loop through our randomly picked samples
    for plot_y_offset, idx in enumerate(samples_idx):
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = axs[plot_y_offset,
                     plot_x_offset]
            # get the correponding filename for the current index
            fname = os.path.join(data_dir, filenames[neighbor_idx])
            # plot the image
            ax.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            ax.axis("off")

def visualize_augmentations(datamodule, num_samples=6, num_augmentations=6):
    """
    Visualize augmentations applied to random samples from the dataset.
    
    Args:
        datamodule: SimCLRDataModule instance
        num_samples: Number of different images to show
        num_augmentations: Number of augmentations per image
    """
    # Setup the datamodule
    datamodule.setup(stage="fit")
    
    # Get random samples
    dataset = datamodule.dataset_train
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Create subplot grid
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, 
                            figsize=(3*(num_augmentations + 1), 3*num_samples))
    fig.suptitle("Augmentation Examples", fontsize=16)
    
    for i, idx in enumerate(indices):
        # Get original image
        img_path = dataset.files[idx]
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Show original
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Show augmentations
        for j in range(num_augmentations):
            # Get augmented pair
            aug1, aug2 = dataset[idx][0]
            # Convert from tensor to numpy and adjust dimensions
            aug_img = aug1.permute(1, 2, 0).numpy()
            # Denormalize
            aug_img = aug_img * np.array(datamodule.transform_config["normalize"]["std"]) + \
                     np.array(datamodule.transform_config["normalize"]["mean"])
            aug_img = np.clip(aug_img, 0, 1)
            
            axes[i, j+1].imshow(aug_img)
            axes[i, j+1].set_title(f'Aug {j+1}')
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('augmentation_examples', exist_ok=True)
    
    # Save the figure
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'augmentation_examples/augmentations_{timestamp}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def get_args():
    """
    Parse command line arguments
    ---------
    Returns:
        argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration file",
        default="./config.yaml",
    )
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize augmentations and exit')
    return parser.parse_args()

def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Trade-off precision for performance when using CUDA device that has Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    # Let's set the seed for our experiments
    pl.seed_everything(config["seed"])

    data_dir = config["data"]["data_dir"]

    # Extract the name of the config file without the directory path or extension
    config_filename = os.path.splitext(os.path.basename(args.config))[0]

    # Create a timestamped version string that includes the config filename
    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec='minutes')
    version = f"{config_filename}_dinov3_{''.join(re.split(r'[-:]', date_str))}"

    # Set up logger
    logger = TensorBoardLogger(config["output"]["out_dir"],
                               name=config["model"]["name"],
                               version=version)

    if args.visualize:
        print("Generating augmentation visualization...")
        datamodule = SimCLRDataModule(config)
        visualize_augmentations(datamodule)
        print("Visualization saved in augmentation_examples/")
        return

    # Set up dataloader and model for training
    dm = SimCLRDataModule(config)
    model = SimCLRModel(config["model"])
    
    # Handle callbacks from config
    callbacks = []
    if "callbacks" in config["train"]:
        from pytorch_lightning.callbacks import ModelCheckpoint
        for callback_config in config["train"]["callbacks"]:
            if callback_config.get("class_path") == "pytorch_lightning.callbacks.ModelCheckpoint":
                callbacks.append(ModelCheckpoint(**callback_config["init_args"]))
    
    # Remove callbacks from train config to avoid passing them twice
    train_config = config["train"].copy()
    train_config.pop("callbacks", None)
    
    trainer = pl.Trainer(**train_config, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=dm)

    # Log configuration
    with open(os.path.join(logger.log_dir, "config.json"), 'w') as f:
        json.dump(config, f)

    # Set up dataloader and model for inference then generate embeddings
    dm.setup(stage="predict")
    model.eval()

    print("Generating embeddings...")
    embeddings, filenames = generate_embeddings(model, dm.predict_dataloader())

    # Save embeddings to output dir
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f'x{i+1}' for i in range(len(df_embeddings.columns))]
    df_filenames = pd.DataFrame(filenames, columns=['filename'])
    df = pd.concat([df_filenames, df_embeddings], axis=1)
    output_filename = f"embeddings_{version}.csv"
    df.to_csv(
        os.path.join(logger.log_dir, output_filename),
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        float_format='%.15f')

    # Create sample plots
    plot_knn_examples(embeddings, filenames, data_dir)

    # Save plots to output dir
    plt.savefig(os.path.join(logger.log_dir, "knn_examples.png"))

if __name__ == "__main__":
    args = get_args()
    main(args)