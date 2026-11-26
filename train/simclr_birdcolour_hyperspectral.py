import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import datetime as dt
import json
import os
import sys
# Add the parent directory of 'train' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import re
import yaml
import glob
from typing import Dict, Any, List, Tuple, Optional, Union

# Dynamically add the 'lepidoptera' directory to sys.path to import 'read_iml_hyp'
_script_dir = os.path.dirname(os.path.abspath(__file__))
_lepidoptera_dir = os.path.abspath(os.path.join(_script_dir, '..', 'lepidoptera'))
if _lepidoptera_dir not in sys.path:
    sys.path.insert(0, _lepidoptera_dir)

try:
    from read_iml_hyp import read_iml_hyp
except ImportError as e:
    print(f"ERROR: Could not import 'read_iml_hyp' from '{_lepidoptera_dir}'. "
          f"Please ensure 'read_iml_hyp.py' exists in that directory and it's a Python module. Details: {e}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import cv2

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import rasterio
from rasterio.plot import reshape_as_image

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from pytorch_lightning.loggers import TensorBoardLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Import Kornia augmentations
import kornia.augmentation as K
from kornia.constants import BorderType
from kornia.geometry.transform import get_perspective_transform, warp_perspective
import kornia 

from data.hyperspectral_data import HyperspectralDataModule, HyperspectralBilDataset
import pytz

import tempfile
# Set temporary directory on same filesystem as checkpoint destination
# tmpdir = "/shared/cooney_lab/Shared/Eleftherios-Ioannou/tmp"
# os.environ["TMPDIR"] = tmpdir
# os.environ["FSSPEC_TEMP"] = tmpdir
# tempfile.tempdir = tmpdir

import wandb


class SpectralAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_dim = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Global Average Pooling → (B, C)
        avg_pool = nn.functional.adaptive_avg_pool2d(x, 1).view(B, C)

        # MLP → Attention weights
        attn = self.fc1(avg_pool)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn).view(B, C, 1, 1)

        # Reweight original input
        return x * attn


def modify_vit_input_channels(model: nn.Module, in_channels: int):
    """Modify ViT patch embedding to accept in_channels instead of RGB."""
    patch_embed = model.conv_proj  # Conv2d(3, hidden_dim, kernel_size=16, stride=16)
    new_patch_embed = nn.Conv2d(in_channels, patch_embed.out_channels,
                                kernel_size=patch_embed.kernel_size,
                                stride=patch_embed.stride,
                                padding=patch_embed.padding,
                                bias=patch_embed.bias is not None)
    
    model.conv_proj = new_patch_embed
    return model


class MultispectralBackbone(nn.Module):
    """Custom backbone for high-channel-count images (hyperspectral)."""
    
    def __init__(self, base_model, in_channels=6, checkpoint_path=None):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 3, kernel_size=1)
        )

        self.backbone = base_model

        if checkpoint_path:
            print(f"Loading pretrained weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                adapter_key = 'model.adapter.weight'
                if adapter_key in state_dict:
                    self.adapter.weight.data = state_dict[adapter_key]
                    print("Successfully loaded adapter weights")
                
                loaded_params = 0
                total_params = 0
                for name, param in self.backbone.named_parameters():
                    total_params += 1
                    checkpoint_key = f'model.original_model.{name}'
                    if checkpoint_key in state_dict:
                        param.data = state_dict[checkpoint_key]
                        loaded_params += 1
                
                print(f"Successfully loaded backbone weights ({loaded_params}/{total_params} parameters)")
    
    def forward(self, x):
        x = self.adapter(x)
        
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = x.contiguous()
        x = self.backbone(x)
        
        return x


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


class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5, in_channels: int = 408):
        super().__init__()
        self.save_hyperparameters()
        self.T_max = config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = config.get('optimizer').get('lr', lr)
        self.temp = config.get('criterion').get('params', {}).get('temperature', 0.5)
        self.spectral_attn = None 
        
        # Get loss configuration
        criterion_config = config.get('criterion', {})
        self.loss_type = criterion_config.get('type', 'ntxent')
        self.global_weight = criterion_config.get('global_weight', 1.0)  
        self.local_weight = criterion_config.get('local_weight', 0.0)
        
        print(f"Using loss={self.loss_type} with weights: Global NTXent={self.global_weight}, Local NTXent={self.local_weight}")

        self.use_local_loss = criterion_config.get('use_local_loss', False)  
        print(f"Using local loss: {self.use_local_loss}")
        
        model = config.get("backbone")
        self.is_dino = False

        if model.startswith('resnet') or model == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            hidden_dim = base_model.fc.in_features
            base_model.fc = nn.Flatten()
            self.is_vit = False
        elif model.startswith('dino') or model == 'dinov3':
            self.dinov3_config = config.get("dinov3", {})
            self.is_vit = True
            self.is_dino = True
            self.patch_size = 16
            base_model, hidden_dim = self._load_dinov3_backbone()
        else:
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            
            weights = ViT_L_16_Weights.IMAGENET1K_V1
            base_model = vit_l_16(weights=weights, image_size=224)
            
            base_model.heads = nn.Identity()
            
            hidden_dim = 1024
            self.is_vit = True
            self.patch_size = 16
        
        # For hyperspectral, always use single encoder with adapter
        if self.is_dino:
            if in_channels > 3:
                self.backbone = MultispectralBackbone(base_model, in_channels=in_channels)
            else:
                self.backbone = base_model
        elif in_channels > 3:
            checkpoint_path = config.get('checkpoint_path', None)
            if model.startswith('resnet') or model == 'resnet50':
                self.backbone = MultispectralBackbone(base_model, in_channels=in_channels, checkpoint_path=checkpoint_path)
            else:
                base_model = modify_vit_input_channels(base_model, in_channels=in_channels)
                self.backbone = base_model
        else:
            self.backbone = base_model
        
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        
        if self.is_vit and self.use_local_loss:
            self.local_projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 128)
            )
        
        self.ntxent_criterion = NTXentLoss(self.temp)

    def _load_dinov3_backbone(self):
        """Load DINOv3 model using torch.hub"""
        repo = self.dinov3_config.get("repo", "facebookresearch/dinov3")
        arch = self.dinov3_config.get("arch", None)
        weights_path = self.dinov3_config.get("weights_path", None)
        source = self.dinov3_config.get("source", "auto")

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
        
        print(f"DINOv3 backbone loaded with hidden dimension: {hidden_dim}")
        return model, hidden_dim

    def _extract_dino_features(self, backbone, x):
        """Extract features from a DINOv3 backbone, handling dict outputs.
        
        Returns:
            h: Global feature vector [batch_size, hidden_dim]
            patch_tokens: Patch token features [batch_size, num_patches, hidden_dim] or None
        """
        features = backbone(x)
        
        patch_tokens = None
        if isinstance(features, dict):
            if 'x_norm_patchtokens' in features:
                patch_tokens = features['x_norm_patchtokens']
                h = patch_tokens.mean(dim=1)
            elif 'x_norm_clstoken' in features:
                h = features['x_norm_clstoken']
            else:
                h = None
                for key in ['x_cls', 'features', 'pooler_output']:
                    if key in features:
                        h = features[key]
                        break
                if h is None:
                    tensor_values = [v for v in features.values() if isinstance(v, torch.Tensor)]
                    if tensor_values:
                        h = tensor_values[0]
                    else:
                        raise RuntimeError(f"Could not extract features from DINO output: {list(features.keys())}")
        else:
            h = features
            if h.dim() > 2:
                h = h.flatten(start_dim=1)
        
        return h, patch_tokens

    def _extract_patch_tokens(self, x):
        """Extract patch tokens from ViT model before class token aggregation"""
        if not self.is_vit:
            return None
            
        n, c, h, w = x.shape
        p = self.patch_size
        
        x = self.backbone.conv_proj(x)
        n_h = h // p
        n_w = w // p
        x = x.reshape(n, self.backbone.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        x = self.backbone.encoder.layers(x)
        x = self.backbone.encoder.ln(x)
        
        patch_tokens = x[:, 1:, :]
        
        return patch_tokens

    def forward(self, x):
        x = x.contiguous()
        
        if self.spectral_attn is not None:
            x = self.spectral_attn(x)
        
        patch_tokens = None
        
        if self.is_dino:
            h, patch_tokens = self._extract_dino_features(self.backbone, x)
            if not self.use_local_loss:
                patch_tokens = None
        else:
            if self.is_vit and self.use_local_loss:
                patch_tokens = self._extract_patch_tokens(x)
            
            h = self.backbone(x)
        
        h = h.contiguous()
        z = self.projection_head(h)
        
        local_z = None
        if patch_tokens is not None:
            local_z = self.local_projection_head(patch_tokens)
            
        return z, local_z

    def compute_local_loss(self, local_z1, local_z2):
        """Compute NTXent loss at the patch level."""
        batch_size, num_patches, proj_dim = local_z1.shape
        
        local_loss = 0.0
        
        for i in range(num_patches):
            p1_i = local_z1[:, i, :]
            p2_i = local_z2[:, i, :]
            
            patch_loss = self.ntxent_criterion(p1_i, p2_i)
            local_loss += patch_loss
            
        local_loss = local_loss / num_patches

        return local_loss

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        
        x0 = x0.contiguous()
        x1 = x1.contiguous()
        
        z0, local_z0 = self.forward(x0)
        z1, local_z1 = self.forward(x1)
        
        global_loss = self.ntxent_criterion(z0, z1)
        self.log("train_global_loss", global_loss)
        
        total_loss = self.global_weight * global_loss
        
        if self.local_weight > 0 and self.use_local_loss and local_z0 is not None and local_z1 is not None:
            local_loss = self.compute_local_loss(local_z0, local_z1)
            self.log("train_local_loss", local_loss)
            total_loss += self.local_weight * local_loss
            
        self.log("train_loss_ssl", total_loss)
        
        if wandb.run is not None:
            log_dict = {
                'train_loss_ssl': total_loss,
                'train_loss_global': global_loss,
                'train_loss_local': local_loss if self.use_local_loss else 0,
                'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                'epoch': self.current_epoch + 1
            }
            wandb.log(log_dict)
        
        return total_loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max)


        if self.is_dino:
            # switch to adamw optimizer with a lower learning rate
            optim = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr/10,
                weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max/10)
            
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]
        
    def on_train_epoch_end(self):
        current_epoch = self.current_epoch + 1
        avg_loss = self.trainer.callback_metrics.get('train_loss_ssl', None)
        
        if wandb.run is not None:
            metrics = {
                'epoch': current_epoch,
                'epoch_avg_loss': avg_loss
            }
            
            global_loss = self.trainer.callback_metrics.get('train_global_loss', None)
            if global_loss is not None:
                metrics['epoch_avg_global_loss'] = global_loss
                
            if self.use_local_loss:
                local_loss = self.trainer.callback_metrics.get('train_local_loss', None)
                if local_loss is not None:
                    metrics['epoch_avg_local_loss'] = local_loss
            
            wandb.log(metrics, commit=True)
            
            if avg_loss is not None and (wandb.run.summary.get('best_epoch_loss', float('inf')) > avg_loss):
                wandb.run.summary['best_epoch_loss'] = avg_loss
                wandb.run.summary['best_epoch'] = current_epoch


def generate_embeddings(model, dataloader):
    """Generate embeddings for all images in the dataloader."""
    model.eval()
    embeddings = []
    filenames = []
    
    with torch.no_grad():
        for batch in dataloader:
            img_pair, batch_filenames = batch
            
            if isinstance(img_pair, tuple):
                img = img_pair[0]
            else:
                img = img_pair
                
            img = img.to(model.device)
            
            if model.is_dino:
                embedding, _ = model._extract_dino_features(model.backbone, img)
            else:
                embedding = model.backbone(img)
            
            if embedding.dim() > 2:
                embedding = embedding.flatten(start_dim=1)
            
            embeddings.append(embedding.cpu())
            filenames.extend(batch_filenames)
    
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings, filenames


def get_image_as_np_array(filename: str):
    """Returns an image as a numpy array, supporting common formats and .bil hyperspectral."""
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        try:
            img = Image.open(filename)
            if img.mode not in ['RGB', 'L', 'RGBA']:
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"Error opening image file {filename} with PIL: {e}")
            return None
    
    elif filename.lower().endswith(".bil"):
        try:
            from spectral import open_image
            hdr_file = filename + '.hdr'
            if not os.path.exists(hdr_file):
                hdr_file = filename[:-4] + '.hdr'

            if os.path.exists(hdr_file):
                img_spectral = open_image(hdr_file)
                img_data = img_spectral.read_bands(range(img_spectral.nbands))
                if img_data.shape[0] == img_spectral.nbands and img_data.ndim == 3:
                    img_data = np.transpose(img_data, (1, 2, 0))
                return img_data
            else:
                print(f"Warning: BIL file {filename} found, but no corresponding .hdr file. Falling back to read_iml_hyp.")
                directory = os.path.dirname(filename)
                base_name = os.path.basename(filename)[:-4]
                img_data, _, _, _ = read_iml_hyp(directory, base_name)
                return img_data

        except ImportError:
            print("INFO: 'spectral' module not found. Falling back to custom BIL reader (read_iml_hyp) for .bil files.")
            directory = os.path.dirname(filename)
            base_name = os.path.basename(filename)[:-4]
            img_data, _, _, _ = read_iml_hyp(directory, base_name)
            return img_data
        except Exception as e:
            print(f"Error processing BIL file {filename}: {e}")
            try:
                directory = os.path.dirname(filename)
                base_name = os.path.basename(filename)[:-4]
                img_data, _, _, _ = read_iml_hyp(directory, base_name)
                return img_data
            except Exception as fallback_e:
                print(f"Final fallback with read_iml_hyp also failed for {filename}: {fallback_e}")
                return None

    else:
        try:
            directory = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            img_data, _, _, _ = read_iml_hyp(directory, base_name)
            return img_data
        except Exception as e:
            print(f"Error loading file {filename} using read_iml_hyp: {e}")
            return None
    
    print(f"Warning: Could not determine how to load file: {filename}")
    return None


def plot_knn_examples(embeddings, filenames, data_dir, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors"""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    
    fig, axs = plt.subplots(num_examples, n_neighbors, figsize=(12, 2*num_examples))
    for plot_y_offset, idx in enumerate(samples_idx):
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            ax = axs[plot_y_offset, plot_x_offset]
            fname = os.path.join(data_dir, filenames[neighbor_idx])
            
            try:
                img_data = get_image_as_np_array(fname)
                
                processed_img_for_display = None
                if img_data is not None and img_data.size > 0:
                    if img_data.ndim == 3 and img_data.shape[2] > 4:
                        num_bands = img_data.shape[2]
                        b_idx = min(num_bands - 1, int(num_bands * 0.3))
                        g_idx = min(num_bands - 1, int(num_bands * 0.5))
                        r_idx = min(num_bands - 1, int(num_bands * 0.7))
                        processed_img_for_display = img_data[:, :, [r_idx, g_idx, b_idx]]
                    elif img_data.ndim == 3 and img_data.shape[2] == 1:
                        processed_img_for_display = img_data[:, :, 0]
                    else:
                        processed_img_for_display = img_data
                else:
                    processed_img_for_display = np.zeros((50, 50, 3), dtype=np.uint8)
                
                if processed_img_for_display.dtype != np.uint8 and processed_img_for_display.size > 0:
                    max_val = np.max(processed_img_for_display)
                    if max_val > 0:
                        processed_img_for_display = (processed_img_for_display / max_val * 255).astype(np.uint8)
                    else:
                        processed_img_for_display = np.zeros_like(processed_img_for_display, dtype=np.uint8)
                
                ax.imshow(processed_img_for_display)
            except Exception as e:
                print(f"Failed to display image {fname}: {e}")
                ax.text(0.5, 0.5, "Image load error", ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('lightgray')
                
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            ax.axis("off")
    
    plt.tight_layout()
    return fig


def visualize_augmentations(datamodule, num_samples=5, save_dir="augmentation_examples"):
    """Visualize augmentations applied to images from the hyperspectral dataset."""
    import os
    from tqdm import tqdm
    
    os.makedirs(save_dir, exist_ok=True)
    datamodule.setup()
    dataset = datamodule.train_dataset
    
    def convert_to_rgb(tensor):
        """Convert hyperspectral tensor to pseudo-RGB for visualization"""
        if tensor.shape[0] == 3:
            return tensor
        
        channels = tensor.shape[0]
        red_idx = min(channels // 6, channels - 1)
        green_idx = min(channels // 2, channels - 1)
        blue_idx = min(5 * channels // 6, channels - 1)
        
        rgb = torch.stack([
            tensor[red_idx],
            tensor[green_idx],
            tensor[blue_idx]
        ], dim=0)
        
        for i in range(3):
            if rgb[i].max() != rgb[i].min():
                rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())
            else:
                rgb[i] = torch.zeros_like(rgb[i])
        
        return rgb
    
    print(f"Generating augmentation examples for {num_samples} random samples...")
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]
        
        if isinstance(sample, tuple) and len(sample) == 2:
            first_element, filename = sample
            
            if isinstance(first_element, tuple) and len(first_element) == 2:
                view1, view2 = first_element
                
                view1_rgb = convert_to_rgb(view1)
                view2_rgb = convert_to_rgb(view2)
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.title(f"Augmentation 1")
                plt.imshow(view1_rgb.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.title(f"Augmentation 2")
                plt.imshow(view2_rgb.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                
                file_id = os.path.splitext(os.path.basename(filename))[0]
                save_path = os.path.join(save_dir, f"aug_sample_{file_id}.png")
                plt.suptitle(f"Sample: {file_id}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                print(f"Skipping sample {i} - not in SimCLR format")
        else:
            print(f"Skipping sample {i} - unexpected format")
    
    print(f"Augmentation examples saved to {save_dir}/")


def get_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="SimCLR training script for hyperspectral data. "
                    "Supports all channel configurations and loss types."
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration file",
        default="./config.yaml",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Use only RGB channels (bands closest to RGB wavelengths)",
    )
    parser.add_argument(
        "--sample-bands",
        type=str,
        help="Comma-separated list of band indices to sample from hyperspectral data (e.g., '0,50,100,150,200')",
        default=None,
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        help="Path to pretrained checkpoint to use as backbone",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize augmentations applied to random samples from the dataset",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=["global", "local", "combined"],
        help="Loss function to use (global, local, or combined). Defaults to YAML config or 'global'.",
    )
    parser.add_argument(
        "--global-weight",
        type=float,
        default=None,
        help="Weight for global NTXentLoss. Defaults to YAML config or 1.0.",
    )
    parser.add_argument(
        "--local-weight",
        type=float,
        default=None,
        help="Weight for local NTXentLoss. Defaults to YAML config or 1.0.",
    )
    parser.add_argument(
        "--test-embeddings-only",
        action="store_true",
        help="Skip training and only generate embeddings for testing",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--augmentation-strategy",
        type=str,
        default=None,
        choices=["crop", "resize"],
        help="Augmentation strategy to use ('crop' or 'resize'). Defaults to YAML config or 'crop'.",
    )
    return parser.parse_args()

def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    # --- Determine input channels based on channel selection ---
    sample_bands = None
    if args.sample_bands:
        try:
            sample_bands = [int(band) for band in args.sample_bands.split(',')]
            print(f"Using specific bands: {sample_bands} ({len(sample_bands)} channels)")
            config["data"]["sample_bands"] = sample_bands
            in_channels = len(sample_bands)
        except ValueError:
            print("Error parsing sample bands. Make sure it's a comma-separated list of integers.")
            raise
    elif args.rgb_only:
        print("RGB-like mode enabled: Using only bands closest to RGB wavelengths (3 channels)")
        config["data"]["rgb_only"] = True
        config["model"]["rgb_only"] = True
        in_channels = 3
    else:
        print("Using all hyperspectral bands")
        temp_dataset = HyperspectralBilDataset(
            input_dir=config["data"]["data_dir"],
            transform=None,
            rgb_only=False,
            normalize=True
        )
        try:
            first_item, _ = temp_dataset[0]
            actual_channels = first_item.shape[0]
            print(f"Detected {actual_channels} channels from the hyperspectral dataset")
            in_channels = actual_channels
        except Exception as e:
            print(f"Could not determine channels from dataset: {e}")
            print("Falling back to default of 408 channels")
            in_channels = 408
        
    # Print backbone info
    backbone_name = config["model"].get("backbone", "vit_l_16")
    if backbone_name.startswith('dinov3') or backbone_name == 'dino':
        dinov3_cfg = config["model"].get("dinov3", {})
        print(f"Backbone: DINOv3 (arch={dinov3_cfg.get('arch', 'auto-detect')}, "
              f"repo={dinov3_cfg.get('repo', 'facebookresearch/dinov3')})")
    else:
        print(f"Backbone: {backbone_name}")

    # Add model checkpoint path to config if specified
    if args.model_checkpoint:
        print(f"Using pretrained weights from: {args.model_checkpoint}")
        config["model"]["checkpoint_path"] = args.model_checkpoint

    # --- Merge CLI args with YAML config (CLI overrides YAML, YAML is the default) ---
    if 'criterion' not in config["model"]:
        config["model"]['criterion'] = {}
    criterion_cfg = config["model"]['criterion']

    if args.loss_type is not None:
        criterion_cfg['type'] = args.loss_type
    loss_type = criterion_cfg.get('type', 'global')

    if args.global_weight is not None:
        criterion_cfg['global_weight'] = args.global_weight
    if args.local_weight is not None:
        criterion_cfg['local_weight'] = args.local_weight

    global_weight = criterion_cfg.get('global_weight', 1.0)
    local_weight = criterion_cfg.get('local_weight', 1.0)

    if loss_type == 'combined':
        print(f"Using combined loss with weights: Global NTXent={global_weight}, Local NTXent={local_weight}")
    else:
        print(f"Using {loss_type} loss")

    # Augmentation strategy: CLI > YAML > 'crop'
    if args.augmentation_strategy is not None:
        config.setdefault("augmentations", {})["strategy"] = args.augmentation_strategy
    elif "strategy" not in config.get("augmentations", {}):
        config.setdefault("augmentations", {})["strategy"] = "crop"

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    pl.seed_everything(config["seed"])

    data_dir = config["data"]["data_dir"]

    config_filename = os.path.splitext(os.path.basename(args.config))[0]

    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec='minutes')
    version = f"{config_filename}_{''.join(re.split(r'[-:]', date_str))}"
    
    logger = TensorBoardLogger(config["output"]["out_dir"],
                               name=config["model"]["name"],
                               version=version)
    
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize wandb if enabled
    if args.wandb:
        os.environ["WANDB_CONSOLE"] = "off"
        wandb.init(
            name=f"simclr_hyp_{config['model'].get('backbone', 'vit')}_{timestamp}",
            project="simclr",
            dir=os.path.join(config["output"]["out_dir"], "wandb"),
        )
    
    # Add callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=os.path.join(config["output"]["out_dir"], "checkpoints", version),
            filename='{epoch:02d}-{train_loss_ssl:.4f}',
            save_top_k=3,
            monitor='train_loss_ssl',
            mode='min'
        )
    ]
    
    if args.visualize:
        print("Generating augmentation visualization...")
        datamodule = HyperspectralDataModule(config)
        visualize_augmentations(datamodule)
        print("Visualization saved in augmentation_examples/")
        return

    # Set up dataloader and model for training
    dm = HyperspectralDataModule(
        config,
        rgb_only=args.rgb_only,
        sample_bands=sample_bands,
        augmentation_strategy=args.augmentation_strategy,
    )
    model = SimCLRModel(config["model"], in_channels=in_channels)
    
    # Set up trainer with default values if not specified
    trainer_config = config.get("train", {})
    default_trainer_config = {
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": 1,
        "log_every_n_steps": 10
    }
    
    for key, value in default_trainer_config.items():
        if key not in trainer_config:
            trainer_config[key] = value
    
    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)

    if not args.test_embeddings_only:
        print("Starting training...")
        trainer.fit(model, datamodule=dm)
        print("Training finished.")
    else:
        print("Skipping training, proceeding to generate embeddings.")

    # Save config to wandb
    if args.wandb and wandb.run is not None:
        wandb.config.update(config)
    
    # Log configuration
    with open(os.path.join(logger.log_dir, "config.json"), 'w') as f:
        json.dump(config, f)
    
    # Generate embeddings
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
    embedding_path = os.path.join(logger.log_dir, output_filename)
    print(f"Embeddings will be saved to: {embedding_path}")
    df.to_csv(
        embedding_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        float_format='%.15f')

    print('Embeddings generated...')
    print('Plotting KNN examples...')
    plot_knn_examples(embeddings, filenames, data_dir)

    plt.savefig(os.path.join(logger.log_dir, "knn_examples.png"))
    
    if args.wandb and wandb.run is not None:
        wandb.log({"knn_examples": wandb.Image(os.path.join(logger.log_dir, "knn_examples.png"))})
        wandb.finish()


def test_model_with_dummy_input(model, in_channels=408, input_size=224, device=None):
    """Test if the model works correctly with a dummy input."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model with dummy input (channels={in_channels}, size={input_size}x{input_size})...")
    
    dummy_input = torch.randn(1, in_channels, input_size, input_size)
    dummy_input = dummy_input.to(device)
    
    model.eval()
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Full model output shape: {output[0].shape}")
            print(f"Output contains NaN: {torch.isnan(output[0]).any().item()}")
        
        print("Model test successful!")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False


if __name__ == "__main__":
    args = get_args()
    main(args)
