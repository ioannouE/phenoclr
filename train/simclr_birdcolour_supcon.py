import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import datetime as dt
import json
import os
import re
import yaml

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
from lightly.loss import NTXentLoss  # Changed from SupConLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform, utils

import kornia
import torchvision.transforms.v2 as v2

import tempfile

class SupConLoss(nn.Module):
    """
    Computes the Supervised Contrastive Loss as described in the paper
    "Supervised Contrastive Learning" (https://arxiv.org/pdf/2004.11362.pdf)
    and supports the unsupervised contrastive loss in SimCLR.

    The contrastive loss encourages the embeddings to be close to their positive
    samples and far away from negative samples. It measures the similarity between
    two samples by the dot product of their embeddings and apply a temperature
    scaling.

    Args:
        temperature (float, optional): The temperature scaling.Default: `0.07`.
        contrast_mode (str, optional): Specifies the mode to compute contrastive loss.
            There are two modes: `all` and `one`. In `all` mode, every sample is used
            as an anchor. In `one` mode, only the first is used as an anchor.
            Default: `'all'`.
        base_temperature (float, optional): The base temperature used to normalize the
            temperature. Default: `0.07`.
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Debug: Check input features
        # print(f"SupConLoss DEBUG: batch_size={batch_size}, features.shape={features.shape}")
        # print(f"Features min/max: {features.min().item():.6f}/{features.max().item():.6f}")

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # L2-normalize for cosine similarity (stability), then compute logits
        anchor_feature = nn.functional.normalize(anchor_feature, dim=1)
        contrast_feature = nn.functional.normalize(contrast_feature, dim=1)
        # compute logits
        anchor_dot_contrast = (
            torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob with masked log-sum-exp for numerical stability
        logits_den = logits.masked_fill(logits_mask == 0, float('-inf'))
        log_prob = logits - torch.logsumexp(logits_den, dim=1, keepdim=True)

        # compute mean of log-likelihood over positive
        pos_counts = mask.sum(1).clamp_min(1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(f"Final loss: {loss.item():.6f}")
        return loss

def extract_class_label_from_filename(filename):
    """
    Extracts a class label (species_sex) from a filename.
    """
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    if len(parts) >= 4 and parts[3] in ['M', 'F', 'U']:
        species = f"{parts[0]}_{parts[1]}"
        sex = parts[3]
        return f"{species}_{sex}"
    if len(parts) > 2:
        species = "_".join(parts[:-2])
        return species
    return base_name

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
    """Custom dataset that uses Kornia transforms and returns labels."""
    def __init__(self, input_dir, transform=None):
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform
        self.files = []
        self.labels = []
        all_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(root, f))
        all_files.sort()
        self.class_to_idx = {}
        for f in all_files:
            class_name = extract_class_label_from_filename(f)
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            self.files.append(f)
            self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, img_path, label

def collate_fn(batch):
    """Custom collate function to handle images, filenames, and labels."""
    aug_pairs, filenames, labels = zip(*batch)
    aug1s, aug2s = zip(*aug_pairs)
    aug1s = torch.stack(aug1s)
    aug2s = torch.stack(aug2s)
    labels = torch.tensor(labels, dtype=torch.long)
    return (aug1s, aug2s), filenames, labels

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
            transforms.append(kornia.augmentation.Resize(size=self.input_size, side='long'))
            transforms.append(kornia.augmentation.PadTo(size=(self.input_size, self.input_size)))
        if cfg["horizontal_flip"]["enabled"]:
            transforms.append(kornia.augmentation.RandomHorizontalFlip(p=cfg["horizontal_flip"]["p"]))
        if cfg["vertical_flip"]["enabled"]:
            transforms.append(kornia.augmentation.RandomVerticalFlip(p=cfg["vertical_flip"]["p"]))
        if cfg["rotate"]["enabled"]:
            transforms.append(kornia.augmentation.RandomRotation(degrees=cfg["rotate"]["limit"], p=cfg["rotate"]["p"]))
        if cfg["grayscale"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGrayscale(p=cfg["grayscale"]["p"]))
        if cfg["color_jitter"]["enabled"]:
            transforms.append(kornia.augmentation.ColorJitter(
                brightness=cfg["color_jitter"]["brightness"], contrast=cfg["color_jitter"]["contrast"],
                saturation=cfg["color_jitter"]["saturation"], hue=cfg["color_jitter"]["hue"],
                p=cfg["color_jitter"]["p"]
            ))
        if cfg["gaussian_blur"]["enabled"]:
            transforms.append(kornia.augmentation.RandomGaussianBlur(
                kernel_size=(3, 3), sigma=cfg["gaussian_blur"]["sigma_limit"], p=cfg["gaussian_blur"]["p"]
            ))
        transforms.append(kornia.augmentation.Normalize(
            mean=torch.tensor(cfg["normalize"]["mean"]), std=torch.tensor(cfg["normalize"]["std"])
        ))
        return KorniaTransform(nn.Sequential(*transforms))
    
    def setup(self, stage: str = "train"):
        if stage == 'fit':
            self.dataset_train = SimCLRDataset(self.data_dir, transform=self.train_transform)
        if stage == "predict":
            predict_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=utils.IMAGENET_NORMALIZE["mean"], std=utils.IMAGENET_NORMALIZE["std"]
                ),
            ])
            self.dataset_predict = LightlyDataset(input_dir=self.data_dir, transform=predict_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True, pin_memory=True, collate_fn=collate_fn
        )
    
    def predict_dataloader(self):
        # Use smaller batch size for prediction to avoid OOM
        predict_batch_size = min(8, self.batch_size)  # Max 8 for prediction
        return torch.utils.data.DataLoader(
            self.dataset_predict, batch_size=predict_batch_size, shuffle=False,
            num_workers=min(4, self.num_workers), pin_memory=True, drop_last=False
        )

class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.T_max =  config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = config.get('optimizer').get('lr', lr)
        self.temp = config.get('criterion').get('params', {}).get('temperature', 0.07)  # Changed from 0.5
        model = config.get("backbone")
        weights = config.get('weights')
        if model == 'resnet50':
            base_model = torchvision.models.get_model(model, weights=weights)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            hidden_dim = base_model.fc.in_features
        else:
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            weights = ViT_L_16_Weights.IMAGENET1K_V1
            base_model = vit_l_16(weights=weights, image_size=224)
            base_model.heads = nn.Identity()
            self.backbone = base_model
            hidden_dim = 1024
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        self.criterion = SupConLoss(temperature=self.temp)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, labels = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # Debug prints
        # only on first batch
        if batch_idx == 0:
            print(f"Batch {batch_idx}: z0 shape={z0.shape}, z1 shape={z1.shape}")
            print(f"z0 min/max: {z0.min().item():.6f}/{z0.max().item():.6f}")
            print(f"z1 min/max: {z1.min().item():.6f}/{z1.max().item():.6f}")
            print(f"z0 has NaN: {torch.isnan(z0).any().item()}")
            print(f"z1 has NaN: {torch.isnan(z1).any().item()}")
            print(f"Labels: {labels[:5]}")  # First 5 labels
        
        # Reshape features for SupConLoss: [batch_size, n_views, feature_dim]
        features = torch.cat([z0.unsqueeze(1), z1.unsqueeze(1)], dim=1)
        loss = self.criterion(features, labels)

        if batch_idx == 0:
            print(f"Loss: {loss.item():.6f}")
            print(f"Loss has NaN: {torch.isnan(loss).item()}")
            print("=" * 50)
    
        
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max)
        return [optim], [scheduler]

def generate_embeddings(model, dataloader):
    embeddings = []
    filenames = []
    with torch.no_grad():
        for batch in dataloader:
            # Support both (img, filename) and (img, label, filename)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    img, fnames = batch
                elif len(batch) == 3:
                    img, _, fnames = batch
                else:
                    raise ValueError(f"Unexpected batch format in predict dataloader: len={len(batch)}")
            else:
                raise ValueError(f"Unexpected batch type from dataloader: {type(batch)}")

            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb.cpu())
            filenames.extend(fnames)
            torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames

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

def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    pl.seed_everything(config["seed"])
    data_dir = config["data"]["data_dir"]
    config_filename = os.path.splitext(os.path.basename(args.config))[0]
    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec='minutes')
    version = f"{config_filename}_{''.join(re.split(r'[-:]', date_str))}"
    logger = TensorBoardLogger(config["output"]["out_dir"], name=config["model"]["name"], version=version)
    
    dm = SimCLRDataModule(config)
    model = SimCLRModel(config["model"])
    trainer = pl.Trainer(**config["train"], logger=logger)
    trainer.fit(model, datamodule=dm)

    with open(os.path.join(logger.log_dir, "config.json"), 'w') as f:
        json.dump(config, f)

    print("Generating embeddings...")
    dm.setup(stage="predict")
    model.eval()
    embeddings, filenames = generate_embeddings(model, dm.predict_dataloader())
    
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f'x{i+1}' for i in range(len(df_embeddings.columns))]
    df_filenames = pd.DataFrame(filenames, columns=['filename'])
    df = pd.concat([df_filenames, df_embeddings], axis=1)
    output_filename = f"embeddings_{version}.csv"
    df.to_csv(os.path.join(logger.log_dir, output_filename), index=False, quoting=csv.QUOTE_ALL, encoding='utf-8', float_format='%.15f')
    
    plot_knn_examples(embeddings, filenames, data_dir)
    plt.savefig(os.path.join(logger.log_dir, "knn_examples.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to configuration file", default="./config.yaml")
    args = parser.parse_args()
    main(args)