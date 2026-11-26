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
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

import kornia
import torchvision.transforms.v2 as v2


class SimCLRDataset(torch.utils.data.Dataset):
    """Custom dataset that returns raw images for learnable augmentations"""
    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
        
        # Get all image files
        self.files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(root, f))
        
        self.files.sort()  # Ensure consistent ordering
    
    def __len__(self):
        return len(self.files)
    
    # def __getitem__(self, idx):
    #     img_path = self.files[idx]
    #     # Read image using cv2
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    #     # Convert to tensor
    #     img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            # Read image using cv2
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and scale to [0, 1]
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            # # Apply normalization
            # img = (img - self.mean.to(img.device)) / self.std.to(img.device)
            
            return img, img_path
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            raise RuntimeError(f"Failed to process image {img_path}: {str(e)}")
        
        return img, img_path


def collate_fn(batch):
    """Custom collate function to handle tensors"""
    images, filenames = zip(*batch)
    images = torch.stack(images)
    return images, filenames



class SimCLRDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config.get("dataloader", {}).get("batch_size", 32)
        self.num_workers = config.get("dataloader", {}).get("num_workers", 4)
        self.data_dir = config["data"]["data_dir"]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SimCLRDataset(self.data_dir)
        
        if stage == 'predict' or stage is None:
            self.predict_dataset = SimCLRDataset(self.data_dir)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )



class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5):
        super().__init__()
        self.save_hyperparameters()
        model_config = config["model"]

        self.T_max = model_config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = model_config.get('optimizer').get('lr', lr)
        self.temp = model_config.get('criterion').get('params', {}).get('temperature', 0.5)
        
       
        # Create backbone and projection head
        model = model_config.get("backbone")
        weights = model_config.get('weights')
        base_model = torchvision.models.get_model(model, weights=weights)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        hidden_dim = base_model.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        
        # Create learnable augmentation layers
        self.augmentation_layers = self._create_augmentation_layers(config["augmentations"]["transforms"])
        self.criterion = NTXentLoss(self.temp)
    

    def _create_augmentation_layers(self, cfg):
        """Create augmentation layers with learnable parameters"""
        transforms = []
        
        for aug_name, aug_config in cfg.items():
            if aug_config.get("enabled", False) and aug_config.get("learnable", False):
                if aug_name == "color_jitter":
                    transforms.append(kornia.augmentation.ColorJitter(
                        brightness=aug_config["brightness"],
                        contrast=aug_config["contrast"],
                        saturation=aug_config["saturation"],
                        hue=aug_config["hue"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "gaussian_blur":
                    transforms.append(kornia.augmentation.RandomGaussianBlur(
                        kernel_size=(3, 3),
                        sigma=aug_config["sigma_limit"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "horizontal_flip":
                    transforms.append(kornia.augmentation.RandomHorizontalFlip(p=aug_config["p"]))
                elif aug_name == "vertical_flip":
                    transforms.append(kornia.augmentation.RandomVerticalFlip(p=aug_config["p"]))
                elif aug_name == "rotate":
                    transforms.append(kornia.augmentation.RandomRotation(
                        degrees=aug_config["limit"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "random_resize_crop":
                    transforms.append(kornia.augmentation.RandomResizedCrop(
                        size=tuple(aug_config["size"]),
                        scale=tuple(aug_config["scale"]),
                        p=aug_config["p"]
                    ))
                elif aug_name == "grayscale":
                    transforms.append(kornia.augmentation.RandomGrayscale(p=aug_config["p"]))
                elif aug_name == "gaussian_noise":
                    transforms.append(kornia.augmentation.RandomGaussianNoise(
                        mean_range=aug_config["mean_range"],
                        std_range=aug_config["std_range"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "thin_plate_spline":
                    transforms.append(kornia.augmentation.RandomThinPlateSpline(
                        scale=aug_config["scale"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "random_sharpness":
                    transforms.append(kornia.augmentation.RandomSharpness(
                        sharpness=aug_config["sharpness"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "random_perspective":
                    transforms.append(kornia.augmentation.RandomPerspective(
                        distortion_scale=aug_config["distortion_scale"],
                        p=aug_config["p"]
                    ))
                elif aug_name == "random_posterize":
                    transforms.append(kornia.augmentation.RandomPosterize(
                        bits=aug_config["bits"],
                        p=aug_config["p"]
                    ))
                else:
                    raise ValueError(f"Unknown augmentation layer: {aug_name}")
        
        # Enable gradients for all transform parameters
        transform_seq = nn.Sequential(*transforms)
        for module in transform_seq.modules():
            if hasattr(module, 'requires_grad_'):
                module.requires_grad_(True)
        
        return transform_seq
    

    def forward(self, x):
        # Apply learnable augmentations twice
        x1 = self.augmentation_layers(x)
        x2 = self.augmentation_layers(x)
        
        # Get embeddings
        h1 = self.backbone(x1).flatten(start_dim=1)
        h2 = self.backbone(x2).flatten(start_dim=1)
        
        # Get projections
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        return z1, z2
    

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Now receiving raw images
        z1, z2 = self(x)  # Apply augmentations in forward pass
        
        # Calculate contrastive loss
        loss = self.criterion(z1, z2)
        
        # Add regularization for augmentation parameters
        aug_params = []
        for module in self.augmentation_layers.modules():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                aug_params.append(module.weight)
        
        if aug_params:
            reg_loss = sum(torch.norm(p) for p in aug_params) * 0.01  # L2 regularization
            loss = loss + reg_loss
        
        # Log augmentation parameters
        self._log_augmentation_params()
        
        self.log('train_loss', loss)
        return loss
    
    def _log_augmentation_params(self):
        """Log learnable augmentation parameters"""
        for name, module in self.augmentation_layers.named_modules():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                self.log(f'aug_param/{name}', module.weight.mean())
    
    def on_train_epoch_start(self):
        """Implement curriculum learning for augmentations"""
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        strength = min(1.0, epoch / (max_epochs * 0.3))  # Ramp up over first 30% of training
        for module in self.augmentation_layers.modules():
            if hasattr(module, 'p'):
                module.p = module.p * strength
    
    def configure_optimizers(self):
        # Include augmentation parameters in optimization
        params = list(self.backbone.parameters()) + \
                list(self.projection_head.parameters()) + \
                list(self.augmentation_layers.parameters())
        
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
        return [optimizer], [scheduler]



def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """
    embeddings = []
    filenames = []
    with torch.no_grad():
        for img,_, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

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


def visualize_augmentations(model, datamodule, num_samples=6, num_augmentations=6):
    """
    Visualize augmentations applied to random samples from the dataset.
    
    Args:
        model: SimCLRModel instance with learnable augmentations
        datamodule: Lightning datamodule
        num_samples: Number of samples to visualize
        num_augmentations: Number of augmentations per sample
    """
    # Set up the dataset
    datamodule.setup(stage="fit")
    dataset = datamodule.train_dataset
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Create subplot grid
    fig, axes = plt.subplots(num_samples, num_augmentations + 1, 
                            figsize=(2*num_augmentations + 2, 2*num_samples))
    
    # Put model in eval mode temporarily
    model.eval()
    
    # Store original probabilities and set to 1.0 for visualization
    original_ps = {}
    for name, module in model.augmentation_layers.named_modules():
        if hasattr(module, 'p'):
            original_ps[name] = module.p
            module.p = 0.5
    
    for i, idx in enumerate(indices):
        # Get original image
        img, _ = dataset[idx]
        
        # Show original
        orig_img = img.permute(1, 2, 0).numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Show augmentations
        with torch.no_grad():  # We don't need gradients for visualization
            for j in range(num_augmentations):
                # Apply augmentations (add batch dimension)
                aug_img = model.augmentation_layers(img.unsqueeze(0))
                
                # Convert from tensor to numpy and adjust dimensions
                aug_img = aug_img.squeeze(0).permute(1, 2, 0).numpy()
                
                # Clip values to valid range
                aug_img = np.clip(aug_img, 0, 1)
                
                axes[i, j+1].imshow(aug_img)
                axes[i, j+1].set_title(f'Aug {j+1}')
                axes[i, j+1].axis('off')
    
    # Restore original augmentation probabilities
    for name, module in model.augmentation_layers.named_modules():
        if hasattr(module, 'p') and name in original_ps:
            module.p = original_ps[name]
    
    # Put model back in training mode
    model.train()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('augmentation_examples', exist_ok=True)
    
    # Save the figure with timestamp
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
    date_str = dt.datetime.isoformat(dt.datetime.utcnow(), timespec='minutes')
    version = f"{config_filename}_{''.join(re.split(r'[-:]', date_str))}"

    # Set up logger
    logger = TensorBoardLogger(config["output"]["out_dir"],
                               name=config["model"]["name"],
                               version=version)

    # Log under the run's root artifact directory during run
    # with mlflow.start_run():
    #     mlflow.log_dict(config, 'config.yaml')

    # Set up dataloader
    dm = SimCLRDataModule(config)
    
    # Initialize model
    model = SimCLRModel(config)
    
    if args.visualize:
        print("Visualizing augmentations...")
        visualize_augmentations(model, dm)
        return


    trainer = pl.Trainer(**config["train"], logger=logger)
    trainer.fit(model, datamodule=dm)

    # Log configuration
    with open(os.path.join(logger.log_dir, "config.json"), 'w') as f:
        json.dump(config, f)

    # Set up dataloader and model for inference then generate embeddings
    dm.setup(stage="predict")
    model.eval()
    embeddings, filenames = generate_embeddings(model, dm.predict_dataloader())

    # Save embeddings to output dir
    # pd.DataFrame(embeddings, index=filenames).to_csv(os.path.join(logger.log_dir, "embeddings.csv"), quoting=csv.QUOTE_ALL, encoding='utf-8')
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
