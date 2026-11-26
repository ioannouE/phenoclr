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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils


from lightning.pytorch.loggers import TensorBoardLogger
from scipy.spatial.distance import pdist, squareform
import torchvision.transforms.v2 as v2
from skbio.stats.distance import DistanceMatrix, permanova

class SimCLRDataModule(pl.LightningDataModule):

    def __init__(self,
                 config: dict,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 input_size: int = 224):
        super().__init__()
        self.batch_size = config.get("dataloader", {}).get("batch_size", batch_size)
        self.num_workers = config.get("dataloader", {}).get("num_workers", num_workers)
        self.data_dir = config["data"]["data_dir"]
        self.transforms = config["augmentations"]["transforms"]
        self.input_size = self.transforms.get("input_size", input_size)

    def setup(self, stage: str = "train"):

        if stage == "fit":
            # SimCLR transforms
            transform = SimCLRTransform(**self.transforms)
            self.dataset_train = LightlyDataset(input_dir=self.data_dir,
                                                transform=transform)
        if stage == "predict":
            # Transformation for embedding the dataset after training
            predict_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((self.input_size, self.input_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=utils.IMAGENET_NORMALIZE["mean"],
                        std=utils.IMAGENET_NORMALIZE["std"],
                    ),
                ]
            )
            self.dataset_predict = LightlyDataset(input_dir=self.data_dir,
                                                  transform=predict_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

# %%
# Create the SimCLR Model
# -----------------------
# Now we create the SimCLR model. We implement it as a PyTorch Lightning Module
# and use a model backbone from Torchvision. Lightly provides implementations
# of the SimCLR projection head and loss function in the `SimCLRProjectionHead`
# and `NTXentLoss` classes. We can simply import them and combine the building
# blocks in the module.

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.T_max =  config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = config.get('optimizer').get('lr', lr)
        self.temp = config.get('criterion').get('params', {}).get('temperature', 0.5)

        # create a backbone and remove the classification head
        model = config.get("backbone")
        weights = config.get('weights')
        base_model = torchvision.models.get_model(model, weights=weights)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        hidden_dim = base_model.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss(self.temp)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max)
        return [optim], [scheduler]



# %%
# Next we create a helper function to generate embeddings
# from our test images using the model we just trained.
# Note that only the backbone is needed to generate embeddings,
# the projection head is only required for the training.
# Make sure to put the model into eval mode for this part!


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings.cpu())
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

def perform_permanova(embeddings, filenames):
    """Performs a Permanova analysis given embeddings and filenames"""

    # Extract species names from filenames and create metadata DataFrame
    print("Extracting species names...")
    def extract_species(filename):
        # Get just the filename without path
        base_name = os.path.basename(filename)
        # Split by underscore and take first two parts as genus_species
        parts = base_name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # Always take first two parts as genus_species
        return base_name
    
    metadata = pd.DataFrame({
        'filename': filenames,
        'species': [extract_species(f) for f in filenames]
    })
    metadata.set_index('filename', inplace=True)
    
    # Print species counts before proceeding
    print("\nSpecies counts:")
    species_counts = metadata['species'].value_counts()
    print(species_counts)
    
    # Calculate distance matrix using cosine distance
    print("\nCalculating distance matrix...")
    normalized_embeddings = normalize(embeddings)
    distances = pdist(normalized_embeddings, metric='cosine')
    distance_matrix = squareform(distances)
    
    # Convert to skbio DistanceMatrix using filenames as IDs
    distance_matrix_skbio = DistanceMatrix(distance_matrix, ids=metadata.index)
    
    # Perform PERMANOVA
    print("Performing PERMANOVA...")
    # First with regular distances
    permanova_results = permanova(
        distance_matrix_skbio,
        grouping=metadata['species'],
        permutations=99
    )
    
    # Then with "euclidified" distances (square root transformation)
    print("Performing PERMANOVA with euclidified distances...")
    euclidified_distances = np.sqrt(distance_matrix)
    euclidified_matrix_skbio = DistanceMatrix(euclidified_distances, ids=metadata.index)
    permanova_results_euclidified = permanova(
        euclidified_matrix_skbio,
        grouping=metadata['species'],
        permutations=99
    )
    
    # Save results
    results = {
        'Regular PERMANOVA': {
            'test statistic': permanova_results['test statistic'].item(),
            'p-value': permanova_results['p-value'].item(),
            'number of permutations': permanova_results['number of permutations']
        },
        'Euclidified PERMANOVA': {
            'test statistic': permanova_results_euclidified['test statistic'].item(),
            'p-value': permanova_results_euclidified['p-value'].item(),
            'number of permutations': permanova_results_euclidified['number of permutations']
        }
    }
    
    return results







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

    # Set up dataloader and model for training
    dm = SimCLRDataModule(config)
    model = SimCLRModel(config["model"])
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

    # perform Permanova analysis given embeddings
    permanova_results = perform_permanova(embeddings, filenames)
    with open(os.path.join(logger.log_dir, "permanova_results.json"), 'w') as f:
        json.dump(permanova_results, f)

if __name__ == "__main__":
    args = get_args()
    main(args)
