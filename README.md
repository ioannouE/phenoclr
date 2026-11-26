# PhenoCLR

**PhenoCLR: A Contrastive Representation Learning Framework for Organismal Colouration Across Imaging Modalities**

Eleftherios Ioannou, Yichen He, Kathryn Harris, Tamora D. James, Steve Maddock, Julien P. Renoult, Christopher R. Cooney

---

PhenoCLR is a self-supervised framework for learning representations of organismal colouration from images. Representation-learning tools for biological images are mostly limited to RGB and tuned for prediction. PhenoCLR extends contrastive learning (SimCLR) to **multispectral** and **hyperspectral** inputs, so that ultraviolet signals invisible to humans become part of the learned phenotype space.

The framework provides:

- **Transformer backbones:** ViT (torchvision `vit_l_16`) and **DINOv3** (`dinov3_vits16` … `dinov3_vit7b16`), plus ResNet for comparison.
- **Spectral adapters** that map an arbitrary number of spectral channels into the backbone. Options are a learned 1×1-convolution adapter (`MultispectralBackbone`) or an expanded ViT patch embedding.
- **Cross-modal RGB–UV contrastive learning:** RGB and UV views of the same specimen form positive pairs, with optional modality-specific encoders.
- **Attention-based fusion** of RGB and UV features (`ModalityFusion`), with `concat` and `separate` alternatives.
- **Evaluation tools** that compare learned spaces against human perception (BAPPS 2AFC/JND, LPIPS baseline) and against independent biological ground truth (species and sex structure, sexual dichromatism, cross-modal agreement).

PhenoCLR is trained without labels on museum bird specimens (6-channel RGB + UV multispectral images) and applied to hyperspectral butterfly scans (408 bands).

## Manuscript analyses and their code

| Analysis | Code |
| --- | --- |
| Agreement with human perceptual similarity judgements, compared with LPIPS | [`perception_eval/`](perception_eval/) (`eval_bapps_2afc.py`, `eval_bapps_jnd.py`), bootstrapped CIs in [`eval/eval_full_bootstrap.py`](eval/eval_full_bootstrap.py) |
| Consistency of representations across spectral domains (RGB, UV, RGB + UV) | [`eval/mantel_embedding_correlation.py`](eval/mantel_embedding_correlation.py), [`visualisations/paired_retrieval_test.py`](visualisations/paired_retrieval_test.py), [`visualisations/plot_paired_knn_cross_modality.py`](visualisations/plot_paired_knn_cross_modality.py) |
| Male–female separation per species (sexual dichromatism), without sex labels during training | [`eval/species_sexual_dichromatism.py`](eval/species_sexual_dichromatism.py) |
| Effect of colour-invariance augmentations on species and sex structure | `color_jitter` / `grayscale` switches in [`configs/`](configs/), evaluated with [`eval/eval_silhouette_variance_bootstrap.py`](eval/eval_silhouette_variance_bootstrap.py) |
| Hyperspectral butterflies | [`train/simclr_birdcolour_hyperspectral.py`](train/simclr_birdcolour_hyperspectral.py), [`lepidoptera/`](lepidoptera/), [`visualisations/plot_embeddings_hyperspectral.py`](visualisations/plot_embeddings_hyperspectral.py) |

## Installation

PhenoCLR needs Python ≥ 3.10. Install PyTorch for your platform and CUDA version first ([instructions](https://pytorch.org/get-started/locally/)), then install the remaining dependencies:

```bash
git clone https://github.com/ioannouE/phenoclr.git
cd phenoclr

conda create -n phenoclr python=3.10
conda activate phenoclr
pip install -r requirements.txt
```

### DINOv3 weights

DINOv3 backbones are loaded via `torch.hub` from [`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3). The pretrained weights must be downloaded separately after accepting the DINOv3 licence. Point the config at the downloaded weights:

```yaml
model:
  backbone: 'dinov3'
  dinov3:
    repo: "facebookresearch/dinov3"
    arch: "dinov3_vitl16"          # dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16 | dinov3_vit7b16
    weights_path: "path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    source: "auto"                 # auto | github | local
```

All commands below are run from the repository root.

## Data

### Multispectral bird specimens (RGB + UV)

Each specimen is a 6-channel GeoTIFF (`.tif`/`.tiff`). Channels 0–2 are RGB and channels 3–5 are UV. Images are discovered recursively under `data.data_dir` and read by [`data/multispectral_data.py`](data/multispectral_data.py).

Filenames carry the metadata that the evaluation scripts use. They are **not** used during training:

```
Genus_species_<id>_<SEX>_<view>.tif      e.g. Anthochaera_chrysoptera_1_M_Belly.tif
```

`SEX` is `M` or `F`. Specimens without a parseable sex are excluded from sex-based analyses.

Utilities:

- [`data/inspect_multispectral_tif.py`](data/inspect_multispectral_tif.py) prints the band layout and pixel statistics of `.tif` files.
- [`data/convert_multispectral_to_rgb_png.py`](data/convert_multispectral_to_rgb_png.py) converts the multispectral database to RGB PNGs, for RGB-only training and thumbnails in plots.
- [`visualisations/visualize_multispectral_specimen.py`](visualisations/visualize_multispectral_specimen.py) and [`visualisations/visualize_spectral_bands.py`](visualisations/visualize_spectral_bands.py) visualise individual specimens and bands.

### Hyperspectral butterflies

The hyperspectral Lepidoptera scans come from [*A computational neuroscience framework for quantifying warning signals*](https://arts.st-andrews.ac.uk/lepidoptera/index.html). They are ENVI `.bil` files with `.bil.hdr` headers, 408 bands each. To download them:

```bash
python download_spectral_data.py --output-dir data/lepidoptera
```

The files are read by [`lepidoptera/read_iml_hyp.py`](lepidoptera/read_iml_hyp.py), a Python re-implementation of the original MATLAB reader ([`lepidoptera/readIMLhyp.m`](lepidoptera/readIMLhyp.m)), and wrapped by [`data/hyperspectral_data.py`](data/hyperspectral_data.py). With `--rgb-only`, the bands closest to 450, 550 and 650 nm form a pseudo-RGB image. With `--sample-bands`, a subset of bands is used.

## Training

Every training script reads a YAML config (`-c/--config`) and writes TensorBoard logs, checkpoints, an embeddings CSV and k-NN example plots to `output.out_dir`. Add `--visualize` to preview the augmentation pipeline on random samples, and `--wandb` to log to Weights & Biases (project `phenoclr`).

| Modality | Script | Example config |
| --- | --- | --- |
| RGB (ResNet / ViT / DINOv3) | [`train/simclr_birdcolour_RGB.py`](train/simclr_birdcolour_RGB.py) | [`configs/config_rgb_dino_stanage.yaml`](configs/config_rgb_dino_stanage.yaml), [`configs/config_rgb_vit_stanage.yaml`](configs/config_rgb_vit_stanage.yaml) |
| Multispectral RGB + UV, incl. cross-modal | [`train/simclr_birdcolour_multispectral.py`](train/simclr_birdcolour_multispectral.py) | [`configs/config_rgbuv_vit_stanage.yaml`](configs/config_rgbuv_vit_stanage.yaml), [`configs/config_kornia_spectral.yaml`](configs/config_kornia_spectral.yaml) |
| Hyperspectral | [`train/simclr_birdcolour_hyperspectral.py`](train/simclr_birdcolour_hyperspectral.py) | [`configs/config_kornia_hyperspectral.yaml`](configs/config_kornia_hyperspectral.yaml) |

The configs in `configs/` record the settings used for the manuscript. Their `data_dir`, `out_dir` and `weights_path` values are placeholders (`path/to/...`); replace them with your own locations.

### RGB

```bash
python train/simclr_birdcolour_RGB.py -c configs/config_rgb_dino_stanage.yaml
```

`model.backbone` selects `resnet`, `vit` or `dino` (DINOv3).

### Multispectral (RGB + UV)

```bash
# all 6 channels through a single encoder with a spectral adapter
python train/simclr_birdcolour_multispectral.py -c configs/config_kornia_spectral.yaml

# single-domain baselines
python train/simclr_birdcolour_multispectral.py -c configs/config_kornia_spectral.yaml --rgb-only
python train/simclr_birdcolour_multispectral.py -c configs/config_kornia_spectral.yaml --uv-only
```

The script supports the following options; each command-line flag overrides the corresponding YAML value:

| Flag | Description |
| --- | --- |
| `--rgb-only` / `--uv-only` | Use only channels 0–2 (RGB) or 3–5 (UV) |
| `--channels 0 2 4` | Use an arbitrary subset of channels |
| `--usml` | Convert to the avian tetrahedral colour space (USML + DL) |
| `--loss-type {global,local,combined,crossmodal}` | Contrastive objective (see below) |
| `--global-weight`, `--local-weight`, `--cross-modal-weight` | Loss weights |
| `--use-modality-specific` | Separate encoders for RGB and UV |
| `--fusion-type {concat,separate,attention}` | How RGB and UV features are combined |
| `--augmentation-strategy {crop,resize_and_pad}` | Crop to, or letterbox into, the input size |
| `--model-checkpoint` | Initialise the backbone from a checkpoint |

Loss types:

- `global`: standard NT-Xent loss between two augmented views of the whole image.
- `local`: NT-Xent loss applied to corresponding local patches.
- `combined`: weighted sum of `global` and `local`.
- `crossmodal`: the RGB and UV views of the same specimen are a positive pair.

#### Cross-modal learning with attention-based fusion

```bash
python train/simclr_birdcolour_multispectral.py -c configs/config_rgbuv_vit_stanage.yaml \
    --loss-type crossmodal --use-modality-specific --fusion-type attention
```

With `--use-modality-specific`, separate RGB and UV encoders (ResNet, ViT or DINOv3) are trained. Their features are combined by `ModalityFusion`:

- `attention`: cross-attention in which RGB features query UV features.
- `concat`: concatenation followed by a linear projection.
- `separate`: no fusion; both embeddings are kept.

### Hyperspectral

```bash
python train/simclr_birdcolour_hyperspectral.py -c configs/config_kornia_hyperspectral.yaml
python train/simclr_birdcolour_hyperspectral.py -c configs/config_kornia_hyperspectral.yaml --sample-bands 0,100,200
python train/simclr_birdcolour_hyperspectral.py -c configs/config_kornia_hyperspectral.yaml --rgb-only
```

All selected bands pass through a single encoder with a spectral adapter. `--test-embeddings-only` skips training and only exports embeddings.

### Configuration

| Key | Description |
| --- | --- |
| `data.data_dir` | Image directory (searched recursively) |
| `output.out_dir` | Output and logging directory |
| `model.backbone` | `resnet`, `vit_l_16` / `vit`, or `dinov3` / `dino` |
| `model.weights` | torchvision pretrained weights (e.g. `ViT_L_16_Weights`) or `null` |
| `model.dinov3.*` | DINOv3 hub repo, architecture and weights path (see above) |
| `model.optimizer.lr`, `model.scheduler.params.{T_max,eta_min}` | Optimiser and cosine schedule |
| `model.criterion.type` | `global`, `local`, `combined` or `crossmodal` |
| `model.criterion.params.temperature` | NT-Xent temperature |
| `dataloader.{batch_size,num_workers}` | Data loading |
| `train.{max_epochs,accelerator,devices,deterministic}` | PyTorch Lightning trainer settings |
| `seed` | Random seed |
| `augmentations.input_size` | Backbone input resolution |
| `augmentations.transforms.*` | Augmentation pipeline (below) |

### Augmentations and colour invariance

Augmentations are implemented with [Kornia](https://kornia.readthedocs.io/) on the GPU and apply jointly to every spectral channel. Each transform is switched on and off with an `enabled` field:

`random_resize_crop`, `color_jitter`, `grayscale`, `gaussian_blur`, `horizontal_flip`, `vertical_flip`, `rotate`, `gaussian_noise`, `thin_plate_spline`, `random_perspective`, `random_sharpness`, `random_posterize`, `random_erasing`.

SimCLR's default augmentations make representations invariant to colour through `color_jitter` and `grayscale`. Because colour is the phenotype of interest here, the manuscript's augmentation analysis compares otherwise-identical runs with these two transforms enabled and disabled:

```yaml
augmentations:
  transforms:
    color_jitter:
      enabled: false
    grayscale:
      enabled: false
```

The resulting embeddings are scored with the species–sex silhouette analysis described under [Evaluation](#evaluation). To preview a configuration's augmentations, run [`visualisations/visualize_augmentations_from_config.py`](visualisations/visualize_augmentations_from_config.py).

## Generating embeddings

The training scripts export embeddings automatically. Each export is a CSV with a `filename` column followed by `x1 … xN`. To regenerate embeddings from a saved checkpoint:

```bash
python embeddings/generate_embeddings.py      --model_path <ckpt> --config <config.yaml> --data_dir <images> --output_dir <out>
python embeddings/generate_embeddings_dino.py --model_path <ckpt> --config <config.yaml> --data_dir <images> --output_dir <out>

# intermediate ViT layers (e.g. concatenate the last 4 blocks, mean-pooled)
python embeddings/generate_embeddings_vit_layers.py --model_path <ckpt> --config <config.yaml> \
    --data_dir <images> --output_dir <out> --vit_last_k 4 --vit_pool mean --vit_aggregate concat
```

To extract the trained backbone for use outside PhenoCLR, run [`model_utils/extract_vit_from_simclr.py`](model_utils/extract_vit_from_simclr.py) or [`model_utils/extract_dinov3_from_simclr.py`](model_utils/extract_dinov3_from_simclr.py).

## Evaluation

### Human perceptual similarity (BAPPS)

PhenoCLR embeddings are compared with LPIPS on the Berkeley-Adobe Perceptual Patch Similarity dataset, using both 2AFC and JND. Full instructions are in [`perception_eval/README.md`](perception_eval/README.md).

```bash
python perception_eval/eval_bapps_2afc.py --bapps-root <PerceptualSimilarity/dataset/2afc> \
    --eval-lpips --eval-simclr --simclr-checkpoint <ckpt> --simclr-module train.simclr_birdcolour_RGB
```

To compute bootstrapped confidence intervals for the perceptual and embedding-space metrics together:

```bash
python eval/eval_full_bootstrap.py --embeddings <embeddings.csv> --output <out> \
    --run-perceptual --eval-lpips --eval-simclr --simclr-checkpoint <ckpt> \
    --bapps-2afc-root <.../2afc> --bapps-jnd-root <.../jnd>
```

### Species and sex structure

```bash
python eval/eval_silhouette_variance_bootstrap.py --embeddings <embeddings.csv> --output <out> --n-bootstrap 1000
```

This reports the silhouette score of species × sex classes, and the variance explained by species, sex and species × sex, with bootstrap CIs. It uses subsampling without replacement by default (`--subsample-frac 0.8`). [`eval/eval_silhouette_variance.py`](eval/eval_silhouette_variance.py) is the same analysis without bootstrapping.

### Sexual dichromatism

```bash
python eval/species_sexual_dichromatism.py --embeddings <embeddings.csv> --top_n 20 \
    --rank_by centroid_distance --output_dir <out> --plot --image_dir <rgb_pngs>
```

This ranks species by male–female separation in embedding space (centroid distance, mean pairwise distance or per-species silhouette), without using sex labels during training. If `--image_dir` is given, it also plots each top-ranked species with image thumbnails on the shared embedding projection.

### Cross-domain consistency

```bash
python eval/mantel_embedding_correlation.py \
    --embeddings rgb.csv uv.csv rgbuv.csv --labels RGB UV RGB+UV --permutations 999
```

This computes pairwise Mantel correlations between the distance matrices of embedding spaces from different modalities or models. The paired retrieval and cross-modality k-NN scripts in `visualisations/` provide complementary checks:

- [`paired_retrieval_test.py`](visualisations/paired_retrieval_test.py)
- [`plot_paired_knn_cross_modality.py`](visualisations/plot_paired_knn_cross_modality.py)
- [`plot_knn_multispectral.py`](visualisations/plot_knn_multispectral.py)

### Other analyses

| Script | Purpose |
| --- | --- |
| [`eval/umap_plot.py`](eval/umap_plot.py), [`plots/umap_plot.py`](plots/umap_plot.py) | UMAP projections of embeddings |
| [`eval/pagerank.py`](eval/pagerank.py), [`plots/pagerank.py`](plots/pagerank.py) | k-NN graph PageRank: most and least "central" specimens |
| [`eval/diffusion_maps.py`](eval/diffusion_maps.py) | Diffusion-map embedding of the learned space |
| [`plots/cv_metrics_diss.py`](plots/cv_metrics_diss.py) | LPIPS / SSIM / PSNR dissimilarity matrices for comparison |
| [`visualisations/plot_pca_pattern_space.py`](visualisations/plot_pca_pattern_space.py) | PCA "pattern space" plots with image thumbnails |
| [`visualisations/plot_tsne_highlight.py`](visualisations/plot_tsne_highlight.py) | t-SNE / UMAP with selected taxa highlighted |
| [`visualisations/plot_embeddings_*.py`](visualisations/) | Embedding plots for RGB, multispectral and hyperspectral data |
| [`visualisations/compute_wasserstein_distance.py`](visualisations/compute_wasserstein_distance.py) | Distribution distances between groups in embedding space |
| [`lepidoptera/eval_channel_variance.py`](lepidoptera/eval_channel_variance.py) | Per-band variance of the hyperspectral data (for band selection) |

## Repository structure

```
configs/          YAML configurations for the experiments
data/             Multispectral and hyperspectral datasets / data modules and conversion utilities
train/            PhenoCLR training scripts (RGB, multispectral, cross-modal, hyperspectral)
  hyperparameter_tuning/   Ray Tune and grid-search scripts (development)
embeddings/       Embedding export from trained checkpoints
model_utils/      Backbone extraction from Lightning checkpoints
eval/             Embedding-space evaluation (silhouette, variance, Mantel, dichromatism, bootstrap)
perception_eval/  Human perceptual similarity evaluation (BAPPS 2AFC / JND vs. LPIPS)
lepidoptera/      Hyperspectral reader (Python + original MATLAB) and butterfly utilities
visualisations/   Embedding, retrieval and augmentation visualisations
plots/            Additional UMAP / PageRank / image-metric plots
```

Earlier iterations of the training code are kept in `train/` for reference:

- `simclr_birdcolour_kornia*.py`
- `simclr_birdcolour_rse.py`
- `simclr_birdcolour_supcon.py`

The scripts listed under [Training](#training) supersede them. The hyperparameter-tuning scripts also depend on these earlier versions.

## Citation

If you use PhenoCLR, please cite:

```bibtex
@article{ioannou2026phenoclr,
  title   = {PhenoCLR: A Contrastive Representation Learning Framework for Organismal Colouration Across Imaging Modalities},
  author  = {Ioannou, Eleftherios and He, Yichen and Harris, Kathryn and James, Tamora D. and Maddock, Steve and Renoult, Julien P. and Cooney, Christopher R.},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

## Acknowledgements

PhenoCLR builds on the following projects:

- The SimCLR implementation in [Lightly SSL](https://github.com/lightly-ai/lightly)
- The [RSE-bird-colour](https://github.com/christophercooney/RSE-bird-colour) repository
- [Kornia](https://github.com/kornia/kornia) for augmentations
- [DINOv3](https://github.com/facebookresearch/dinov3) backbones
- The [LPIPS / BAPPS](https://github.com/richzhang/PerceptualSimilarity) benchmark

The hyperspectral reader re-implements the MATLAB code from the [Lepidoptera hyperspectral project](https://arts.st-andrews.ac.uk/lepidoptera/index.html).
