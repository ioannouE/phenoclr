# Perceptual evaluation of PhenoCLR embeddings

This directory evaluates how well distances between PhenoCLR embeddings agree with human perceptual similarity judgements. It uses the Berkeley-Adobe Perceptual Patch Similarity (BAPPS) dataset from the LPIPS paper (*"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"*). LPIPS is supervised on human judgements, so it serves as a reference. PhenoCLR is trained without any perceptual supervision.

The code for LPIPS and for downloading BAPPS is in the [PerceptualSimilarity repository](https://github.com/richzhang/PerceptualSimilarity):

```bash
git clone https://github.com/richzhang/PerceptualSimilarity
cd PerceptualSimilarity && bash scripts/download_dataset.sh   # creates dataset/2afc and dataset/jnd
```

Run the commands below from the repository root.

## Distances

- **LPIPS:** `d = LPIPS(x, y)`; lower means more similar.
- **PhenoCLR:** backbone features `f(x)`, `f(y)` are L2-normalised and compared with cosine distance `d = 1 − cos(f̂(x), f̂(y))`. Pass `--simclr-distance euclidean` for `d = ‖f̂(x) − f̂(y)‖`, and `--no-simclr-normalize` to skip normalisation.

## 2AFC

Each BAPPS 2AFC item is a triplet: a reference patch `x` and two distorted patches `x0` and `x1`, together with the human answer to "which is closer to `x`?". A metric scores a point when it picks the same patch as the humans. The reported score is the **2AFC accuracy**: the percentage of triplets on which the metric agrees with the human choice.

```bash
python perception_eval/eval_bapps_2afc.py \
    --bapps-root /path/to/PerceptualSimilarity/dataset/2afc \
    --eval-lpips --lpips-net alex \
    --eval-simclr --simclr-checkpoint /path/to/phenoclr.ckpt \
    --simclr-module train.simclr_birdcolour_RGB \
    --simclr-distance cosine
```

## JND

Each BAPPS JND item is a pair of patches (a reference and a distorted version) together with the human answer to "are the images the same?". Results are reported as ROC-AUC and as accuracy at the best threshold.

```bash
python perception_eval/eval_bapps_jnd.py \
    --bapps-root /path/to/PerceptualSimilarity/dataset/jnd \
    --eval-lpips --lpips-net alex \
    --eval-simclr --simclr-checkpoint /path/to/phenoclr.ckpt \
    --simclr-module train.simclr_birdcolour_RGB \
    --simclr-distance euclidean
```

## Arguments

| Argument | Description |
| --- | --- |
| `--bapps-root` | BAPPS `2afc` or `jnd` directory |
| `--subsets` | Subsets to evaluate. 2AFC default: all six `val/*` subsets. JND default: auto-discover. |
| `--eval-lpips`, `--lpips-net {alex,vgg,squeeze}`, `--lpips-size` | Evaluate the LPIPS baseline |
| `--eval-simclr` | Evaluate a PhenoCLR checkpoint |
| `--simclr-checkpoint` | Path to the PhenoCLR Lightning checkpoint (`.ckpt`) |
| `--simclr-module` | Training module that defines the checkpoint's `SimCLRModel`, e.g. `train.simclr_birdcolour_RGB` or `train.simclr_birdcolour_multispectral` (default) |
| `--simclr-config` | Optional config override. By default, the config embedded in the checkpoint is used. |
| `--simclr-input-size` | Resize resolution for the backbone input (default 224) |
| `--simclr-distance {cosine,euclidean}`, `--no-simclr-normalize` | Distance settings |
| `--device {cpu,cuda,mps}`, `--batch-size`, `--num-workers` | Runtime settings |
| JND only: `--positive-label`, `--judge-threshold`, `--jnd-{p0,p1,same}-dir`, `--jnd-same-exts` | JND label handling |

The `--simclr-*` flag names are kept for compatibility with existing scripts. They refer to the PhenoCLR model.

To compute bootstrapped confidence intervals for both evaluations, run [`eval/eval_full_bootstrap.py --run-perceptual`](../eval/eval_full_bootstrap.py).
