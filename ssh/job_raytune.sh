#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_RayTune_embeddings10_Kornia_Passerines-Segmented-Back-RGB-PNG_parallel_12Samples

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

python scripts/simclr_birdcolour_raytune.py --config config_raytune_album.yaml --num-parallel 2 --num-gpus 2 --num-samples 12 --album