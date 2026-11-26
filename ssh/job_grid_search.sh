#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64000
#SBATCH --time=99:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_GridSearch_Passerines-Segmented-Back-RGB-PNG

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

python scripts/simclr_birdcolour_grid.py --config config_grid.yaml