#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_Passerines_Kornia_Segmented-test

module load Anaconda3/2022.05
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1

python train/simclr_birdcolour_kornia02.py --config configs/config_kornia02_stanage.yaml

# python train/simclr_birdcolour_kornia02_multidata.py --config configs/config_kornia02_multidata.yaml