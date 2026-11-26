#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_RayTune_embeddings14_Kornia_Passerines_parallel_16Samples

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

python scripts/simclr_birdcolour_raytune_kornia.py --config config_raytune_kornia.yaml --num-parallel 1 --num-gpus 4 --num-samples 16 --album