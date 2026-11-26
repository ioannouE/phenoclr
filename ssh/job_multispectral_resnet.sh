#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=ResNetMultiSpectralClassifier300

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

python multispectral_classifier.py \
        --mode train \
        --data_path /shared/cooney_lab/Shared/RSE/Multispec_test_images_5000 \
        --epochs 300 \
        --batch_size 8