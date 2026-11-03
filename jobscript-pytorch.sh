#!/bin/env bash

#SBATCH -A NAISS2024-5-119  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 20:00:00  # hours:minutes:seconds
#SBATCH --gpus-per-node=A100:1
#SBATCH -J "continue to train Turbo Autoencoder R2 with batch size 500"

# Set-up environment
ml purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
# module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load wandb/0.13.4-GCCcore-11.3.0


# Interactive (but prefer Alvis OnDemand for interactive jupyter sessions)
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
# ipython -c "%run data-pytorch.ipynb"

# or you can instead use
#jupyter nbconvert --to python data-pytorch.ipynb &&
# python3 ./src/train/train_CNN_AutoEncoder_AWGN.py
# python3 ./src/train/train_Turbo_AutoEncoder_AWGN.py
python3 ./src/train/train_Turbo_AutoEncoder_AWGN_R2.py
# python3 ./src/inference/inference_CNN_AutoEncoder_AWGN.py

# python3 ./src/train/train_CNN_AutoEncoder_sync_equ.py
# python3 ./src/inference/inference_CNN_AutoEncoder_sync_equ.py


