#!/bin/env bash
#SBATCH -A NAISS2025-5-150
#SBATCH -p alvis
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH -N 1
#SBATCH -J "TurboAE, 1tap, raised cosine, add parity bit, p_array 9, seed 10"  # multi node, multi GPU

# Display basic information
echo "Hostname: $HOSTNAME"
echo "Node List: $SLURM_JOB_NODELIST"

# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
echo "MASTER_ADDR is set to $MASTER_ADDR"
echo "MASTER_PORT is set to $MASTER_PORT"

# Set up the environment
module purge
module load wandb/0.13.4-GCCcore-11.3.0
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0

# Run DistributedDataParallel with torchrun
srun -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -c "
torchrun \
   --node_rank=$SLURM_NODEID \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --nproc_per_node=$SLURM_GPUS_ON_NODE \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
   ./src/train/train_PHY_AE.py
"
