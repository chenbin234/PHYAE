#!/bin/env bash
#SBATCH -A C3SE2026-1-20
#SBATCH -p vera
#SBATCH -t 0-30:00:00
#SBATCH --gpus-per-node=H100:2
#SBATCH -N 1
#SBATCH -J "Vera PHY-AE, 5taps, STO0.0, parray6, seed12"  # multi node, multi GPU

# Display basic information
echo "Hostname: $HOSTNAME"
echo "Node List: $SLURM_JOB_NODELIST"

# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
echo "MASTER_ADDR is set to $MASTER_ADDR"
echo "MASTER_PORT is set to $MASTER_PORT"

# test
# Set up the environment
module purge
module load virtualenv/20.23.1-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# If you have a venv with all your Python deps (including wandb), activate it:
source /cephyr/users/chenbin/Vera/my_venv/bin/activate

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
