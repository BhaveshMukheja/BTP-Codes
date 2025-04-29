#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- Environment Setup ---
module purge
module load compiler/cuda/11.7

# GPU Optimization Flags
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TF_ENABLE_ONEDNN_OPTS=0

# --- Conda Setup ---
source ~/miniconda3/bin/activate
conda activate yolo11

# --- Verification ---
echo "=== ENVIRONMENT CHECK ==="
nvidia-smi
python -c "
import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
from ultralytics import YOLO; print('Ultralytics import success')
"
echo "========================="

# --- Training ---
cd /scratch/21ph10008/
yolo train \
  model=/scratch/21ph10008/yolo11n.pt \
  data=/scratch/21ph10008/config.yaml \
  epochs=400 \
  batch=16 \
  imgsz=640 \
  device=0 \
  workers=8 \
  project=/scratch/21ph10008/runs \
  name=train_btp_model
