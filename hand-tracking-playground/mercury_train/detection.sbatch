#!/bin/bash
#SBATCH --job-name="xr-handtrack-detection-workingcode"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# XR Hand Detection (DetNet) Training Job — working-code checkout
# No AD4_ENABLEGUI flag: detection has no visualizer/GUI path at all
# (per DETECTION_PIPELINE.md). wandb controlled via WANDB_API_KEY/WANDB_MODE.
# ---------------------------------------------------------------------------
# Submitting: sbatch detection.sbatch
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Set WANDB_API_KEY in ~/.bashrc, or run `wandb login` once interactively.
# export WANDB_MODE=disabled   # uncomment to disable wandb instead

# CHANGE ME: same placeholder as train.sbatch — set to wherever you actually
# clone/rsync working-code on the cluster.
REPO_ROOT="/home/stud/praa/storage/user/working_code_mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/detection"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
# 1. local_config.py — cp local_config_cluster.py over it, then fill in the
#    REPLACE_ME paths (HMDHandRects/EgoHands/EpicKitchens) — nobody has
#    confirmed these locations on the cluster yet, unlike the keypoint data.
# 2. HOT3D is not yet obtained — trainer_detection.py's test split is a
#    placeholder, this only trains + validates.
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "===== Job info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "===== Starting detection training ====="

srun python trainer_detection.py
