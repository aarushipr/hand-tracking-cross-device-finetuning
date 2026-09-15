#!/bin/bash
#SBATCH --job-name="xr-handtrack-detection-workingcode"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
# ↑ chair SLURM policy: no more than 4 CPU cores per GPU requested.
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
# ↑ raised from 32G to match train.sbatch: with num_workers=4 each DataLoader
#   worker opens its own AriaDataProvider per sequence, so provider memory
#   scales with the worker count. Watch `seff <jobid>` and trim if the
#   headroom turns out to be large.
#SBATCH --time=4-00:00:00
# ↑ raised from the partition's untouched 08:00:00 default, on the same
#   verified basis as train.sbatch: partition NORMAL has MaxTime=14-00:00:00
#   and QOS `stud` sets no MaxWall, and the practical-student cap does not
#   apply to this account. A ceiling, not an estimate; early stopping
#   (patience 8) is expected to end the run well before it.
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# XR Hand Detection (DetNet) Training Job: working-code checkout
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

# Each DataLoader worker keeps its own .vrs handles open for every sequence it
# touches; 4 workers over ~210 sequences runs to a few thousand file
# descriptors, comfortably above the usual 1024 default. Same reasoning as
# train.sbatch.
ulimit -n 8192
# Without these, PyTorch sizes its CPU thread pool to every core it can see on
# the node rather than the 4 actually allocated to this job.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Set to 1 for a quick smoke test: 2 training and 1 validation HOT3D sequence,
# writing to checkpoints_loadfast/ so it cannot contaminate a real run.
export AD4_LOADFAST=0

# Set WANDB_API_KEY in ~/.bashrc, or run `wandb login` once interactively.
# export WANDB_MODE=disabled   # uncomment to disable wandb instead

# Confirmed clone location (2026-07-20); matches train.sbatch.
REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/detection"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
# 1. local_config.py: cp local_config_cluster.py over it, then fill in the
#    REPLACE_ME paths (HMDHandRects/EgoHands/EpicKitchens); nobody has
#    confirmed these locations on the cluster yet, unlike the keypoint data.
# 2. Split pre-flight: same as train.sbatch:
#      python py/training/common/hot3d_split.py <hot3d_dataset_root>
#    trainer_detection.py trains on train_mixed and validates on a carve-out
#    from it; it holds no test set. Test scoring is eval_detnet.py's job.
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
