#!/bin/bash
#SBATCH --job-name="xr-handtrack-train-workingcode"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
# ↑ chair SLURM policy: no more than 4 CPU cores per GPU requested.
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# ↑ 8 hours is plenty for a sanity-scale dataset. Do NOT raise this past
#   8:00:00 without first confirming your account isn't subject to the
#   practical-student cap (max 1x 24GB GPU, 8h jobs, requires
#   --part=PRACT --qos=practical_course below). If you've confirmed you're
#   NOT a practical student, it's safe to raise this for full-dataset runs.
# If you ARE a practical student, uncomment the next line (mandatory):
# #SBATCH --part=PRACT
# #SBATCH --qos=practical_course
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# XR Hand Tracking (KeyNet) Training Job — working-code checkout
# Modeled on the root mercury_train repo's train.sbatch, repointed at
# working-code's own clone location on the cluster.
# ---------------------------------------------------------------------------
# Submitting: sbatch train.sbatch
# Monitor log: tail -f /storage/slurm/logs/slurm-<jobid>.out
# Cancel job:  scancel <jobid>
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Disable GUI - there is no display on the cluster. This is required.
export AD4_ENABLEGUI=0

# Set to 1 to enable Weights & Biases experiment tracking
export AD4_ENABLEWANDB=1

# Set to 1 for a quick sanity check run (only loads real sequence count,
# capped at 25 — working-code's ArtificialData.py was patched to not crash
# when fewer than 25 sequences exist on disk).
export AD4_LOADFAST=0

# Confirmed clone location (2026-07-20): includes the
# hand-tracking-playground/mercury_train subpath, unlike the root repo which
# was cloned directly as "mercury_train".
REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/keypoint"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
# 1. wandb login — run once interactively before submitting (or set
#    WANDB_API_KEY in ~/.bashrc).
# 2. local_config.py — copy local_config_cluster.py over it and adjust paths
#    if your data isn't at /storage/user/praa/synth_hands_output:
#      cp py/training/keypoint/local_config_cluster.py py/training/keypoint/local_config.py
# 3. C++ extension — must be rebuilt fresh on the cluster before first run:
#      rm -rf build/ && python setup.py install --user
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "===== Job info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "===== Starting training ====="

srun python kpest_trainer.py
