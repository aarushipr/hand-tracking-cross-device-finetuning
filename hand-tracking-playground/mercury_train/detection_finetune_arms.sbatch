#!/bin/bash
#SBATCH --job-name="xr-handtrack-detection-finetune"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# DetNet fine-tuning, selectable start point (AD4_INIT) and training set
# (AD4_ARM). See trainer_detection_finetune_arms.py's docstring for what each
# combination is and why AD4_INIT also selects the hyperparameters.
#
# THE RUN THE SUPERVISOR ASKED FOR (2026-09-09): Monado's shipped weights
# fine-tuned on Phanesim, evaluated on HOT3D alongside the untouched weights;
#   sbatch --export=ALL,AD4_INIT=default,AD4_ARM=phanesim \
#          --job-name=xr-detection-monado-phanesim detection_finetune_arms.sbatch
# AD4_INIT defaults to "default" and AD4_ARM to "phanesim", so a bare
# `sbatch detection_finetune_arms.sbatch` runs exactly that.
#
# Other combinations (separate checkpoint dirs, safe to run concurrently):
#   AD4_INIT=default,AD4_ARM=hot3d      reproduces phase 1
#   AD4_INIT=phase1,AD4_ARM=mixed       phase-2 rehearsal
#   AD4_INIT=phase1,AD4_ARM=hot3d       phase-2 HOT3D control
#
# Expected wall time, from checkpoint timestamps on this cluster (2026-09-09):
# HOT3D ~5.6 min/epoch, Phanesim ~1.2 min/epoch. With AD4_INIT=default the
# schedule is phase 1's 120 epochs / patience 8, so the phanesim arm is roughly
# 3h and the hot3d and mixed arms roughly 12-15h if early stopping never fires.
# HOT3D validation (33,680 samples) runs every epoch in every arm and adds under
# a minute per epoch at DetNet speeds. 24h covers all of it.
#
# The HOT3D index cache is warm at frame_stride=5 (294 stride5__v1 entries,
# matching this loader's INDEX_FORMAT_VERSION=1), so no .vrs rebuild is expected.
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/detection"

# ---------------------------------------------------------------------------
# Pre-flight:
# 1. local_config.py has hot3d_dataset_root / hot3d_repo_root (HOT3D validation
#    is measured in every configuration) and, for the phanesim and mixed arms,
#    phanesim_dataset_roots.
# 2. checkpoints_monado_<arm>/ (or checkpoints_phase2_<arm>/) does not exist yet
#    for the combination being submitted.
# 3. AD4_LOADFAST=1 smoke test passed for that combination.
# 4. AD4_INIT=phase1 additionally needs checkpoints_train_mixed/
#    checkpoint_best.pth. AD4_INIT=default does not read it at all.
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "===== Job info ====="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "AD4_INIT:   ${AD4_INIT:-default (default)}"
echo "AD4_ARM:    ${AD4_ARM:-phanesim (default)}"
echo "AD4_SELECT: ${AD4_SELECT:-per-init default}"
echo "===== Starting DetNet fine-tuning ====="

srun python trainer_detection_finetune_arms.py
