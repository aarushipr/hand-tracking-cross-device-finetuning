#!/bin/bash
#SBATCH --job-name="xr-handtrack-detection-phase2"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# DetNet PHASE 2, corrected. Continues from the phase-1 HOT3D checkpoint like
# detection_phanesim.sbatch, but at lr=1e-4 with early stopping and checkpoint
# selection driven by HOT3D validation. AD4_ARM picks the training set; see
# trainer_detection_phase2_arms.py's docstring for what each arm isolates.
#
# Submitting (three arms, safe to run concurrently -- separate checkpoint dirs):
#   sbatch --export=ALL,AD4_ARM=mixed    --job-name=xr-detection-p2-mixed    detection_phase2_arms.sbatch
#   sbatch --export=ALL,AD4_ARM=phanesim --job-name=xr-detection-p2-phanesim detection_phase2_arms.sbatch
#   sbatch --export=ALL,AD4_ARM=hot3d    --job-name=xr-detection-p2-hot3d    detection_phase2_arms.sbatch
# AD4_ARM defaults to "mixed" if not exported.
#
# Expected wall time, from phase-1/phase-2 checkpoint timestamps on this
# cluster (2026-09-09): HOT3D ~5.6 min/epoch, Phanesim ~1.2 min/epoch, so the
# mixed arm is ~7 min/epoch and its 30-epoch ceiling ~3.5h; the hot3d arm
# ~2.8h; the phanesim arm well under an hour. HOT3D validation adds to each.
# 12h leaves room for early-stopping not firing and for queue variance.
#
# The HOT3D index cache is warm at frame_stride=5 (294 stride5__v1 entries in
# local_config.hot3d_index_cache_dir, matching this loader's
# INDEX_FORMAT_VERSION=1), so no .vrs index rebuild is expected.
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
# 1. local_config.py has hot3d_dataset_root / hot3d_repo_root (every arm needs
#    them -- HOT3D validation drives selection in all three) and, for the mixed
#    and phanesim arms, phanesim_dataset_roots.
# 2. checkpoints_train_mixed/checkpoint_best.pth (phase 1) exists. Every arm
#    reads it as its starting weights and none writes to that directory.
# 3. checkpoints_phase2_<arm>/ does not exist yet for the arm being submitted.
# 4. AD4_LOADFAST=1 smoke test passed for that arm (writes to
#    checkpoints_loadfast_phase2_<arm>/).
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "===== Job info ====="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "AD4_ARM: ${AD4_ARM:-mixed (default)}"
echo "===== Starting DetNet phase-2 fine-tuning ====="

srun python trainer_detection_phase2_arms.py
