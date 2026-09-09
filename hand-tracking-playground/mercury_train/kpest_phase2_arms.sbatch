#!/bin/bash
#SBATCH --job-name="xr-handtrack-keynet-phase2"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# KeyNet PHASE 2, corrected. Mirrors detection_phase2_arms.sbatch; see
# kpest_trainer_phase2_arms.py's docstring for what each arm isolates and why
# this variant also switches AdamW's default weight_decay off.
#
# Submitting (three arms, safe to run concurrently -- separate checkpoint dirs):
#   sbatch --export=ALL,AD4_ARM=mixed    --job-name=xr-keynet-p2-mixed    kpest_phase2_arms.sbatch
#   sbatch --export=ALL,AD4_ARM=phanesim --job-name=xr-keynet-p2-phanesim kpest_phase2_arms.sbatch
#   sbatch --export=ALL,AD4_ARM=hot3d    --job-name=xr-keynet-p2-hot3d    kpest_phase2_arms.sbatch
# AD4_ARM defaults to "mixed" if not exported.
#
# Expected wall time, from phase-1/phase-2 checkpoint timestamps on this
# cluster (2026-09-09): KeyNet's HOT3D epochs measured ~50-60 min against
# Phanesim's ~3 min, so the mixed and hot3d arms are roughly an hour per epoch
# and their 20-epoch ceiling is around 20h; the phanesim arm is a couple of
# hours. HOT3D validation adds to each. 36h rather than 24h because phase 1's
# own KeyNet job showed one 19-hour, 10-epoch stretch -- node contention, not
# compute -- and these runs must not die to that before the deadline.
#
# The HOT3D keypoint index cache is warm at frame_stride=5 (294 stride5__v3
# entries, matching HOT3DKeypointDataset's INDEX_FORMAT_VERSION=3), and this
# trainer does not change the stride, so no .vrs index rebuild is expected.
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Required -- without it visualizer.py's periodic display_and_log_output()
# calls cv2.imshow and crashes on the headless cluster (job 1700705).
export AD4_ENABLEGUI=0

export AD4_ENABLEWANDB=1

REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/keypoint"

# ---------------------------------------------------------------------------
# Pre-flight:
# 1. py/training/keypoint/local_config.py has hot3d_dataset_root /
#    hot3d_repo_root / hot3d_object_library_path (every arm needs them -- HOT3D
#    validation drives selection in all three) and, for the mixed and phanesim
#    arms, phanesim_dataset_roots. local_config.py is gitignored; see
#    local_config_cluster.py.
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
echo "===== Starting KeyNet phase-2 fine-tuning ====="

srun python kpest_trainer_phase2_arms.py
