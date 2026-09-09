#!/bin/bash
#SBATCH --job-name="xr-handtrack-keynet-finetune"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# KeyNet fine-tuning, selectable start point (AD4_INIT) and training set
# (AD4_ARM). Mirrors detection_finetune_arms.sbatch; see
# kpest_trainer_finetune_arms.py's docstring for the reasoning, including why
# AD4_INIT=default deliberately keeps AdamW's default weight_decay.
#
# THE RUN THE SUPERVISOR ASKED FOR (2026-09-09):
#   sbatch --export=ALL,AD4_INIT=default,AD4_ARM=phanesim \
#          --job-name=xr-keynet-monado-phanesim kpest_finetune_arms.sbatch
# AD4_INIT defaults to "default" and AD4_ARM to "phanesim", so a bare
# `sbatch kpest_finetune_arms.sbatch` runs exactly that.
#
# Expected wall time, from checkpoint timestamps on this cluster (2026-09-09):
# KeyNet HOT3D training epochs ~50-60 min, Phanesim epochs ~3 min.
#
# CAUTION on the phanesim arm: HOT3D validation runs every epoch in every
# configuration, and at KeyNet's ~0.15 s/sample .vrs read cost the 23-sequence
# HOT3D val split is likely to dominate a Phanesim training epoch rather than be
# a rounding error on it. The first epochs of the smoke test / real run will say
# by how much. If HOT3D validation turns out to cost more than the training
# epoch itself, the fix is to validate HOT3D every Nth epoch rather than to drop
# it -- ask before changing it, since the per-epoch HOT3D curve is what the
# thesis reports.
#
# The HOT3D keypoint index cache is warm at frame_stride=5 (294 stride5__v3
# entries, matching HOT3DKeypointDataset's INDEX_FORMAT_VERSION=3), and this
# trainer does not change the stride, so no .vrs rebuild is expected.
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
#    hot3d_repo_root / hot3d_object_library_path (HOT3D validation is measured
#    in every configuration) and, for the phanesim and mixed arms,
#    phanesim_dataset_roots. local_config.py is gitignored; see
#    local_config_cluster.py.
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
echo "===== Starting KeyNet fine-tuning ====="

srun python kpest_trainer_finetune_arms.py
