#!/bin/bash
#SBATCH --job-name="xr-handtrack-eval-phanesim-checkpoints"
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# Diagnostic sweep, NOT a training job: scores every intermediate checkpoint
# already saved from the phanesim phase-2 runs (checkpoint_0.pth,
# checkpoint_10.pth, ..., checkpoint_best.pth) against the real HOT3D
# test_mixed split, using the SAME eval_detnet.py / eval_keynet.py scripts
# unchanged. Purpose: phase-2's "best" checkpoint was chosen by validation
# loss measured on phanesim's OWN val clips, never against HOT3D -- this
# sweep shows whether HOT3D performance degrades gradually over phase-2
# training (in which case an earlier stopping point loses less) or is
# already gone by the first saved checkpoint (in which case the phanesim-
# only data source itself, not the exact stopping epoch, is the problem).
# Submitting: sbatch eval_phanesim_checkpoints.sbatch
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
cd "$REPO_ROOT"
mkdir -p results

echo "===== Job info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"

echo
echo "===== DetNet phanesim phase-2 checkpoints ====="
# Trimmed to checkpoint_20 only: job 1701397 was cancelled at the 4h time
# limit after scoring just checkpoint_0 (each checkpoint eval reloads the
# full HOT3D VRS dataset from scratch, which is slow). We already have
# checkpoint_0 and checkpoint_best from earlier runs, so checkpoint_20 is
# the one new midpoint needed to see whether forgetting is gradual or
# already complete by epoch 0.
for ckpt in py/training/detection/checkpoints_phanesim_phase2/checkpoint_20.pth; do
    [ -e "$ckpt" ] || continue
    name=$(basename "$ckpt" .pth)
    echo "----- DetNet $name -----"
    python py/evaluation/eval_detnet.py \
        --weights "$ckpt" --split test_mixed \
        --out "results/detnet_phanesim_phase2_${name}_mixed.json"
done

echo
echo "===== KeyNet phanesim phase-2 checkpoints ====="
# Trimmed to checkpoint_0 and checkpoint_10: we already have checkpoint_best
# from the earlier single-checkpoint eval, so these two give KeyNet's own
# epoch-0 and one midpoint, same reasoning as the DetNet trim above.
for ckpt in py/training/keypoint/checkpoints_phanesim_phase2/checkpoint_0.pth py/training/keypoint/checkpoints_phanesim_phase2/checkpoint_10.pth; do
    [ -e "$ckpt" ] || continue
    name=$(basename "$ckpt" .pth)
    echo "----- KeyNet $name -----"
    python py/evaluation/eval_keynet.py \
        --weights "$ckpt" --split test_mixed \
        --out "results/keynet_phanesim_phase2_${name}_mixed.json"
done

echo
echo "===== Sweep complete ====="
