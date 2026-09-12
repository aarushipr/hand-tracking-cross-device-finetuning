#!/bin/bash
#SBATCH --job-name="xr-eval-all-labeled"
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# Scores ONE model against the full labelled HOT3D catalogue ("all_labeled",
# 294 sequences, both devices) rather than the 59-sequence test_mixed split.
#
# Under the current design HOT3D is never trained on, so holding participants
# out of it buys nothing: every labelled sequence is equally unseen by every
# model scored here. Scoring all of them removes the question of whether the
# four held-out participants happened to be easy or hard.
#
# One model per job, selected by TARGET, so the evaluations run concurrently
# on separate GPUs instead of serially inside one job. A serial loop over all
# four does not fit in the time remaining. Submit with:
#   sbatch --export=ALL,TARGET=detnet_baseline eval_all_labeled.sbatch
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

DET_SYN="py/training/detection/checkpoints_monado_phanesim/checkpoint_best.pth"
DET_EP1="py/training/detection/checkpoints_monado_phanesim/checkpoint_best_hot3d.pth"
KEY_SYN="py/training/keypoint/checkpoints_monado_phanesim/checkpoint_best.pth"

case "$TARGET" in
  detnet_baseline)  NET=detnet; W="monado";    TAG=baseline ;;
  detnet_synthetic) NET=detnet; W="$DET_SYN";  TAG=synthetic ;;
  detnet_epoch1)    NET=detnet; W="$DET_EP1";  TAG=epoch1 ;;
  keynet_baseline)  NET=keynet; W="monado";    TAG=baseline ;;
  keynet_synthetic) NET=keynet; W="$KEY_SYN";  TAG=synthetic ;;
  # An unset or misspelled TARGET must not silently score some default model
  # and write a plausible-looking results file under the wrong name.
  *) echo "TARGET=${TARGET:-<unset>} is not a known target" >&2; exit 1 ;;
esac

# Fail before the dataset is loaded rather than after, since loading the HOT3D
# VRS catalogue takes long enough that a missing checkpoint would otherwise
# waste a scheduling slot before reporting itself.
if [ "$W" != "monado" ] && [ ! -e "$W" ]; then
    echo "checkpoint not found: $W" >&2
    exit 1
fi

echo "===== $TARGET : $NET $TAG on all_labeled ====="
echo "Job ID: $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
date

OUT="results/${NET}_${TAG}_all_labeled.json"

if [ "$NET" = "detnet" ]; then
    python py/evaluation/eval_detnet.py \
        --weights "$W" --split all_labeled \
        --out "$OUT" \
        --out-csv "results/${NET}_${TAG}_all_labeled.csv"
else
    python py/evaluation/eval_keynet.py \
        --weights "$W" --split all_labeled \
        --out "$OUT"
fi

date
echo "===== wrote $OUT ====="
