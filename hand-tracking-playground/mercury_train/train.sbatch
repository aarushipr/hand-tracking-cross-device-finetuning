#!/bin/bash
#SBATCH --job-name="xr-handtrack-train-workingcode"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
# ↑ chair SLURM policy: no more than 4 CPU cores per GPU requested.
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
# ↑ raised from 32G: with num_workers=4 each DataLoader worker opens its
#   own Hot3dDataProvider per sequence, so provider memory scales with the
#   worker count. Watch `seff <jobid>` after the first run and trim if the
#   headroom turns out to be large.
#SBATCH --time=4-00:00:00
# ↑ verified 2026-08-26, not guessed: partition NORMAL has
#   MaxTime=14-00:00:00 and DefaultTime=08:00:00, and QOS `stud` sets no
#   MaxWall (`scontrol show partition`, `sacct -j 1590166`). The old
#   08:00:00 here was simply the untouched partition default, and the
#   practical-student cap it warned about does not apply to this account.
#   72h is a ceiling, not an estimate: early stopping (patience 8) is
#   expected to end the run well before it.
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
# Each DataLoader worker keeps its own .vrs handles open for every sequence
# it touches; 4 workers over ~110 sequences runs to a few thousand file
# descriptors, comfortably above the usual 1024 default.
ulimit -n 8192
# Without these, PyTorch sizes its CPU thread pool to every core it can see
# on the node rather than the 4 actually allocated to this job.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Disable GUI - there is no display on the cluster. This is required.
export AD4_ENABLEGUI=0

# Set to 1 to enable Weights & Biases experiment tracking
export AD4_ENABLEWANDB=1

# Set to 1 for a quick smoke test: 2 training and 1 validation HOT3D
# sequence, writing to checkpoints_loadfast/ so it cannot contaminate a real
# run's checkpoints. (This used to refer to ArtificialData.py's 25-sequence
# cap; that data source is no longer used -- training is HOT3D-only.)
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
#    REQUIRED after 2026-09-02: the HOT3D dataset key was renamed
#    hot3d_dataset_path -> hot3d_dataset_root so both trainers use one name.
#    An old local_config.py will fail with AttributeError until re-copied.
# 4. Split pre-flight — confirm the participant assignment and per-device
#    train share before committing a run to it:
#      python py/training/common/hot3d_split.py <hot3d_dataset_root>
# 5. Quest keypoint check — the mixed split puts Quest recordings into
#    KeyNet's training set for the first time. Confirm they load:
#      python py/evaluation/check_quest_keypoints.py <a Quest sequence dir>
#      python py/evaluation/check_quest_keypoints.py <an Aria sequence dir>
#    Compare the two; Aria is the control.
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
