#!/bin/bash
#SBATCH --job-name="xr-handtrack-keynet-phanesim-phase2"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# KeyNet Training Job — PHASE 2: Phanesim fine-tuning on top of the phase-1
# HOT3D checkpoint. See kpest_trainer_phanesim.py's docstring for why this
# is a separate script/checkpoint dir from phase 1. Mirrors
# detection_phanesim.sbatch's own structure.
# ---------------------------------------------------------------------------
# Submitting: sbatch kpest_phanesim.sbatch
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/keypoint"

# ---------------------------------------------------------------------------
# Pre-flight checks (confirm before submitting this real job):
# 1. py/training/keypoint/local_config.py has phanesim_dataset_roots
#    pointing at both dataset roots under /storage/user/praa/phanesim_dataset/
#    (local_config.py is gitignored -- added by hand, see local_config_cluster.py).
# 2. checkpoints_train_mixed/checkpoint_best.pth (phase 1) exists here --
#    this job loads it as its starting weights; it does NOT retrain from
#    scratch or touch that file.
# 3. checkpoints_phanesim_phase2/ does not exist yet -- nothing for this
#    run to collide with.
# 4. AD4_LOADFAST=1 smoke test (checkpoints_loadfast_phanesim/) completed
#    successfully before this real job was ever submitted.
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "===== Job info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "===== Starting Phanesim phase-2 KeyNet fine-tuning ====="

srun python kpest_trainer_phanesim.py
