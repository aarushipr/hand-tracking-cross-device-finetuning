#!/bin/bash
#SBATCH --job-name="xr-handtrack-detection-phanesim-smoketest"
#SBATCH --partition=NORMAL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# Smoke test ONLY -- AD4_LOADFAST=1, 2 train clips + 1 val clip, capped at 2
# epochs, writes to checkpoints_loadfast/. Exists purely to prove
# trainer_detection_phanesim.py runs end to end (loads the phase-1
# checkpoint, trains, validates, saves a checkpoint) before the real
# multi-hour job in detection_phanesim.sbatch gets submitted.
# Submitting: sbatch detection_phanesim_smoketest.sbatch
# ---------------------------------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate handtrack

module load cuda/11.8.0
module load opencv/4.5.3-nonfree

export LD_LIBRARY_PATH="$HOME/lib_shim:$CONDA_PREFIX/lib:/storage/software/opencv/4.5.3-nonfree/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export AD4_LOADFAST=1

REPO_ROOT="/storage/user/praa/working_code_mercury_train/hand-tracking-playground/mercury_train"
SCRIPT_DIR="$REPO_ROOT/py/training/detection"

cd "$SCRIPT_DIR"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "===== Job info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Working dir: $(pwd)"
echo "===== Starting Phanesim phase-2 LOADFAST smoke test ====="

srun python trainer_detection_phanesim.py
