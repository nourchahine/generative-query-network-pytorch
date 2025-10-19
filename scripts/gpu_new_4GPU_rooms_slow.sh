#!/bin/bash
#SBATCH --job-name=traingqnrooms
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/train-gqn-rooms-%j.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/train-gqn-rooms-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:48gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=168:00:00

# 1. SETUP
# -------------------------
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules and activate Conda
module --force purge
module load anaconda/3
conda activate ~/venvs/gqn_env
echo "Conda environment activated."

# --- Configuration ---
PROJECT_DIR="/home/mila/c/chahinen/gqn_project/generative-query-network-pytorch"
DATASET_NAME="rooms_free_camera_with_object_rotations"
# This is the directory with the CONVERTED data
SOURCE_DATA_DIR="/home/mila/c/chahinen/scratch/Data/gqndata_converted_rooms/rooms_free_camera_with_object_rotations"
LOG_DIR="/network/scratch/c/chahinen/gqn_logs/rooms_batch16"


# --- MODIFIED: Always check for a checkpoint ---
RESUME_ARGS=""
echo "Searching for the latest checkpoint to resume from..."

# Find the checkpoint file with the highest version number
LATEST_CHECKPOINT=$(ls -v "$LOG_DIR"/checkpoint_checkpoint_*.pt 2>/dev/null | tail -n 1)

# Check if a checkpoint file was actually found
if [ -f "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint to resume from: $LATEST_CHECKPOINT"
    # Set the argument for the python script
    RESUME_ARGS="--resume_from $LATEST_CHECKPOINT"
else
    # If no checkpoint is found, start from scratch
    echo "No checkpoint found. Starting from scratch."
fi


# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# 2. DATA STAGING - REMOVED
# -------------------------
# (The data staging/copying section has been removed)

# 3. EXECUTION
# -------------------------
echo "Starting training script..."
cd "$PROJECT_DIR" # Change to the project directory to run the script

# MODIFIED: Use the SOURCE_DATA_DIR directly
python run-gqn.py \
    --data_dir "$SOURCE_DATA_DIR" \
    --log_dir "$LOG_DIR" \
    --data_parallel "True" \
    --batch_size 1 \
    --workers 6 \
    --n_epochs 10 \
    $RESUME_ARGS

echo "Training script finished with exit code $?."

# 4. END OF JOB
# -------------------------
echo "Job finished at $(date)"