#!/bin/bash
#SBATCH --job-name=train-gqn
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/train-gqn-%j.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/train-gqn-%j.err
#SBATCH --partition=long          # Or any other partition with GPUs
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8         # Requesting more CPUs for the 6 data workers + main process
#SBATCH --mem=48G                 # Increased memory for training
#SBATCH --time=100:00:00           # Request 24 hours for training

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
# Use absolute paths for robustness
PROJECT_DIR="/home/mila/c/chahinen/gqn_project/generative-query-network-pytorch"
DATASET_NAME="shepard_metzler_5_parts"
# This is the directory with the CONVERTED data
SOURCE_DATA_DIR="/home/mila/c/chahinen/scratch/Data/gqndata_converted/tmp/$DATASET_NAME"
LOG_DIR="/network/scratch/c/chahinen/gqn_logs" # Persistent log directory for TensorBoard

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Path for data on the node's fast local disk
DATA_PATH_ON_NODE="$SLURM_TMPDIR/$DATASET_NAME"

# 2. DATA STAGING
# -------------------------
echo "Staging data from scratch to node-local disk..."
echo "Source: $SOURCE_DATA_DIR"
echo "Destination: $SLURM_TMPDIR"

cp -r "$SOURCE_DATA_DIR" "$SLURM_TMPDIR/"
echo "Data copy complete. Verifying contents:"
ls -lR "$DATA_PATH_ON_NODE" | head -n 10 # List first 10 lines of the copied data structure

# 3. EXECUTION
# -------------------------
echo "Starting training script..."
cd "$PROJECT_DIR" # Change to the project directory to run the script

python run-gqn.py \
    --data_dir "$DATA_PATH_ON_NODE" \
    --log_dir "$LOG_DIR" \
    --data_parallel "True" \
    --batch_size 1 \
    --workers 6

echo "Training script finished with exit code $?."

# 4. END OF JOB
# -------------------------
echo "Job finished at $(date)"