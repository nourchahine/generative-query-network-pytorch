#!/bin/bash
#!/bin/bash
#SBATCH --job-name=train-gqn
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/train-gqn-%j.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/train-gqn-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:48gb:4
#SBATCH --cpus-per-task=16        # MODIFIED: Request more CPUs for data loading
#SBATCH --mem=24G                 # MODIFIED: More memory is safer with a larger batch
#SBATCH --time=168:00:00       # Request 7 days for training

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
LOG_DIR="/network/scratch/c/chahinen/gqn_logs/oneper" # Persistent log directory for TensorBoard


# RESUME_ARGS=""
# # Check if the job has been restarted
# # Ensure there are spaces inside the brackets: [ SPACE ... SPACE ]
# if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
#     echo "Job is being restarted (Restart count: $SLURM_RESTART_COUNT). Finding latest checkpoint..."
    
#     LATEST_CHECKPOINT=$(ls -v "$LOG_DIR"/checkpoint_checkpoint_*.pt 2>/dev/null | tail -n 1)

#     if [ -f "$LATEST_CHECKPOINT" ]; then
#         echo "Found checkpoint to resume from: $LATEST_CHECKPOINT"
#         RESUME_ARGS="--resume_from $LATEST_CHECKPOINT"
#     else
#         echo "No checkpoint found. Starting from scratch."
#     fi
# fi

# In your sbatch script

RESUME_ARGS=""
# --- MODIFIED: Always check for a checkpoint ---
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

python run-gqn-new.py \
    --dataset_dir "$DATA_PATH_ON_NODE" \
    --log_dir "$LOG_DIR" \
    --workers 0\
    $RESUME_ARGS

echo "Training script finished with exit code $?."

# 4. END OF JOB
# -------------------------
echo "Job finished at $(date)"