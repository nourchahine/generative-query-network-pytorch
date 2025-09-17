#!/bin/bash
#!/bin/bash
#SBATCH --job-name=train-gqn-rooms
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/train-gqn-rooms-%j.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/train-gqn-rooms-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:48gb:2
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
DATASET_NAME="rooms_free_camera_with_object_rotations"
# This is the directory with the CONVERTED data
SOURCE_DATA_DIR="/home/mila/c/chahinen/scratch/Data/gqndata_converted_rooms/rooms_free_camera_with_object_rotations"
LOG_DIR="/network/scratch/c/chahinen/gqn_logs/rooms" # Persistent log directory for TensorBoard


RESUME_ARGS=""
# Check if the job has been restarted
# Ensure there are spaces inside the brackets: [ SPACE ... SPACE ]
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "Job is being restarted (Restart count: $SLURM_RESTART_COUNT). Finding latest checkpoint..."
    
    LATEST_CHECKPOINT=$(ls -v "$LOG_DIR"/checkpoint_checkpoint_*.pt 2>/dev/null | tail -n 1)

    if [ -f "$LATEST_CHECKPOINT" ]; then
        echo "Found checkpoint to resume from: $LATEST_CHECKPOINT"
        RESUME_ARGS="--resume_from $LATEST_CHECKPOINT"
    else
        echo "No checkpoint found. Starting from scratch."
    fi
fi

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# ...
# Path for data on the node's fast local disk
# NOTE: The python script will be passed the PARENT directory
DATA_PATH_ON_NODE="$SLURM_TMPDIR/$DATASET_NAME"

# The path the ShepardMetzler class ACTUALLY tries to open
EXPECTED_PATH_ON_NODE="$DATA_PATH_ON_NODE/train"

# ...
echo "Staging a SMALL FRACTION of data into a 'train' subdirectory..."

# First, create the destination directory including the 'train' subdirectory
mkdir -p "$EXPECTED_PATH_ON_NODE"

# Find all files... and copy them INTO the 'train' subdirectory
find "$SOURCE_DATA_DIR" -type f | shuf | head -n 20 | xargs -I {} cp {} "$EXPECTED_PATH_ON_NODE/"

echo "Small data fraction copy complete. Verifying contents of 'train' dir:"
ls -l "$EXPECTED_PATH_ON_NODE"

# 3. EXECUTION
# -------------------------
echo "Starting training script..."
cd "$PROJECT_DIR" # Change to the project directory to run the script

python run-gqn.py \
    --data_dir "$DATA_PATH_ON_NODE" \
    --log_dir "$LOG_DIR" \
    --data_parallel "True" \
    --batch_size 1 \
    --workers 6 \
    --n_epochs 2 \
    $RESUME_ARGS

echo "Training script finished with exit code $?."

# 4. END OF JOB
# -------------------------
echo "Job finished at $(date)"