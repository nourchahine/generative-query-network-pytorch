#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8         # Increased CPUs for larger images
#SBATCH --mem=64GB                # Increased memory for larger images
#SBATCH --time=24:00:00
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/prepare_rooms-%j.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/prepare_rooms-%j.err
#SBATCH --job-name=prepare-rooms

# 1. SETUP
# =========================
echo "Job started on $(hostname) at $(date)"
module --force purge
module load anaconda/3
conda activate ~/venvs/gqn_env

# --- Configuration for the NEW dataset ---
DATASET_NAME="rooms_free_camera_with_object_rotations"
BATCH_SIZE=32 # Reduced batch size for larger images to conserve memory
IMG_SIZE=128
SEQ_LEN=10

# --- Paths ---
RAW_DATA_DIR="/network/scratch/c/chahinen/Data/gqndata_raw_rooms"
CONVERTED_DATA_DIR="/network/scratch/c/chahinen/Data/gqndata_converted_rooms"
DATASET_PATH_ON_NODE="$SLURM_TMPDIR/$DATASET_NAME"

# 2. DOWNLOAD (if necessary)
# =========================
if [ ! -d "$RAW_DATA_DIR/$DATASET_NAME" ]; then
    echo "Raw data not found. Starting download..."
    GCLOUD_SDK_PATH="/home/mila/c/chahinen/google-cloud-sdk"
    source "${GCLOUD_SDK_PATH}/path.bash.inc"
    mkdir -p "$RAW_DATA_DIR"
    gsutil -m cp -R "gs://gqn-dataset/$DATASET_NAME" "$RAW_DATA_DIR/"
    echo "Download completed."
else
    echo "Raw data already exists. Skipping download."
fi

# 3. STAGE & CONVERT
# =========================
echo "Copying raw data to node-local disk ($SLURM_TMPDIR)..."
cp -r "$RAW_DATA_DIR/$DATASET_NAME" "$SLURM_TMPDIR/"

echo "Starting data conversion..."
PYTHON_SCRIPT_PATH="gqn_project/generative-query-network-pytorch/scripts/tfrecord_converter_rooms.py"

# MODIFIED: Pass new dimensions to the python script
echo "Converting training data..."
python "$PYTHON_SCRIPT_PATH" "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "train" --img_size $IMG_SIZE --seq_len $SEQ_LEN
echo "Training data conversion: done"

echo "Converting testing data..."
python "$PYTHON_SCRIPT_PATH" "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "test" --img_size $IMG_SIZE --seq_len $SEQ_LEN
echo "Testing data conversion: done"

# 4. SAVE & CLEAN UP
# =========================
echo "Copying CONVERTED data back to scratch..."
mkdir -p "$CONVERTED_DATA_DIR"
cp -r "$DATASET_PATH_ON_NODE" "$CONVERTED_DATA_DIR/"
echo "Converted data saved to: $CONVERTED_DATA_DIR"

rm -rf "$SLURM_TMPDIR/*"
echo "Job finished successfully."