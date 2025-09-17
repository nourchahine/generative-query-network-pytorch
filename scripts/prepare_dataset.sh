#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/prepare_dataset-%j.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/prepare_dataset-%j.err
#SBATCH --job-name=prepare-data

# 1. SETUP
# =========================
echo "Job started on $(hostname) at $(date)"

# --- Load Modules and Activate Conda ---
module --force purge
module load anaconda/3
conda activate ~/venvs/gqn_env

# --- Configuration ---
DATASET_NAME="shepard_metzler_5_parts"
BATCH_SIZE=64

# --- Paths ---
# Directory for the RAW downloaded data (TFRecords)
RAW_DATA_DIR="/network/scratch/c/chahinen/Data/gqndata_raw"
# Directory for the FINAL CONVERTED data (PyTorch .pt.gz files)
CONVERTED_DATA_DIR="/network/scratch/c/chahinen/Data/gqndata_converted"

# Path for the dataset on the node's fast local disk
DATASET_PATH_ON_NODE="$SLURM_TMPDIR/$DATASET_NAME"


# 2. DOWNLOAD (if necessary)
# =========================
# Check if the raw data directory already exists. If not, download it.
if [ ! -d "$RAW_DATA_DIR/$DATASET_NAME" ]; then
    echo "Raw data not found at $RAW_DATA_DIR. Starting download from Google Cloud..."

    # Add Google Cloud SDK to PATH
    GCLOUD_SDK_PATH="/home/mila/c/chahinen/google-cloud-sdk"
    source "${GCLOUD_SDK_PATH}/path.bash.inc"

    # Create the target directory
    mkdir -p "$RAW_DATA_DIR"

    # Download the data using gsutil
    gsutil -m cp -R "gs://gqn-dataset/$DATASET_NAME" "$RAW_DATA_DIR/"

    if [ $? -eq 0 ]; then
        echo "Download completed successfully."
    else
        echo "Error: Download failed. Exiting."
        exit 1
    fi
else
    echo "Raw data already exists at $RAW_DATA_DIR. Skipping download."
fi


# 3. STAGE & CONVERT
# =========================
# --- Step 3a: Copy raw data to the fast, local node storage ---
echo "Copying raw data from scratch to node-local disk ($SLURM_TMPDIR)..."
cp -r "$RAW_DATA_DIR/$DATASET_NAME" "$SLURM_TMPDIR/"
echo "Data copy complete. Verifying contents:"
ls -l "$SLURM_TMPDIR"

# --- Step 3b: Run the Python conversion script ---
echo "Starting data conversion..."
PYTHON_SCRIPT_PATH="gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py"

echo "Converting training data..."
python "$PYTHON_SCRIPT_PATH" "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "train"
echo "Training data conversion: done"

echo "Converting testing data..."
python "$PYTHON_SCRIPT_PATH" "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "test"
echo "Testing data conversion: done"


# 4. SAVE & CLEAN UP
# =========================
# --- Step 4a: Copy converted data back to scratch ---
echo "Copying CONVERTED data from node back to scratch..."
# Create the final destination directory
mkdir -p "$CONVERTED_DATA_DIR"
# Copy only the converted dataset folder, not the whole temp directory
cp -r "$DATASET_PATH_ON_NODE" "$CONVERTED_DATA_DIR/"
echo "Converted data saved to: $CONVERTED_DATA_DIR"

# --- Step 4b: Clean up temporary files from the node ---
echo "Removing all temporary files from node..."
rm -rf "$SLURM_TMPDIR/*"

echo "Job finished successfully at $(date)."