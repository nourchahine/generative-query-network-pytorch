#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/network/scratch/c/chahinen/sbatch_out/dataset%A%a.out
#SBATCH --error=/network/scratch/c/chahinen/sbatch_err/dataset%A%a.err
#SBATCH --job-name=dataset

# Load modules and activate Conda
# -------------------------
module --force purge
module load anaconda/3

# Activate your GQN Conda environment
conda activate ~/venvs/gqn_env

# --- Configuration ---
BATCH_SIZE=64
DATASET_NAME="shepard_metzler_5_parts"
SOURCE_DATA_PATH="/network/scratch/c/chahinen/Data/gqndata/$DATASET_NAME"
DATASET_PATH_ON_NODE="$SLURM_TMPDIR/$DATASET_NAME"

# --- Step 1: Copy data to the fast, local node storage ---
echo "Preparing environment on node $(hostname)..."
echo "Source data is at: $SOURCE_DATA_PATH"
echo "Destination node directory is: $SLURM_TMPDIR"
echo "Copying data..."

# Make sure the source path exists before trying to copy
if [ ! -d "$SOURCE_DATA_PATH" ]; then
    echo "Error: Source directory $SOURCE_DATA_PATH not found!"
    exit 1
fi

# Use `cp -r` to copy the entire dataset directory
cp -r "$SOURCE_DATA_PATH" "$SLURM_TMPDIR/"
echo "Data copy complete. Verifying contents:"
ls -l "$SLURM_TMPDIR" # Should now show 'shepard_metzler_5_parts'

# --- Step 2: Run the conversion script ---
echo "Converting training data..."
python gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "train"
echo "Training data: done"

echo "Converting testing data..."
python gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "test"
echo "Testing data: done"

# --- Step 3: Copy results back and clean up ---
echo "Job finished successfully."
echo "Copying converted data back to scratch..."

# This copies the entire contents of SLURM_TMPDIR back.
# You might want to be more specific if other files are created.
cp -r "$SLURM_TMPDIR/" "/network/scratch/c/chahinen/Data/gqndata_converted/"

echo "Removing original records from node..."
# This is a safer way to remove the copied data
rm -rf "$DATASET_PATH_ON_NODE"

echo "All done."