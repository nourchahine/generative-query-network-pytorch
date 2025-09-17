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

# --- Add your custom Google Cloud SDK to the PATH ---
# IMPORTANT: Replace this path with the actual location of your SDK folder
GCLOUD_SDK_PATH="/home/mila/c/chahinen/google-cloud-sdk"  
source "${GCLOUD_SDK_PATH}/path.bash.inc"
#LOCATION=$1   # example: /tmp/data

LOCATION=$SCRATCH/Data/gqndata
BATCH_SIZE=64 # example: 64
DATASET_NAME="shepard_metzler_5_parts"

#echo "Downloading data"
#gsutil -m cp -R gs://gqn-dataset/shepard_metzler_5_parts $LOCATION
#gcloud storage cp --recursive gs://gqn-dataset/shepard_metzler_5_parts $LOCATION



#echo "Deleting small records" # less than 10MB
#DATA_PATH="$LOCATION/shepard_metzler_5_parts/**/*.tfrecord"
#find $DATA_PATH -type f -size -10M | xargs rm

#cp -r /network/scratch/c/chahinen/Data/gqndata/ $SLURM_TMPDIR 

#echo "Converting data"
#python /home/mila/c/chahinen/gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py $SLURM_TMPDIR shepard_metzler_5_parts -b $BATCH_SIZE -m "train"
#echo "Training data: done"
#python /home/mila/c/chahinen/gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py $SLURM_TMPDIR shepard_metzler_5_parts -b $BATCH_SIZE -m "test"
#echo "Testing data: done"

# The directory containing train/test folders is now at this path:
DATASET_PATH_ON_NODE="$SLURM_TMPDIR/$DATASET_NAME"

echo "Converting training data"
# --- CHANGE #3: Updated the command to pass the full dataset path ---
python gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "train"
echo "Training data: done"

echo "Converting testing data"
python gqn_project/generative-query-network-pytorch/scripts/tfrecord-converter_new.py "$DATASET_PATH_ON_NODE" -b $BATCH_SIZE -m "test"
echo "Testing data: done"

echo "Job finished successfully."

echo "Removing original records"
rm -rf "$SLURM_TMPDIR/shepard_metzler_5_parts/**/*.tfrecord"

cp -r $SLURM_TMPDIR /network/scratch/c/chahinen/Data/gqndata/

rm -rf "$LOCATION/shepard_metzler_5_parts/**/*.tfrecord"