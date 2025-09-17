"""
tfrecord-converter (Modernized for TensorFlow 2.x)

Takes a directory of tf-records with Shepard-Metzler data
and converts it into a number of gzipped PyTorch records
with a fixed batch size.
"""
import os
import gzip
import torch
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser

# Constants from the original script
POSE_DIM, IMG_DIM, SEQ_DIM = 5, 64, 15

def parse_tfrecord_function(serialized_example):
    """
    Parses a single tf.train.Example into image and pose tensors.
    This function will be used in our tf.data pipeline.
    """
    feature_description = {
        'frames': tf.io.FixedLenFeature(shape=[SEQ_DIM], dtype=tf.string),
        'cameras': tf.io.FixedLenFeature(shape=[SEQ_DIM * POSE_DIM], dtype=tf.float32)
    }
    
    instance = tf.io.parse_single_example(serialized_example, feature_description)
    
    # Decode the JPEGs
    # The `tf.map_fn` is a clean way to apply the decoding to each of the 15 frame strings
    images = tf.map_fn(lambda x: tf.io.decode_jpeg(x, channels=3), instance['frames'], dtype=tf.uint8)
    poses = instance['cameras']
    
    # Reshape to the correct dimensions
    images = tf.reshape(images, [SEQ_DIM, IMG_DIM, IMG_DIM, 3])
    poses = tf.reshape(poses, [SEQ_DIM, POSE_DIM])
    
    return images, poses

def convert(record_path, batch_size):
    """
    Processes a single tf-record file, batches the data, and saves it
    as gzipped PyTorch (.pt.gz) files.
    """
    path, filename = os.path.split(record_path)
    basename, *_ = os.path.splitext(filename)
    print(f"Processing: {basename}")

    # 1. Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(record_path)
    
    # 2. Map the parsing function over the dataset
    dataset = dataset.map(parse_tfrecord_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 3. Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # 4. Iterate over the batches and save them
    for i, (images_batch, poses_batch) in enumerate(dataset):
        # Convert tensors to numpy arrays before saving
        batch_data = (images_batch.numpy(), poses_batch.numpy())
        
        output_filename = os.path.join(path, f"{basename}-{i:02}.pt.gz")
        with gzip.open(output_filename, 'wb') as f:
            torch.save(batch_data, f)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = ArgumentParser(description='Convert GQN tfrecords to PyTorch gzip files.')
    
    # --- CHANGE #1: Simplified the arguments ---
    # We now take just one argument: the direct path to the dataset directory
    parser.add_argument('dataset_path',
                        help='Path to the dataset directory (e.g., /path/to/shepard_metzler_5_parts)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='Number of sequences in each output file')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='Whether to convert train or test')
    args = parser.parse_args()

    # --- CHANGE #2: Simplified the path construction ---
    dataset_path = os.path.expanduser(args.dataset_path)
    data_dir = os.path.join(dataset_path, args.mode)

    # Find all records (this part remains the same)
    try:
        records = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
        print(records)
        records = [f for f in records if "tfrecord" in f]
    except FileNotFoundError:
        print(f"Error: Directory not found at {data_dir}")
        print("Please ensure the path passed to the script is correct.")
        exit(1)


    print(f"Found {len(records)} TFRecord files in {data_dir}")
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        f = partial(convert, batch_size=args.batch_size)
        pool.map(f, records)
        
    print("Conversion complete.")