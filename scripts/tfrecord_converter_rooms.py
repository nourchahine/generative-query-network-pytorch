"""
tfrecord-converter (Modernized for TensorFlow 2.x)
Takes a directory of tf-records with GQN data and converts it
into a number of gzipped PyTorch records with a fixed batch size.
"""
import os
import gzip
import torch
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser

# Pose dimension is usually consistent
POSE_DIM = 5

def parse_tfrecord_function(serialized_example, seq_dim, img_dim):
    """
    Parses a single tf.train.Example into image and pose tensors.
    """
    feature_description = {
        'frames': tf.io.FixedLenFeature(shape=[seq_dim], dtype=tf.string),
        'cameras': tf.io.FixedLenFeature(shape=[seq_dim * POSE_DIM], dtype=tf.float32)
    }
    
    instance = tf.io.parse_single_example(serialized_example, feature_description)
    
    images = tf.map_fn(lambda x: tf.io.decode_jpeg(x, channels=3), instance['frames'], dtype=tf.uint8)
    poses = instance['cameras']
    
    # Reshape to the correct dimensions
    images = tf.reshape(images, [seq_dim, img_dim, img_dim, 3])
    poses = tf.reshape(poses, [seq_dim, POSE_DIM])
    
    return images, poses

def convert(record_path, batch_size, seq_dim, img_dim):
    """
    Processes a single tf-record file, batches data, and saves as .pt.gz files.
    """
    path, filename = os.path.split(record_path)
    basename, *_ = os.path.splitext(filename)
    print(f"Processing: {basename}")

    dataset = tf.data.TFRecordDataset(record_path)
    
    # Use a lambda to pass the dimensions to the parsing function
    parser_fn = lambda x: parse_tfrecord_function(x, seq_dim=seq_dim, img_dim=img_dim)
    dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    
    for i, (images_batch, poses_batch) in enumerate(dataset):
        batch_data = (images_batch.numpy(), poses_batch.numpy())
        
        output_filename = os.path.join(path, f"{basename}-{i:02}.pt.gz")
        with gzip.open(output_filename, 'wb') as f:
            torch.save(batch_data, f)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = ArgumentParser(description='Convert GQN tfrecords to PyTorch gzip files.')
    
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Number of sequences in each output file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='Whether to convert train or test')
    
    # ADDED: New arguments for dataset dimensions
    parser.add_argument('--img_size', type=int, default=64, help='Height and width of the images')
    parser.add_argument('--seq_len', type=int, default=15, help='Number of viewpoints per scene')
    
    args = parser.parse_args()

    dataset_path = os.path.expanduser(args.dataset_path)
    data_dir = os.path.join(dataset_path, args.mode)

    try:
        records = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
        records = [f for f in records if "tfrecord" in f]
    except FileNotFoundError:
        print(f"Error: Directory not found at {data_dir}")
        exit(1)

    print(f"Found {len(records)} TFRecord files in {data_dir}")
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        f = partial(convert, batch_size=args.batch_size, seq_dim=args.seq_len, img_dim=args.img_size)
        pool.map(f, records)
        
    print("Conversion complete.")