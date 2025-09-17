# inspect_tfrecord.py (Version 2)
import tensorflow as tf
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_tfrecord.py <path_to_tfrecord_file>")
    sys.exit(1)

tfrecord_path = sys.argv[1]
print(f"--- Inspecting Record Structure in: {tfrecord_path} ---")

try:
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\nFound features in the TFRecord:")
        for key, feature in example.features.feature.items():
            # Check which type of list the feature holds
            if feature.HasField('bytes_list'):
                value_type = 'bytes_list'
                num_elements = len(feature.bytes_list.value)
            elif feature.HasField('float_list'):
                value_type = 'float_list'
                num_elements = len(feature.float_list.value)
            elif feature.HasField('int64_list'):
                value_type = 'int64_list'
                num_elements = len(feature.int64_list.value)
            else:
                value_type = 'unknown'
                num_elements = 0
            
            print(f"- Key: '{key}', Type: {value_type}, Number of elements: {num_elements}")

except Exception as e:
    print(f"An error occurred: {e}")