#
# SCRIPT 1: generate_poses.py (FINAL VERSION)
#
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os

# --- Configuration ---
# Directory containing your raw OptiTrack .csv files
OPTI_DATA_DIR = "./optitrack_data/"

# Directory to save the newly generated poses as .npy files
GENERATED_POSES_DIR = "./generated_poses/"
os.makedirs(GENERATED_POSES_DIR, exist_ok=True)

NUM_POSES_PER_SCENE = 10 # Number of camera views to generate for each scene
KDE_BANDWIDTH = 0.5      # Smoothing factor for the KDE. Adjust if needed.

# --- Main Script ---
print("Starting OptiTrack data processing and pose generation...")

for filename in os.listdir(OPTI_DATA_DIR):
    if not filename.endswith(".csv"):
        continue

    scene_name = os.path.splitext(filename)[0]
    print(f"\nProcessing scene: {scene_name}...")

    # 1. Load the OptiTrack CSV file using pandas
    #    - Skip the first 6 metadata rows to get to the header.
    #    - The actual header is on row 7, so we tell pandas it's row 0 after skipping.
    df = pd.read_csv(os.path.join(OPTI_DATA_DIR, filename), skiprows=6, header=[0, 1])

    # 2. Extract Position and Rotation data
    #    The columns are hierarchical. We select the 'RigidBody' columns.
    try:
        position_data = df['RigidBody']['Position']
        rotation_data = df['RigidBody']['Rotation']
    except KeyError:
        print(f"  ERROR: Could not find 'RigidBody' columns in {filename}. Please check the CSV format.")
        continue

    # 3. Convert Rotation from Degrees to Radians
    #    OptiTrack exports in degrees, but math functions and Blender use radians.
    rotation_rad = np.radians(rotation_data)

    # 4. Create the 5D Pose Vector [pos_x, pos_y, pos_z, yaw, pitch]
    #    - Position: We use the X, Y, and Z columns directly.
    #    - Yaw: This is rotation around the vertical (Y) axis in OptiTrack's YXZ format.
    #    - Pitch: This is rotation around the X axis.
    #    - We assume Roll (Z-axis rotation) is not needed for the camera pose.
    
    # Ensure columns exist before trying to access them
    required_pos_cols = {'X', 'Y', 'Z'}
    required_rot_cols = {'Y', 'X'}
    
    if not required_pos_cols.issubset(position_data.columns) or not required_rot_cols.issubset(rotation_rad.columns):
        print(f"  ERROR: Missing required Position/Rotation columns in {filename}.")
        continue

    poses_5d = np.zeros((len(df), 5))
    poses_5d[:, 0] = position_data['X']
    poses_5d[:, 1] = position_data['Y']
    poses_5d[:, 2] = position_data['Z']
    poses_5d[:, 3] = rotation_rad['Y'] # Yaw
    poses_5d[:, 4] = rotation_rad['X'] # Pitch
    
    # Drop any rows with missing data
    poses_5d = poses_5d[~np.isnan(poses_5d).any(axis=1)]

    if len(poses_5d) == 0:
        print(f"  ERROR: No valid pose data found in {filename} after cleaning.")
        continue
    
    print(f"  ...extracted {len(poses_5d)} valid poses from the CSV.")

    # 5. Fit Kernel Density Estimation (KDE) model
    kde = KernelDensity(kernel='gaussian', bandwidth=KDE_BANDWIDTH).fit(poses_5d)

    # 6. Sample new, plausible poses from the distribution
    new_plausible_poses = kde.sample(NUM_POSES_PER_SCENE)

    # 7. Save the generated poses for this scene
    output_path = os.path.join(GENERATED_POSES_DIR, f"{scene_name}.npy")
    np.save(output_path, new_plausible_poses)
    
    print(f"  ...generated {NUM_POSES_PER_SCENE} new poses and saved to {output_path}")

print("\nAll scenes processed. Pose generation complete.")