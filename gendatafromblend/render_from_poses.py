#
# SCRIPT 2: render_from_poses.py (64x64 Version)
#
import bpy
import numpy as np
import math
import os

# ===================================================================
# ==> IMPORTANT: YOU MUST EDIT THESE TWO LINES
# ===================================================================
POSES_DIR = "/path/to/your/generated_poses/"
OUTPUT_IMAGE_DIR = "/path/to/your/rendered_images/"
CAMERA_NAME = "Camera"
# ===================================================================

# --- Main Script ---
try:
    camera = bpy.data.objects[CAMERA_NAME]
    print(f"Successfully found camera object named '{CAMERA_NAME}'.")
except KeyError:
    print(f"FATAL ERROR: Could not find an object named '{CAMERA_NAME}' in the scene.")

# --- CHANGE IS HERE ---
# Set the render resolution to 64x64
bpy.context.scene.render.resolution_x = 64
bpy.context.scene.render.resolution_y = 64
# --- END OF CHANGE ---

bpy.context.scene.render.resolution_percentage = 100
print("Set render resolution to 64x64.")

for pose_filename in os.listdir(POSES_DIR):
    if not pose_filename.endswith(".npy"):
        continue
    
    scene_name = os.path.splitext(pose_filename)[0]
    poses_path = os.path.join(POSES_DIR, pose_filename)
    poses = np.load(poses_path)
    
    print(f"Rendering scene: {scene_name}...")
    
    for i, pose in enumerate(poses):
        camera.location.x = pose[0]
        camera.location.y = pose[1]
        camera.location.z = pose[2]
        
        yaw_rad = pose[3]
        pitch_rad = pose[4]
        
        camera.rotation_mode = 'XYZ'
        camera.rotation_euler.x = pitch_rad 
        camera.rotation_euler.y = 0
        camera.rotation_euler.z = yaw_rad
        
        scene_output_dir = os.path.join(OUTPUT_IMAGE_DIR, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        output_path = os.path.join(scene_output_dir, f"view_{i:03d}")
        bpy.context.scene.render.filepath = output_path
        
        bpy.ops.render.render(write_still=True)
        
        print(f"  ...rendered view {i+1}/{len(poses)} to {output_path}.png")

print("All scenes have been rendered.")