#
# SCRIPT 3: format_dataset.py (64x64 Version)
#
import os
import gzip
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import itertools

# --- Configuration ---
RENDERED_IMAGES_DIR = "./rendered_images/"
POSES_DIR = "./generated_poses/"
FINAL_DATASET_DIR = "./final_dataset_rooms/"

BATCH_SIZE = 32
VIEWS_PER_SCENE = 10

# --- CHANGE IS HERE ---
IMAGE_SIZE = 64
# --- END OF CHANGE ---

os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

# --- Helper Functions ---
def group_into_batches(iterable, batch_size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk

# --- Main Script ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

all_scene_names = sorted([d for d in os.listdir(RENDERED_IMAGES_DIR) if os.path.isdir(os.path.join(RENDERED_IMAGES_DIR, d))])
print(f"Found {len(all_scene_names)} total scenes.")

scene_batches = list(group_into_batches(all_scene_names, BATCH_SIZE))
print(f"Grouped into {len(scene_batches)} batches of up to {BATCH_SIZE} scenes each.")

for i, batch_of_scenes in enumerate(scene_batches):
    
    batch_images = []
    batch_viewpoints = []
    
    print(f"\nProcessing Batch {i+1}/{len(scene_batches)}...")
    
    for scene_name in batch_of_scenes:
        poses_path = os.path.join(POSES_DIR, f"{scene_name}.npy")
        poses_np = np.load(poses_path)
        viewpoints_tensor = torch.from_numpy(poses_np).float()
        
        image_tensors = []
        for v in range(VIEWS_PER_SCENE):
            image_path = os.path.join(RENDERED_IMAGES_DIR, scene_name, f"view_{v:03d}.png")
            image = Image.open(image_path).convert("RGB")
            
            image_tensor = transform(image)
            image_tensors.append(image_tensor.permute(1, 2, 0))

        images_tensor = torch.stack(image_tensors)
        
        batch_images.append(images_tensor)
        batch_viewpoints.append(viewpoints_tensor)
        
    final_images_tensor = torch.stack(batch_images)
    final_viewpoints_tensor = torch.stack(batch_viewpoints)

    print(f"Final Images Tensor Shape:   {final_images_tensor.shape}")
    print(f"Final Viewpoints Tensor Shape: {final_viewpoints_tensor.shape}")
    
    output_filename = f"batch_{i:03d}.pt.gz"
    output_filepath = os.path.join(FINAL_DATASET_DIR, output_filename)
    
    with gzip.open(output_filepath, "wb") as f:
        torch.save((final_images_tensor, final_viewpoints_tensor), f)
        
    print(f"Successfully saved batch to {output_filepath}")

print("\nDataset formatting complete!")