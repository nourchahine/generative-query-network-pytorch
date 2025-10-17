import torch
import os
import gzip
from torch.utils.data import Dataset
import numpy as np

class SceneDataset(Dataset):
    """
    A PyTorch Dataset that can read pre-batched .pt.gz files but
    presents each scene individually, making it compatible with the
    paper's scene-wise training methodology.
    """
    def __init__(self, root_dir, fraction=1.0):
        self.root_dir = root_dir
        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt.gz')])
        
        self.scene_map = []
        print(f"Indexing scenes from {len(self.files)} files...")
        for file_idx, filepath in enumerate(self.files):
            # We need to peek into the file to see how many scenes it contains
            with gzip.open(filepath, 'rb') as f:
                # --- FIX IS HERE ---
                # Reverting to weights_only=False because our files contain NumPy arrays.
                # This is safe because we trust the source of our data.
                images, _ = torch.load(f, weights_only=False)
                # --- END OF FIX ---
                num_scenes_in_file = images.shape[0]
            
            for scene_idx in range(num_scenes_in_file):
                self.scene_map.append((file_idx, scene_idx))
        
        num_scenes = int(len(self.scene_map) * fraction)
        self.scene_map = self.scene_map[:num_scenes]
        
        print(f"Found a total of {len(self.scene_map)} scenes (using {fraction*100:.1f}% of data).")

        self.loaded_files = {} # Cache for loaded files

    def __len__(self):
        return len(self.scene_map)

    def __getitem__(self, idx):
        file_idx, scene_idx = self.scene_map[idx]
        
        if file_idx not in self.loaded_files:
            filepath = self.files[file_idx]
            with gzip.open(filepath, 'rb') as f:
                # --- ALSO FIX HERE ---
                self.loaded_files[file_idx] = torch.load(f, weights_only=False)
                # --- END OF FIX ---

        images, viewpoints = self.loaded_files[file_idx]
        
        scene_images = images[scene_idx]
        scene_viewpoints = viewpoints[scene_idx]
        
        # If the loaded data is a numpy array, convert it to a torch tensor
        if isinstance(scene_images, np.ndarray):
            scene_images = torch.from_numpy(scene_images)
        if isinstance(scene_viewpoints, np.ndarray):
            scene_viewpoints = torch.from_numpy(scene_viewpoints)

        # The data is already (V, H, W, C). Permute to (V, C, H, W) for PyTorch standard.
        if scene_images.dim() == 4 and scene_images.shape[3] == 3:
            scene_images = scene_images.permute(0, 3, 1, 2)
            
        return scene_images.float() / 255.0, scene_viewpoints.float()