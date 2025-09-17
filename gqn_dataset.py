import os, gzip
import numpy as np
import torch
from torch.utils.data import Dataset


def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    # NOTE: The original GQN code expects a batch dimension.
    # We will add it, transform, and then remove it.
    if len(v.shape) == 2:
        v = v.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)
    
    if squeeze_output:
        v_hat = v_hat.squeeze(0)

    return v_hat


class GQNDataset(Dataset): # RENAMED from ShepardMetzler
    """
    GQN Dataset.
    """
    def __init__(self, root_dir, train=True, transform=None, fraction=1.0, target_transform=None): # Removed default for target_transform
        super().__init__()
        assert fraction > 0.0 and fraction <= 1.0
        prefix = "train" if train else "test"
        self.root_dir = os.path.join(root_dir, prefix)
        self.records = sorted([p for p in os.listdir(self.root_dir) if "pt" in p])
        self.records = self.records[:int(len(self.records)*fraction)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # The length of the dataset is the number of pre-computed batch files
        return len(self.records)

    def __getitem__(self, idx):
        # idx refers to the index of the batch file to load
        scene_path = os.path.join(self.root_dir, self.records[idx])
        with gzip.open(scene_path, "r") as f:
            # Load the data and fix the security warning
            data = torch.load(f)
            
            # --- FIX STARTS HERE ---
            # The loaded 'data' is already a tuple of (images_batch, viewpoints_batch).
            # We just need to unpack it directly.
            images, viewpoints = data
            # --- FIX ENDS HERE ---

        # The data is already a batch of numpy arrays. No need to stack.
        # uint8 -> float32 and (B, S, H, W, C) -> (B, S, C, H, W)
        # B: batch size, S: sequence length (15), C: channels (3), H/W: height/width (64)
        images = images.transpose(0, 1, 4, 2, 3)
        images = torch.from_numpy(images).float() / 255.0

        if self.transform:
            # The transform should expect a batch of images
            images = self.transform(images)

        viewpoints = torch.from_numpy(viewpoints).float()
        if self.target_transform:
            # The target transform needs to be applied to the batch
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints

