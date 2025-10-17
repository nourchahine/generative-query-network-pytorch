import torch
import torch.nn as nn
import torch.nn.functional as F

    
# class TowerRepresentation(nn.Module):
#     def __init__(self, n_channels, v_dim, r_dim=256, pool=True):
#         """
#         Network that generates a condensed representation
#         vector from a joint input of image and viewpoint.

#         Employs the tower/pool architecture described in the paper.

#         :param n_channels: number of color channels in input image
#         :param v_dim: dimensions of the viewpoint vector
#         :param r_dim: dimensions of representation
#         :param pool: whether to pool representation
#         """
#         super(TowerRepresentation, self).__init__()
#         # Final representation size
#         self.r_dim = k = r_dim
#         self.pool = pool

#         self.conv1 = nn.Conv2d(n_channels, k, kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(k, k, kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(k, k//2, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(k//2, k, kernel_size=2, stride=2)

#         self.conv5 = nn.Conv2d(k + v_dim, k, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(k + v_dim, k//2, kernel_size=3, stride=1, padding=1)
#         self.conv7 = nn.Conv2d(k//2, k, kernel_size=3, stride=1, padding=1)
#         self.conv8 = nn.Conv2d(k, k, kernel_size=1, stride=1)

#         self.avgpool  = nn.AvgPool2d(k//16)
    
    

#     def forward(self, x, v):
#         """
#         Send an (image, viewpoint) pair into the
#         network to generate a representation
#         :param x: image
#         :param v: viewpoint (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))
#         :return: representation
#         """
#         # Increase dimensions
#         v = v.view(v.size(0), -1, 1, 1)
#         v = v.repeat(1, 1, self.r_dim // 16, self.r_dim // 16)

#         # First skip-connected conv block
#         skip_in  = F.relu(self.conv1(x))
#         skip_out = F.relu(self.conv2(skip_in))

#         x = F.relu(self.conv3(skip_in))
#         x = F.relu(self.conv4(x)) + skip_out

#         # Second skip-connected conv block (merged)
#         skip_in = torch.cat([x, v], dim=1)
#         skip_out  = F.relu(self.conv5(skip_in))

#         x = F.relu(self.conv6(skip_in))
#         x = F.relu(self.conv7(x)) + skip_out

#         r = F.relu(self.conv8(x))

#         if self.pool:
#             r = self.avgpool(r)

#         return r

# class TowerRepresentation(nn.Module):
#     def __init__(self, n_channels, v_dim, r_dim=256, pool=True):
#         """
#         Network that generates a condensed representation
#         vector from a joint input of image and viewpoint.

#         Employs the tower/pool architecture described in the paper.

#         :param n_channels: number of color channels in input image
#         :param v_dim: dimensions of the viewpoint vector
#         :param r_dim: dimensions of representation
#         :param pool: whether to pool representation
#         """
#         super(TowerRepresentation, self).__init__()
#         # Final representation size
#         self.r_dim = k = r_dim
#         self.pool = pool

#         self.conv1 = nn.Conv2d(n_channels, k, kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(k, k, kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(k, k//2, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(k//2, k, kernel_size=2, stride=2)

#         self.conv5 = nn.Conv2d(k + v_dim, k, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(k + v_dim, k//2, kernel_size=3, stride=1, padding=1)
#         self.conv7 = nn.Conv2d(k//2, k, kernel_size=3, stride=1, padding=1)
#         self.conv8 = nn.Conv2d(k, k, kernel_size=1, stride=1)
        
#         # --- CHANGE 2: Use Adaptive Pooling ---
#         # This layer will always pool the feature map to a 1x1 output
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x, v):
#         """
#         Send an (image, viewpoint) pair into the
#         network to generate a representation
#         :param x: image
#         :param v: viewpoint (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))
#         :return: representation
#         """
#         # First skip-connected conv block
#         skip_in  = F.relu(self.conv1(x))
#         skip_out = F.relu(self.conv2(skip_in))

#         x = F.relu(self.conv3(skip_in))
#         x = F.relu(self.conv4(x)) + skip_out

#         # --- CHANGE 1: Dynamic Viewpoint Broadcasting ---
#         # Get the spatial dimensions of the image features dynamically
#         _, _, h, w = x.shape

#         # Increase dimensions of viewpoint and repeat to match
#         v = v.view(v.size(0), -1, 1, 1)
#         v = v.repeat(1, 1, h, w)
        
#         # Second skip-connected conv block (merged)
#         skip_in = torch.cat([x, v], dim=1)
#         skip_out  = F.relu(self.conv5(skip_in))

#         x = F.relu(self.conv6(skip_in))
#         x = F.relu(self.conv7(x)) + skip_out

#         r = F.relu(self.conv8(x))

#         if self.pool:
#             r = self.avgpool(r)

#         return r

import torch
import torch.nn as nn
import torch.nn.functional as F

class TowerRepresentation(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim, pool=True):
        super(TowerRepresentation, self).__init__()
        self.r_dim = r_dim

        # Tower architecture
        self.conv1 = nn.Conv2d(x_dim + v_dim, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256 + v_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.conv8 = nn.Conv2d(256, r_dim, kernel_size=1, stride=1)
        
        self.pool = pool

    def forward(self, x, v):
        """
        The forward pass of the GQN representation network.
        """
        # --- FIX IS HERE ---

        # 1. Reshape the inputs to be compatible with convolutional layers.
        # The network expects a batch of single images, not a batch of scenes.
        batch_size, num_views, *x_dims = x.shape
        _, _, *v_dims = v.shape

        # Merge the batch_size and num_views dimensions
        # x before: [1, num_views, C, H, W] -> after: [num_views, C, H, W]
        x = x.view(batch_size * num_views, *x_dims)
        v = v.view(batch_size * num_views, *v_dims)

        # 2. Broadcast the viewpoint vector v across the spatial dimensions of x.
        # v before: [num_views, v_dim] -> after: [num_views, v_dim, H, W]
        h, w = x.shape[-2:]
        v_broadcast = v.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

        # 3. Concatenate x and v and pass through the convolutional tower
        # This part of your code was already correct.
        skip_in = F.relu(self.conv1(torch.cat([x, v_broadcast], dim=1)))
        skip_out = F.relu(self.conv2(skip_in))
        
        x = F.relu(self.conv3(skip_out))
        x = F.relu(self.conv4(x))

        h, w = x.shape[-2:]
        v_broadcast = v.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        
        x = F.relu(self.conv5(torch.cat([x, v_broadcast], dim=1)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        r_i = F.relu(self.conv8(x + skip_out)) # r_i is the representation for each view

        # 4. If pooling, perform a global average pool. Then sum the representations.
        if self.pool:
            r_i = r_i.mean(dim=[2, 3]) # [num_views, r_dim]

        # 5. Sum the representations of all context views to get the scene representation `r`.
        # This is the step described in the paper (r = Σi ri).
        # r before: [num_views, r_dim] -> after: [r_dim]
        r = r_i.sum(dim=0)

        # 6. Reshape `r` to have the expected dimensions for the generator.
        # It needs a batch dimension and spatial dimensions of 1x1.
        # r before: [r_dim] -> after: [1, r_dim, 1, 1]
        r = r.view(batch_size, self.r_dim, 1, 1)

        return r
    
class PyramidRepresentation(nn.Module):
    def __init__(self, n_channels, v_dim, r_dim=256):
        """
        Network that generates a condensed representation
        vector from a joint input of image and viewpoint.

        Employs the pyramid architecture described in the paper.

        :param n_channels: number of color channels in input image
        :param v_dim: dimensions of the viewpoint vector
        :param r_dim: dimensions of representation
        """
        super(PyramidRepresentation, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim

        self.conv1 = nn.Conv2d(n_channels + v_dim, k//8, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(k//8, k//4, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(k//4, k//2, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(k//2, k, kernel_size=8, stride=8)

    def forward(self, x, v):
        """
        Send an (image, viewpoint) pair into the
        network to generate a representation
        :param x: image
        :param v: viewpoint (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))
        :return: representation
        """
        # Increase dimensions
        batch_size, _, h, w = x.shape

        v = v.view(batch_size, -1, 1, 1)
        v = v.repeat(1, 1, h, w)

        # Merge representation
        r = torch.cat([x, v], dim=1)

        r  = F.relu(self.conv1(r))
        r  = F.relu(self.conv2(r))
        r  = F.relu(self.conv3(r))
        r  = F.relu(self.conv4(r))

        return r