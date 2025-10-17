import torch.nn as nn
from .representation import TowerRepresentation
from .generator import GeneratorNetwork

class GenerativeQueryNetwork(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12):
        super(GenerativeQueryNetwork, self).__init__()
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim)
        #self.generator = GeneratorNetwork(v_dim, r_dim, z_dim, h_dim, L)
        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)

    def forward(self, context_x, context_v, query_x, query_v):
        """
        The forward pass of the GQN.
        """
        # --- FIX IS HERE ---

        # 1. Get the scene representation `r`.
        #    Shape: [batch_size, r_dim, 1, 1]
        r = self.representation(context_x, context_v)

        # 2. Reshape the query inputs for the generator.
        #    The generator expects a batch of single images, not a batch of scenes.
        batch_size, num_query, *x_dims = query_x.shape
        _, _, *v_dims = query_v.shape

        # Merge the batch_size and num_query dimensions together
        # query_x before: [1, num_query, C, H, W] -> after: [num_query, C, H, W]
        query_x = query_x.view(batch_size * num_query, *x_dims)
        query_v = query_v.view(batch_size * num_query, *v_dims)

        # 3. Expand the scene representation `r` to match the query batch.
        #    We need to provide the same `r` for every query view.
        # r before: [1, r_dim, 1, 1] -> after: [num_query, r_dim, 1, 1]
        r = r.repeat_interleave(num_query, dim=0)

        # 4. Pass the correctly shaped tensors to the generator.
        x_mu, kl = self.generator(query_x, query_v, r)

        # Reshape the output to match the original query shape
        x_mu = x_mu.view(batch_size, num_query, *x_dims)

        return x_mu, r, kl # Returning 'r' is also good practice