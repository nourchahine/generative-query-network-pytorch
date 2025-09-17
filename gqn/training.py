import random
import torch

class Annealer(object):
    def __init__(self, init, delta, steps):
        self.init = init
        self.delta = delta
        self.steps = steps
        self.s = 0
        self.data = self.__repr__()
        self.recent = init

    def __repr__(self):
        return {"init": self.init, "delta": self.delta, "steps": self.steps, "s": self.s}

    def __iter__(self):
        return self

    def __next__(self):
        self.s += 1
        value = max(self.delta + (self.init - self.delta) * (1 - self.s / self.steps), self.delta)
        self.recent = value
        return value
    # --- FIX STARTS HERE: Add the required methods for Ignite checkpointing ---

    def state_dict(self):
        """Returns the state of the annealer for checkpointing."""
        return {
            'init': self.init,
            'delta': self.delta,
            'steps': self.steps,
            's': self.s,
            'recent': self.recent
        }

    def load_state_dict(self, state_dict):
        """Loads the annealer's state from a checkpoint."""
        self.init = state_dict['init']
        self.delta = state_dict['delta']
        self.steps = state_dict['steps']
        self.s = state_dict['s']
        self.recent = state_dict['recent']
    
    # --- FIX ENDS HERE ---

# def partition(images, viewpoints):
#     """
#     Partition batch into context and query sets.
#     :param images
#     :param viewpoints
#     :return: context images, context viewpoint, query image, query viewpoint
#     """
#     # Maximum number of context points to use
#     _, b, m, *x_dims = images.shape
#     _, b, m, *v_dims = viewpoints.shape

#     # "Squeeze" the batch dimension
#     images = images.view((-1, m, *x_dims))
#     viewpoints = viewpoints.view((-1, m, *v_dims))

#     # Sample random number of views
#     n_context = random.randint(2, m - 1)
#     indices = random.sample([i for i in range(m)], n_context)

#     # Partition into context and query sets
#     context_idx, query_idx = indices[:-1], indices[-1]

#     x, v = images[:, context_idx], viewpoints[:, context_idx]
#     x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

#     return x, v, x_q, v_q



def partition(images, viewpoints):
    """
    (Corrected and Flexible Version)
    Partitions a batch into context and query sets, dynamically handling tensor shapes.
    """
    # Get the dimensions of the input tensors
    # Shape of images: (batch_size, num_views, C, H, W)
    batch_size, num_views, *x_dims_list = images.shape
    
    # The number of context points is a random number between 1 and all-but-one of the views.
    # The original code's randint(2, m-1) would fail if num_views was small.
    if num_views > 1:
        n_context = random.randint(1, num_views - 1)
    else:
        # Handle the edge case of only one view
        n_context = 1

    # Sample `n_context` views for the context set
    context_idx = random.sample(range(num_views), k=n_context)

    # The query point is a single view chosen randomly from the whole set of views
    query_idx = random.randint(0, num_views - 1)
    
    # Select the data using the sampled indices
    x = images[:, context_idx]
    v = viewpoints[:, context_idx]
    x_q = images[:, query_idx]
    v_q = viewpoints[:, query_idx]

    return x, v, x_q, v_q