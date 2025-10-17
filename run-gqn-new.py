import random
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

from gqn import GenerativeQueryNetwork, Annealer
from gqn_datasets import SceneDataset # <-- IMPORT THE NEW DATASET

# --- Seeding for reproducibility ---
random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed_all(99)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network - Paper Aligned Implementation')
    # --- General arguments ---
    parser.add_argument('--dataset_dir', type=str, help='Location of the dataset directory containing train/test folders')
    parser.add_argument('--log_dir', type=str, default="log", help='Location for logging and checkpoints')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--fraction', type=float, default=1, help='fraction of the dataset to use for quick testing (default: 1, i.e., full dataset)')

    # --- Training arguments aligned with paper ---
    parser.add_argument('--max_steps', type=int, default=2*10**6, help='Total number of training steps (default: 2 million)')
    
    args = parser.parse_args()

    # --- Dataset Hyperparameters ---
    # As we discovered, both datasets use v_dim=5 in your converted files.
    V_DIM = 5
    SIGMA_ANNEAL_STEPS = 200000
    MU_ANNEAL_STEPS = 1.6 * 10**6

    # --- Data Loaders (using the new SceneDataset) ---
    #train_dataset = SceneDataset(root_dir=os.path.join(args.dataset_dir, "train"))
    # Pass the fraction to the SceneDataset
    train_dataset = SceneDataset(root_dir=os.path.join(args.dataset_dir, "train"), fraction=args.fraction)

    # The paper uses a batch size of 1 (one scene per step)
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    # --- Model, Optimizer, and Annealers ---
    model = GenerativeQueryNetwork(x_dim=3, v_dim=V_DIM, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Initial LR

    sigma_scheme = Annealer(2.0, 0.7, SIGMA_ANNEAL_STEPS)
    mu_scheme = Annealer(5e-4, 5e-5, MU_ANNEAL_STEPS)

    # --- Main Training Step (Paper Aligned) ---
    def step(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        x, v = batch # A single scene: x=[V,C,H,W], v=[V,5]
        x = x.squeeze(0).to(device)
        v = v.squeeze(0).to(device)

        num_total_views = x.size(0)
        num_context = random.randint(1, 5) # Random number of context views

        # Randomly shuffle and partition into context and query
        indices = torch.randperm(num_total_views)
        context_indices, query_indices = indices[:num_context], indices[num_context:]
        
        x_context, v_context = x[context_indices], v[context_indices]
        x_query, v_query = x[query_indices], v[query_indices]

        # Add a batch dimension of 1 for the model
        x_context, v_context = x_context.unsqueeze(0), v_context.unsqueeze(0)
        x_query, v_query = x_query.unsqueeze(0), v_query.unsqueeze(0)
        
        x_mu, _, kl = model(x_context, v_context, x_query, v_query)
        
        sigma = next(sigma_scheme)
        ll = Normal(x_mu, sigma).log_prob(x_query)

        likelihood = torch.mean(ll)
        kl_divergence = torch.mean(kl)
        
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward()
        optimizer.step()

        mu = next(mu_scheme)
        for group in optimizer.param_groups:
            group["lr"] = mu

        return {"elbo": elbo.item(), "kl": kl_divergence.item(), "sigma": sigma, "mu": mu}

    # --- Ignite Engine and Handlers ---
    trainer = Engine(step)
    writer = SummaryWriter(log_dir=args.log_dir)

    metric_names = ["elbo", "kl", "sigma", "mu"]
    for metric in metric_names:
        RunningAverage(output_transform=lambda x, name=metric: x[name]).attach(trainer, metric)
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Checkpointing (saves every 10,000 steps for recovery)
    checkpoint_handler = ModelCheckpoint(
        args.log_dir, filename_pattern="checkpoint_{iteration}.pth",
        n_saved=5, require_empty=False
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=10000),
        checkpoint_handler,
        {'model': model, 'optimizer': optimizer, 'annealers': (sigma_scheme, mu_scheme)}
    )

    # Terminate training at max_steps
    @trainer.on(Events.ITERATION_COMPLETED(once=args.max_steps))
    def terminate_training(engine):
        engine.terminate()

    # Logging
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        if engine.state.iteration % 100 == 0:
            writer.add_scalar(f"training/elbo", engine.state.metrics['elbo'], engine.state.iteration)
            writer.add_scalar(f"training/kl", engine.state.metrics['kl'], engine.state.iteration)

    # Use a large number of epochs; termination is handled by max_steps
    trainer.run(train_loader, max_epochs=10000)
    writer.close()