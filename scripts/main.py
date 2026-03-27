#!/usr/bin/env python3
"""
Main training script for AR4-COMPASS.

Usage:
    python scripts/main.py --method compass --env ar4_pick_place
    python scripts/main.py --method attention --env ar4_pick_place
    python scripts/main.py --method nam --env ar4_pick_place
"""

import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ar4_compass.models import create_model, MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description='AR4-COMPASS: Causal Discovery for Sim-to-Real Transfer'
    )
    
    # Model selection
    parser.add_argument(
        '--method', type=str, default='compass',
        choices=list(MODEL_REGISTRY.keys()),
        help='Causal discovery method'
    )
    
    # Environment
    parser.add_argument('--env', type=str, default='ar4_sim2sim',
                       help='Environment name')
    parser.add_argument('--n_params', type=int, default=40,
                       help='Number of environment parameters')
    parser.add_argument('--n_effects', type=int, default=6,
                       help='Number of trajectory difference components')
    parser.add_argument('--action_dim', type=int, default=6,
                       help='Action dimension')
    
    # Training
    parser.add_argument('--n_iters', type=int, default=10,
                       help='Number of main algorithm iterations')
    parser.add_argument('--n_epochs', type=int, default=2000,
                       help='Epochs per iteration for model training')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Data collection
    parser.add_argument('--n_real_rollouts', type=int, default=10,
                       help='Number of "real" rollouts per iteration')
    parser.add_argument('--n_sim_rollouts', type=int, default=64,
                       help='Number of sim rollouts per real rollout')
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=32,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--sparse_weight', type=float, default=0.01,
                       help='Sparsity regularization weight')
    parser.add_argument('--sparse_norm', type=float, default=1.0,
                       help='Sparsity norm (1=L1, 2=L2)')
    
    # Logging
    parser.add_argument('--logdir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Save model every N iterations')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_data(
    n_params: int,
    n_effects: int,
    action_dim: int,
    n_samples: int,
    true_causal_matrix: np.ndarray = None,
    noise_std: float = 0.1
):
    """
    Create synthetic data for testing.
    
    The trajectory difference is modeled as:
        d_k = Σ_i G_ik * f(ε_i) + noise
    
    where G is the true causal matrix.
    """
    if true_causal_matrix is None:
        # Create sparse random causal matrix
        true_causal_matrix = np.zeros((n_params, n_effects))
        # Randomly select ~10% of edges
        n_edges = int(0.1 * n_params * n_effects)
        edges = np.random.choice(n_params * n_effects, n_edges, replace=False)
        for edge in edges:
            i, k = edge // n_effects, edge % n_effects
            true_causal_matrix[i, k] = np.random.randn()
    
    # Generate random parameters
    env_params = np.random.randn(n_samples, n_params)
    actions = np.random.randn(n_samples, action_dim)
    
    # Generate trajectory differences based on causal structure
    traj_diff = np.zeros((n_samples, n_effects))
    for k in range(n_effects):
        for i in range(n_params):
            if abs(true_causal_matrix[i, k]) > 0.01:
                # Non-linear causal effect
                traj_diff[:, k] += true_causal_matrix[i, k] * np.tanh(env_params[:, i])
    
    # Add noise
    traj_diff += noise_std * np.random.randn(n_samples, n_effects)
    
    return {
        'env_params': torch.FloatTensor(env_params),
        'actions': torch.FloatTensor(actions),
        'traj_diff': torch.FloatTensor(traj_diff),
        'true_causal_matrix': true_causal_matrix
    }


def train_causal_model(
    model,
    data,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    writer: SummaryWriter = None,
    iteration: int = 0
):
    """Train the causal model for one iteration."""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    env_params = data['env_params'].to(device)
    actions = data['actions'].to(device)
    traj_diff = data['traj_diff'].to(device)
    
    n_samples = env_params.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_info = {}
        
        # Shuffle data
        perm = torch.randperm(n_samples)
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            
            batch_params = env_params[idx]
            batch_actions = actions[idx]
            batch_targets = traj_diff[idx]
            
            # Forward pass
            predictions = model(batch_params, batch_actions)
            
            # Compute loss
            loss, info = model.loss_function(predictions, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            for k, v in info.items():
                epoch_info[k] = epoch_info.get(k, 0) + v
        
        # Average metrics
        epoch_loss /= n_batches
        for k in epoch_info:
            epoch_info[k] /= n_batches
        
        # Log
        if writer is not None and epoch % 100 == 0:
            writer.add_scalar(f'train/loss', epoch_loss, iteration * n_epochs + epoch)
            for k, v in epoch_info.items():
                writer.add_scalar(f'train/{k}', v, iteration * n_epochs + epoch)
    
    return model, epoch_info


def evaluate_causal_discovery(
    model,
    true_causal_matrix: np.ndarray,
    threshold: float = 0.3
):
    """Evaluate causal discovery accuracy."""
    model.eval()
    
    with torch.no_grad():
        soft_weights, hard_weights = model.get_causal_weights(threshold)
        predicted = soft_weights[:model.input_dim, :].cpu().numpy()
    
    # Binarize for comparison
    true_binary = (np.abs(true_causal_matrix) > 0.01).astype(float)
    pred_binary = (predicted > threshold).astype(float)
    
    # Metrics
    tp = np.sum((true_binary == 1) & (pred_binary == 1))
    fp = np.sum((true_binary == 0) & (pred_binary == 1))
    fn = np.sum((true_binary == 1) & (pred_binary == 0))
    tn = np.sum((true_binary == 0) & (pred_binary == 0))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.exp_name or f"{args.method}_{args.env}_{timestamp}"
    logdir = Path(args.logdir) / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(logdir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Tensorboard
    writer = SummaryWriter(logdir / 'tensorboard')
    
    print(f"=" * 60)
    print(f"AR4-COMPASS: {args.method.upper()}")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Log dir: {logdir}")
    print(f"Parameters: {args.n_params}, Effects: {args.n_effects}")
    print(f"=" * 60)
    
    # Create true causal matrix (for synthetic experiments)
    true_causal_matrix = np.zeros((args.n_params, args.n_effects))
    # Simulate sparse causality: ~10% of param-effect pairs
    np.random.seed(args.seed)
    n_true_edges = int(0.1 * args.n_params * args.n_effects)
    true_edges = np.random.choice(
        args.n_params * args.n_effects, 
        n_true_edges, 
        replace=False
    )
    for edge in true_edges:
        i, k = edge // args.n_effects, edge % args.n_effects
        true_causal_matrix[i, k] = np.random.randn() * 2
    
    print(f"True causal matrix has {n_true_edges} non-zero edges")
    
    # Create model
    model = create_model(
        args.method,
        input_dim=args.n_params,
        output_dim=args.n_effects,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        sparse_weight=args.sparse_weight,
        sparse_norm=args.sparse_norm
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Main training loop
    best_f1 = 0.0
    
    for iteration in range(args.n_iters):
        print(f"\n--- Iteration {iteration + 1}/{args.n_iters} ---")
        
        # Generate synthetic data (replace with real env rollouts)
        n_samples = args.n_real_rollouts * args.n_sim_rollouts
        data = create_synthetic_data(
            args.n_params,
            args.n_effects,
            args.action_dim,
            n_samples,
            true_causal_matrix=true_causal_matrix
        )
        
        # Train model
        model, train_info = train_causal_model(
            model, data, args.n_epochs, args.batch_size,
            args.lr, device, writer, iteration
        )
        
        # Evaluate
        eval_metrics = evaluate_causal_discovery(
            model, true_causal_matrix, threshold=0.3
        )
        
        print(f"Train Loss: {train_info['mse']:.4f}")
        print(f"Causal Discovery - P: {eval_metrics['precision']:.3f}, "
              f"R: {eval_metrics['recall']:.3f}, F1: {eval_metrics['f1']:.3f}")
        
        # Log
        for k, v in eval_metrics.items():
            writer.add_scalar(f'eval/{k}', v, iteration)
        
        # Visualize causal graph
        if hasattr(model, 'visualize_causal_graph'):
            fig = model.visualize_causal_graph()
            writer.add_figure('causal_graph', fig, iteration)
        
        # Save best model
        if eval_metrics['f1'] > best_f1:
            best_f1 = eval_metrics['f1']
            torch.save(model.state_dict(), logdir / 'best_model.pt')
            print(f"  → New best F1: {best_f1:.3f}")
        
        # Save periodic checkpoint
        if (iteration + 1) % args.save_freq == 0:
            torch.save({
                'iteration': iteration,
                'model_state': model.state_dict(),
                'eval_metrics': eval_metrics
            }, logdir / f'checkpoint_{iteration+1}.pt')
    
    # Final evaluation
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Best F1 Score: {best_f1:.3f}")
    print(f"Results saved to: {logdir}")
    print(f"{'=' * 60}")
    
    writer.close()


if __name__ == '__main__':
    main()
