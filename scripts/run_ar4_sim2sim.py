#!/usr/bin/env python3
"""
AR4 Sim-to-Sim Experiment: Compare Causal Discovery Methods

This script runs a complete sim-to-sim experiment on AR4:
1. Create "sim" and "real" environments with different parameters
2. Collect trajectories with a fixed policy
3. Train all causal discovery models
4. Compare trajectory alignment and parameter recovery

Usage:
    python scripts/run_ar4_sim2sim.py --n_iters 5 --n_epochs 1000
"""

import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import copy

import torch
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ar4_compass.models import create_model, MODEL_REGISTRY
from ar4_compass.envs import (
    AR4MuJoCoEnv, AR4EnvConfig, AR4EnvParams,
    rollout_trajectory, compute_trajectory_difference,
    sine_wave_policy, TRAJECTORY_EFFECT_NAMES
)


def parse_args():
    parser = argparse.ArgumentParser(description='AR4 Sim-to-Sim Experiment')
    
    # Experiment settings
    parser.add_argument('--n_iters', type=int, default=5,
                       help='Number of COMPASS iterations')
    parser.add_argument('--n_epochs', type=int, default=1000,
                       help='Training epochs per iteration')
    parser.add_argument('--n_real_rollouts', type=int, default=5,
                       help='Number of "real" rollouts per iteration')
    parser.add_argument('--n_dr_samples', type=int, default=32,
                       help='Domain randomization samples per real rollout')
    
    # Model settings
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['compass', 'attention', 'nam'],
                       help='Causal discovery methods to compare')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--sparse_weight', type=float, default=0.01,
                       help='Sparsity regularization weight')
    
    # Environment settings
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum trajectory length')
    
    # Logging
    parser.add_argument('--logdir', type=str, default='./experiments',
                       help='Log directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_target_params(base_params: AR4EnvParams, seed: int = 0) -> Tuple[AR4EnvParams, List[int]]:
    """
    Create target ("real") environment parameters by modifying some parameters.
    
    Returns:
        target_params: Modified parameters
        modified_indices: Indices of parameters that were changed
    """
    np.random.seed(seed)
    target = base_params.copy()
    modified_indices = []
    
    # Modify joint damping (indices 0-5)
    target.joint_damping = base_params.joint_damping * np.random.uniform(0.5, 1.5, 6)
    modified_indices.extend(range(6))
    
    # Modify some friction loss (indices 6-11, only change 2 joints)
    joints_to_modify = np.random.choice(6, 2, replace=False)
    target.joint_frictionloss = base_params.joint_frictionloss.copy()
    for j in joints_to_modify:
        target.joint_frictionloss[j] *= np.random.uniform(0.5, 2.0)
        modified_indices.append(6 + j)
    
    # Modify contact friction (index 25)
    target.contact_friction_sliding = base_params.contact_friction_sliding * np.random.uniform(0.7, 1.3)
    modified_indices.append(25)
    
    return target, modified_indices


def domain_randomize_params(
    base_params: AR4EnvParams,
    n_samples: int,
    dr_scale: float = 0.3
) -> List[AR4EnvParams]:
    """
    Generate domain randomized parameter samples around base parameters.
    
    Args:
        base_params: Base parameters to randomize around
        n_samples: Number of samples
        dr_scale: Randomization scale (fraction of base value)
        
    Returns:
        List of randomized parameter sets
    """
    samples = []
    base_vec = base_params.to_vector()
    
    for _ in range(n_samples):
        # Random perturbation
        noise = np.random.uniform(-dr_scale, dr_scale, len(base_vec))
        # Don't let values go negative for most params
        new_vec = base_vec * (1 + noise)
        new_vec = np.clip(new_vec, 0.01, None)  # Minimum positive value
        
        # Handle special cases (damping can be negative in MuJoCo convention)
        new_vec[0:6] = base_vec[0:6] * (1 + noise[0:6])  # damping
        
        samples.append(AR4EnvParams.from_vector(new_vec))
    
    return samples


class SinePolicyWithTime:
    """Policy wrapper that tracks time for sine wave."""
    def __init__(self, freq: float = 0.5):
        self.freq = freq
        self.t = 0
        self.dt = 0.01
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = 0.3 * np.sin(2 * np.pi * self.freq * self.t + np.arange(6) * np.pi / 3)
        self.t += self.dt
        return action
    
    def reset(self):
        self.t = 0


def collect_trajectory_data(
    env: AR4MuJoCoEnv,
    policy,
    params_list: List[AR4EnvParams],
    target_traj: Dict[str, np.ndarray],
    max_steps: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect trajectory difference data for training.
    
    Args:
        env: AR4 environment
        policy: Policy to use
        params_list: List of parameter sets to try
        target_traj: Target ("real") trajectory
        max_steps: Maximum trajectory length
        
    Returns:
        param_vectors: [N, n_params] parameter vectors
        traj_diffs: [N, n_effects] trajectory differences
    """
    param_vectors = []
    traj_diffs = []
    
    for params in params_list:
        # Set environment parameters
        env.set_env_params(params)
        
        # Reset policy
        if hasattr(policy, 'reset'):
            policy.reset()
        
        # Rollout
        traj = rollout_trajectory(env, policy, max_steps=max_steps)
        
        # Compute difference from target
        diff = compute_trajectory_difference(
            traj, target_traj,
            components=['ee_positions']
        )
        
        param_vectors.append(params.to_vector())
        traj_diffs.append(diff)
    
    return np.array(param_vectors), np.array(traj_diffs)


def train_model_one_iter(
    model,
    param_data: np.ndarray,
    traj_diff_data: np.ndarray,
    n_epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train causal model for one iteration."""
    model.to(device)
    model.train()
    
    # Convert to tensors
    params_t = torch.FloatTensor(param_data).to(device)
    diffs_t = torch.FloatTensor(traj_diff_data).to(device)
    
    # Normalize inputs
    params_t = (params_t - params_t.mean(dim=0)) / (params_t.std(dim=0) + 1e-8)
    diffs_t = (diffs_t - diffs_t.mean(dim=0)) / (diffs_t.std(dim=0) + 1e-8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_samples = len(params_t)
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_params = params_t[idx]
            batch_diffs = diffs_t[idx]
            
            # Forward
            pred = model(batch_params)
            loss, info = model.loss_function(pred, batch_diffs)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
    
    return {'final_loss': epoch_loss}


def evaluate_causal_recovery(
    model,
    true_modified_indices: List[int],
    threshold: float = 0.3,
    n_params: int = 37
) -> Dict[str, float]:
    """
    Evaluate how well the model identifies the modified parameters.
    """
    model.eval()
    
    with torch.no_grad():
        soft_weights, _ = model.get_causal_weights(threshold)
        # Max across effects
        param_importance = soft_weights[:n_params].max(dim=-1).values.cpu().numpy()
    
    # Normalize to [0, 1]
    param_importance = param_importance / (param_importance.max() + 1e-8)
    
    # Predicted relevant params
    pred_relevant = set(np.where(param_importance > threshold)[0])
    true_relevant = set(true_modified_indices)
    
    # Metrics
    tp = len(pred_relevant & true_relevant)
    fp = len(pred_relevant - true_relevant)
    fn = len(true_relevant - pred_relevant)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_predicted': len(pred_relevant),
        'n_true': len(true_relevant),
    }


def run_experiment_mock(args):
    """
    Run experiment with mock data (when MuJoCo not available).
    """
    print("\n" + "="*60)
    print("Running MOCK experiment (MuJoCo not available)")
    print("="*60)
    
    n_params = 37
    n_effects = 3
    action_dim = 6
    
    # Create "true" causal matrix (sparse)
    np.random.seed(args.seed)
    true_causal = np.zeros((n_params, n_effects))
    modified_indices = [0, 1, 2, 6, 7, 25]  # Simulated modified params
    for idx in modified_indices:
        for k in range(n_effects):
            if np.random.rand() > 0.5:
                true_causal[idx, k] = np.random.randn()
    
    # Create synthetic data
    def create_synthetic_data(n_samples):
        env_params = np.random.randn(n_samples, n_params)
        traj_diff = np.zeros((n_samples, n_effects))
        for k in range(n_effects):
            for i in range(n_params):
                if abs(true_causal[i, k]) > 0.01:
                    traj_diff[:, k] += true_causal[i, k] * np.tanh(env_params[:, i])
        traj_diff += 0.1 * np.random.randn(n_samples, n_effects)
        return torch.FloatTensor(env_params), torch.FloatTensor(traj_diff)
    
    results = {}
    
    for method in args.methods:
        print(f"\n--- Method: {method} ---")
        
        model = create_model(
            method,
            input_dim=n_params,
            output_dim=n_effects,
            action_dim=0,
            hidden_dim=args.hidden_dim,
            sparse_weight=args.sparse_weight,
            num_phases=4
        )
        
        # Generate data
        n_samples = args.n_real_rollouts * args.n_dr_samples
        param_data, diff_data = create_synthetic_data(n_samples)
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(args.n_epochs):
            pred = model(param_data)
            loss, info = model.loss_function(pred, diff_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            soft_weights, _ = model.get_causal_weights(0.3)
            param_importance = soft_weights[:n_params].max(dim=-1).values.numpy()
        
        param_importance = param_importance / (param_importance.max() + 1e-8)
        pred_relevant = set(np.where(param_importance > 0.3)[0])
        true_relevant = set(modified_indices)
        
        tp = len(pred_relevant & true_relevant)
        fp = len(pred_relevant - true_relevant)
        fn = len(true_relevant - pred_relevant)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        results[method] = {
            'train_loss': final_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"  Loss: {final_loss:.4f}")
        print(f"  F1: {f1:.3f} (P={precision:.3f}, R={recall:.3f})")
    
    return results


def run_experiment(args):
    """Run the full sim-to-sim experiment."""
    set_seed(args.seed)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.exp_name or f"ar4_sim2sim_{timestamp}"
    logdir = Path(args.logdir) / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(logdir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "="*60)
    print(f"AR4 Sim-to-Sim Experiment")
    print(f"Methods: {args.methods}")
    print(f"Log dir: {logdir}")
    print("="*60)
    
    # Check if MuJoCo is available
    try:
        from ar4_compass.envs import AR4MuJoCoEnv
        env = AR4MuJoCoEnv()
        mujoco_available = True
        print("✓ MuJoCo environment loaded")
    except Exception as e:
        print(f"✗ MuJoCo not available: {e}")
        print("  Running with mock data...")
        return run_experiment_mock(args)
    
    # Create base and target parameters
    base_params = AR4EnvParams()
    target_params, modified_indices = create_target_params(base_params, seed=args.seed)
    
    print(f"\nModified parameters ({len(modified_indices)}):")
    param_names = AR4EnvParams.param_names()
    for idx in modified_indices:
        print(f"  [{idx}] {param_names[idx]}")
    
    # Create policy
    policy = SinePolicyWithTime(freq=0.3)
    
    # Collect target ("real") trajectory
    print("\nCollecting target trajectory...")
    env.set_env_params(target_params)
    policy.reset()
    target_traj = rollout_trajectory(env, policy, max_steps=args.max_steps)
    
    # Results storage
    results = {method: [] for method in args.methods}
    
    # Main loop
    for iteration in range(args.n_iters):
        print(f"\n--- Iteration {iteration + 1}/{args.n_iters} ---")
        
        # Domain randomization
        dr_params = domain_randomize_params(
            base_params, 
            args.n_real_rollouts * args.n_dr_samples,
            dr_scale=0.3
        )
        
        # Collect data
        print(f"  Collecting {len(dr_params)} rollouts...")
        param_data, diff_data = collect_trajectory_data(
            env, policy, dr_params, target_traj, args.max_steps
        )
        
        # Train each method
        for method in args.methods:
            print(f"  Training {method}...", end=' ')
            
            model = create_model(
                method,
                input_dim=AR4EnvParams.n_params(),
                output_dim=len(TRAJECTORY_EFFECT_NAMES),
                action_dim=0,
                hidden_dim=args.hidden_dim,
                sparse_weight=args.sparse_weight,
                num_phases=4
            )
            
            train_info = train_model_one_iter(
                model, param_data, diff_data,
                n_epochs=args.n_epochs,
                batch_size=32,
                lr=1e-3
            )
            
            eval_metrics = evaluate_causal_recovery(
                model, modified_indices, threshold=0.3
            )
            
            results[method].append({
                'iteration': iteration,
                'train_loss': train_info['final_loss'],
                **eval_metrics
            })
            
            print(f"F1={eval_metrics['f1']:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for method in args.methods:
        final = results[method][-1]
        print(f"\n{method.upper()}:")
        print(f"  Precision: {final['precision']:.3f}")
        print(f"  Recall:    {final['recall']:.3f}")
        print(f"  F1 Score:  {final['f1']:.3f}")
    
    # Save results
    with open(logdir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {logdir}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
