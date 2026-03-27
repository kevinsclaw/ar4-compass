#!/usr/bin/env python3
"""
Run Full COMPASS Pipeline for AR4 Sim-to-Real Transfer.

This script runs the complete COMPASS algorithm:
1. Initialize simulation with default parameters
2. Create "target" (real) environment with modified parameters
3. Iterate:
   - Collect real trajectories
   - Causality-guided domain randomization
   - Train causal model
   - Optimize parameters via gradient descent
4. Evaluate parameter recovery

Usage:
    python scripts/run_compass_pipeline.py --method attention --n_iters 5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ar4_compass.algorithms import COMPASSConfig, COMPASSPipeline
from ar4_compass.envs import AR4EnvParams


def parse_args():
    parser = argparse.ArgumentParser(description='Run COMPASS Pipeline')
    
    # Method
    parser.add_argument('--method', type=str, default='attention',
                       choices=['compass', 'attention', 'nam', 'variational'],
                       help='Causal discovery method')
    
    # Pipeline settings
    parser.add_argument('--n_iters', type=int, default=5,
                       help='Number of COMPASS iterations')
    parser.add_argument('--n_real_rollouts', type=int, default=5,
                       help='Real rollouts per iteration')
    parser.add_argument('--n_dr_samples', type=int, default=32,
                       help='DR samples per real rollout')
    
    # Causal model
    parser.add_argument('--causal_epochs', type=int, default=1000,
                       help='Epochs for causal model training')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--sparse_weight', type=float, default=0.01,
                       help='Sparsity weight')
    
    # Parameter optimization
    parser.add_argument('--param_opt_steps', type=int, default=300,
                       help='Parameter optimization steps')
    parser.add_argument('--param_opt_lr', type=float, default=0.01,
                       help='Parameter optimization learning rate')
    
    # Logging
    parser.add_argument('--logdir', type=str, default='./experiments',
                       help='Log directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def create_target_params(seed: int = 0) -> tuple:
    """
    Create target parameters by modifying specific parameters.
    
    Returns:
        (target_params, modified_indices, modifications)
    """
    np.random.seed(seed)
    
    base = AR4EnvParams()
    target = base.copy()
    
    modifications = {}
    modified_indices = []
    
    # Modify joint damping (indices 0-5)
    damping_scale = np.random.uniform(0.5, 1.5, 6)
    target.joint_damping = base.joint_damping * damping_scale
    for i in range(6):
        if abs(damping_scale[i] - 1.0) > 0.1:
            modified_indices.append(i)
            modifications[f'joint_{i+1}@damping'] = {
                'base': base.joint_damping[i],
                'target': target.joint_damping[i],
                'scale': damping_scale[i]
            }
    
    # Modify some friction loss (indices 6-11)
    joints_to_modify = np.random.choice(6, 2, replace=False)
    for j in joints_to_modify:
        scale = np.random.uniform(0.5, 2.0)
        target.joint_frictionloss[j] = base.joint_frictionloss[j] * scale
        modified_indices.append(6 + j)
        modifications[f'joint_{j+1}@frictionloss'] = {
            'base': base.joint_frictionloss[j],
            'target': target.joint_frictionloss[j],
            'scale': scale
        }
    
    # Modify contact friction (index 25)
    friction_scale = np.random.uniform(0.7, 1.3)
    target.contact_friction_sliding = base.contact_friction_sliding * friction_scale
    modified_indices.append(25)
    modifications['contact@friction_sliding'] = {
        'base': base.contact_friction_sliding,
        'target': target.contact_friction_sliding,
        'scale': friction_scale
    }
    
    return target, modified_indices, modifications


def evaluate_param_recovery(
    final_params: AR4EnvParams,
    target_params: AR4EnvParams,
    initial_params: AR4EnvParams,
    modified_indices: list
) -> dict:
    """Evaluate how well parameters were recovered."""
    
    final_vec = final_params.to_vector()
    target_vec = target_params.to_vector()
    initial_vec = initial_params.to_vector()
    param_names = AR4EnvParams.param_names()
    
    results = {
        'per_param': {},
        'modified_params': {},
        'unmodified_params': {},
    }
    
    total_error_init = 0
    total_error_final = 0
    
    for idx in modified_indices:
        init_error = abs(initial_vec[idx] - target_vec[idx])
        final_error = abs(final_vec[idx] - target_vec[idx])
        
        improvement = (init_error - final_error) / (init_error + 1e-8)
        
        results['modified_params'][param_names[idx]] = {
            'initial': float(initial_vec[idx]),
            'target': float(target_vec[idx]),
            'final': float(final_vec[idx]),
            'init_error': float(init_error),
            'final_error': float(final_error),
            'improvement': float(improvement),
        }
        
        total_error_init += init_error
        total_error_final += final_error
    
    results['summary'] = {
        'total_init_error': total_error_init,
        'total_final_error': total_error_final,
        'total_improvement': (total_error_init - total_error_final) / (total_error_init + 1e-8),
        'n_modified': len(modified_indices),
    }
    
    return results


def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.exp_name or f"compass_{args.method}_{timestamp}"
    log_dir = Path(args.logdir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPASS Pipeline Experiment")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Iterations: {args.n_iters}")
    print(f"Log dir: {log_dir}")
    print("="*60 + "\n")
    
    # Create target parameters
    target_params, modified_indices, modifications = create_target_params(args.seed)
    initial_params = AR4EnvParams()
    
    print(f"Modified parameters ({len(modified_indices)}):")
    for name, mod in modifications.items():
        print(f"  {name}: {mod['base']:.4f} -> {mod['target']:.4f} (x{mod['scale']:.2f})")
    
    # Save config
    config_dict = {
        'args': vars(args),
        'modified_indices': [int(i) for i in modified_indices],
        'modifications': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                              for kk, vv in v.items()} 
                         for k, v in modifications.items()},
    }
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create COMPASS config
    config = COMPASSConfig(
        n_iterations=args.n_iters,
        n_real_rollouts=args.n_real_rollouts,
        n_dr_samples=args.n_dr_samples,
        causal_model_type=args.method,
        causal_hidden_dim=args.hidden_dim,
        causal_epochs=args.causal_epochs,
        sparse_weight=args.sparse_weight,
        param_opt_steps=args.param_opt_steps,
        param_opt_lr=args.param_opt_lr,
        log_dir=str(log_dir),
        verbose=True,
    )
    
    # Create and run pipeline
    pipeline = COMPASSPipeline(config)
    pipeline.set_initial_params(initial_params)
    pipeline.set_target_params(target_params)
    
    # Generate mock "real" trajectories
    # In a real experiment, these would come from actual robot rollouts
    mock_real_trajs = []
    target_vec = target_params.to_vector()
    
    for _ in range(config.n_real_rollouts):
        # Create trajectories that depend on target parameters
        T = config.max_trajectory_steps
        ee_pos = np.zeros((T, 3))
        
        # Simulate: EE position depends on joint damping
        for t in range(T):
            ee_pos[t] = 0.3 + 0.1 * np.sin(t * 0.1 + target_vec[0:3] * 0.1)
        
        mock_real_trajs.append({
            'observations': np.random.randn(T, 12),
            'actions': np.random.randn(T, 6),
            'ee_positions': ee_pos,
        })
    
    # Run pipeline
    results = pipeline.run(real_trajectories=mock_real_trajs)
    
    # Evaluate parameter recovery
    final_params = pipeline.state.env_params
    recovery_eval = evaluate_param_recovery(
        final_params, target_params, initial_params, modified_indices
    )
    
    results['recovery_evaluation'] = recovery_eval
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nIterations completed: {results['final_iteration'] + 1}")
    print(f"Relevant parameters found: {results['n_relevant_params']}")
    
    print(f"\nTrajectory differences per iteration:")
    for i, diff in enumerate(results['trajectory_diffs']):
        print(f"  Iter {i+1}: {diff:.4f}")
    
    print(f"\nParameter Recovery (modified params):")
    summary = recovery_eval['summary']
    print(f"  Initial total error: {summary['total_init_error']:.4f}")
    print(f"  Final total error:   {summary['total_final_error']:.4f}")
    print(f"  Improvement:         {summary['total_improvement']*100:.1f}%")
    
    print(f"\nPer-parameter results:")
    for name, data in recovery_eval['modified_params'].items():
        print(f"  {name}:")
        print(f"    Target: {data['target']:.4f}, Final: {data['final']:.4f}")
        print(f"    Improvement: {data['improvement']*100:.1f}%")
    
    # Save final results
    with open(log_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to {log_dir}")
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()
