#!/usr/bin/env python3
"""
AR4 Real MuJoCo Sim-to-Sim Experiment.

This runs the COMPASS pipeline with actual MuJoCo simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import json
from datetime import datetime
from tqdm import tqdm

from ar4_compass.envs import (
    AR4MuJoCoEnv, AR4EnvConfig, AR4EnvParams,
    rollout_trajectory, compute_trajectory_difference
)
from ar4_compass.models import create_model
from ar4_compass.algorithms import COMPASSConfig, COMPASSPipeline


class SineWavePolicy:
    """Deterministic sine wave policy for reproducible trajectories."""
    def __init__(self, freq: float = 0.3, amplitude: float = 0.3):
        self.freq = freq
        self.amplitude = amplitude
        self.t = 0
        
    def __call__(self, obs):
        action = self.amplitude * np.sin(
            2 * np.pi * self.freq * self.t * 0.01 + np.arange(6) * np.pi / 3
        )
        self.t += 1
        return action
    
    def reset(self):
        self.t = 0


def create_target_params(base_params: AR4EnvParams, seed: int = 42):
    """Create target parameters with known modifications."""
    np.random.seed(seed)
    target = base_params.copy()
    modified = {}
    
    # Modify joint damping (main effect)
    scales = np.array([0.7, 1.3, 1.2, 0.9, 0.6, 0.8])
    target.joint_damping = base_params.joint_damping * scales
    for i, s in enumerate(scales):
        if abs(s - 1.0) > 0.1:
            modified[f'joint_{i+1}@damping'] = {'scale': s, 'idx': i}
    
    # Modify friction loss for joints 2 and 4
    target.joint_frictionloss[1] = base_params.joint_frictionloss[1] * 1.5
    target.joint_frictionloss[3] = base_params.joint_frictionloss[3] * 0.6
    modified['joint_2@frictionloss'] = {'scale': 1.5, 'idx': 7}
    modified['joint_4@frictionloss'] = {'scale': 0.6, 'idx': 9}
    
    return target, modified


def run_mujoco_sim2sim(n_iters=5, causal_method='attention', seed=42):
    """Run sim-to-sim experiment with real MuJoCo."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("\n" + "="*60)
    print("AR4 MuJoCo Sim-to-Sim Experiment")
    print("="*60)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f'experiments/mujoco_sim2sim_{causal_method}_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    config = AR4EnvConfig(
        mjcf_path="/home/ubuntu/ar4/ar4_mujoco_sim/mjcf/scene.xml",
        max_episode_steps=100
    )
    
    sim_env = AR4MuJoCoEnv(config)
    real_env = AR4MuJoCoEnv(config)  # "Real" is another sim with different params
    
    # Create target params
    base_params = AR4EnvParams()
    target_params, modifications = create_target_params(base_params, seed)
    
    # Apply target params to "real" env
    real_env.set_env_params(target_params)
    
    print(f"\nMethod: {causal_method}")
    print(f"Iterations: {n_iters}")
    print(f"\nModified parameters:")
    for name, info in modifications.items():
        print(f"  {name}: scale={info['scale']:.2f}")
    
    # Create policy
    policy = SineWavePolicy(freq=0.3, amplitude=0.3)
    
    # Collect real trajectories
    print(f"\nCollecting target trajectories...")
    n_real_rollouts = 5
    real_trajs = []
    for _ in range(n_real_rollouts):
        policy.reset()
        real_env.set_env_params(target_params)
        traj = rollout_trajectory(real_env, policy, max_steps=100)
        real_trajs.append(traj)
    
    print(f"  Collected {len(real_trajs)} trajectories")
    
    # Initialize sim params
    current_params = base_params.copy()
    param_history = [current_params.to_vector().copy()]
    traj_diff_history = []
    
    # Number of effects (EE position xyz)
    n_effects = 3
    n_params = AR4EnvParams.n_params()
    
    # Main COMPASS loop
    for iteration in range(n_iters):
        print(f"\n{'='*40}")
        print(f"Iteration {iteration + 1}/{n_iters}")
        print(f"{'='*40}")
        
        # Compute current trajectory difference
        sim_env.set_env_params(current_params)
        policy.reset()
        sim_traj = rollout_trajectory(sim_env, policy, max_steps=100)
        
        avg_diff = 0
        for real_traj in real_trajs:
            diff = compute_trajectory_difference(sim_traj, real_traj, ['ee_positions'])
            avg_diff += diff.sum()
        avg_diff /= len(real_trajs)
        traj_diff_history.append(avg_diff)
        print(f"Trajectory difference: {avg_diff:.4f}")
        
        # Domain randomization
        print(f"Domain randomization...")
        n_dr_samples = 64
        dr_range = 0.3 * (0.9 ** iteration)
        
        all_params = []
        all_diffs = []
        
        base_vec = current_params.to_vector()
        
        for real_traj in real_trajs:
            for _ in range(n_dr_samples // n_real_rollouts):
                # Perturb parameters
                noise = np.random.uniform(-dr_range, dr_range, n_params)
                new_vec = base_vec.copy()
                new_vec = base_vec * (1 + noise)
                
                # Handle negative damping
                new_vec[0:6] = base_vec[0:6] * (1 + noise[0:6])
                
                # Strict clipping for stable simulation
                for j in range(6):  # Damping
                    new_vec[j] = np.clip(new_vec[j], -20, -2)
                for j in range(6, 12):  # Friction loss
                    new_vec[j] = np.clip(new_vec[j], 0.01, 0.5)
                for j in range(12, 18):  # Armature
                    new_vec[j] = np.clip(new_vec[j], 0.01, 0.5)
                
                # Rollout
                dr_params = AR4EnvParams.from_vector(new_vec)
                sim_env.set_env_params(dr_params)
                policy.reset()
                dr_traj = rollout_trajectory(sim_env, policy, max_steps=100)
                
                # Compute diff
                diff = compute_trajectory_difference(dr_traj, real_traj, ['ee_positions'])
                
                all_params.append(new_vec)
                all_diffs.append(diff)
        
        all_params = np.array(all_params)
        all_diffs = np.array(all_diffs)
        print(f"  Collected {len(all_params)} samples")
        
        # Train causal model
        print(f"Training {causal_method} causal model...")
        model = create_model(
            causal_method,
            input_dim=n_params,
            output_dim=n_effects,
            action_dim=0,
            hidden_dim=128,
            sparse_weight=0.01 * (0.5 ** iteration),
            num_phases=4
        )
        
        # Prepare data
        params_t = torch.FloatTensor(all_params)
        diffs_t = torch.FloatTensor(all_diffs)
        
        # Normalize
        p_mean, p_std = params_t.mean(0), params_t.std(0) + 1e-8
        d_mean, d_std = diffs_t.mean(0), diffs_t.std(0) + 1e-8
        params_norm = (params_t - p_mean) / p_std
        diffs_norm = (diffs_t - d_mean) / d_std
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(500):
            pred = model(params_norm)
            loss, _ = model.loss_function(pred, diffs_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"  Final loss: {loss.item():.4f}")
        
        # Get causal weights
        model.eval()
        with torch.no_grad():
            soft_w, _ = model.get_causal_weights(0.3)
        importance = soft_w[:n_params].max(dim=-1).values.numpy()
        importance = importance / (importance.max() + 1e-8)
        
        relevant_idx = np.where(importance > 0.3)[0]
        print(f"Relevant parameters: {len(relevant_idx)}/{n_params}")
        
        # Optimize parameters
        print(f"Optimizing parameters...")
        current_vec = current_params.to_vector()
        opt_vec = (current_vec - p_mean.numpy()) / p_std.numpy()
        opt_t = torch.FloatTensor(opt_vec).unsqueeze(0).requires_grad_(True)
        init_t = opt_t.detach().clone()
        
        opt = torch.optim.Adam([opt_t], lr=0.01)
        
        for step in range(200):
            pred = model(opt_t)
            loss = pred.abs().mean() + 0.1 * ((opt_t - init_t)**2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([opt_t], 1.0)
            opt.step()
            with torch.no_grad():
                opt_t.clamp_(-3, 3)
        
        # Convert back
        new_vec = opt_t.detach().squeeze().numpy() * p_std.numpy() + p_mean.numpy()
        
        # Strict clipping to prevent simulation instability
        base_vec = current_params.to_vector()
        for i in range(n_params):
            if i < 6:  # Joint damping (negative values)
                # Damping typically in range [-20, -2]
                new_vec[i] = np.clip(new_vec[i], -20, -2)
            elif i < 12:  # Friction loss
                new_vec[i] = np.clip(new_vec[i], 0.01, 0.5)
            elif i < 18:  # Armature
                new_vec[i] = np.clip(new_vec[i], 0.01, 0.5)
            else:
                # Other params: stay within 2x of base
                if base_vec[i] > 0:
                    new_vec[i] = np.clip(new_vec[i], base_vec[i] * 0.5, base_vec[i] * 2.0)
                else:
                    new_vec[i] = np.clip(new_vec[i], base_vec[i] * 2.0, base_vec[i] * 0.5)
        
        current_params = AR4EnvParams.from_vector(new_vec)
        param_history.append(new_vec.copy())
        
        print(f"  Damping: {current_params.joint_damping}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("RESULTS")
    print("='*60")
    
    sim_env.set_env_params(current_params)
    policy.reset()
    final_traj = rollout_trajectory(sim_env, policy, max_steps=100)
    
    final_diff = 0
    for real_traj in real_trajs:
        diff = compute_trajectory_difference(final_traj, real_traj, ['ee_positions'])
        final_diff += diff.sum()
    final_diff /= len(real_trajs)
    
    print(f"\nTrajectory difference progression:")
    for i, d in enumerate(traj_diff_history):
        print(f"  Iter {i+1}: {d:.4f}")
    print(f"  Final: {final_diff:.4f}")
    
    improvement = (traj_diff_history[0] - final_diff) / traj_diff_history[0] * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Parameter recovery
    target_vec = target_params.to_vector()
    final_vec = current_params.to_vector()
    initial_vec = base_params.to_vector()
    
    print(f"\nParameter recovery:")
    param_names = AR4EnvParams.param_names()
    for name, info in modifications.items():
        idx = info['idx']
        init_err = abs(initial_vec[idx] - target_vec[idx])
        final_err = abs(final_vec[idx] - target_vec[idx])
        imp = (init_err - final_err) / init_err * 100
        print(f"  {name}:")
        print(f"    Target={target_vec[idx]:.4f}, Final={final_vec[idx]:.4f}")
        print(f"    Improvement: {imp:.1f}%")
    
    # Save results
    results = {
        'method': causal_method,
        'n_iters': n_iters,
        'traj_diff_history': [float(d) for d in traj_diff_history],
        'final_diff': float(final_diff),
        'improvement': float(improvement),
        'modifications': {k: {'scale': float(v['scale']), 'idx': int(v['idx'])} 
                         for k, v in modifications.items()},
    }
    
    with open(log_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {log_dir}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='attention',
                       choices=['compass', 'attention', 'nam'])
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_mujoco_sim2sim(
        n_iters=args.n_iters,
        causal_method=args.method,
        seed=args.seed
    )
