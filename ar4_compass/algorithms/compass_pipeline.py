"""
Complete COMPASS Pipeline for AR4 Sim-to-Real Transfer.

This module implements the full COMPASS algorithm:
1. Train policy in simulation
2. Collect "real" trajectories
3. Causality-guided domain randomization
4. Train causal model
5. Optimize environment parameters via gradient descent
6. Repeat until convergence

Reference: Huang et al., "What Went Wrong? Closing the Sim-to-Real Gap 
via Differentiable Causal Discovery", CoRL 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import copy
from tqdm import tqdm

from ar4_compass.models import create_model, BaseCausalModel
from ar4_compass.envs import AR4EnvParams


@dataclass
class COMPASSConfig:
    """Configuration for COMPASS pipeline."""
    
    # Main algorithm
    n_iterations: int = 10
    convergence_threshold: float = 0.1  # Stop if traj diff below this
    
    # Data collection
    n_real_rollouts: int = 10
    n_dr_samples: int = 64  # Domain randomization samples per real rollout
    max_trajectory_steps: int = 100
    
    # Causal model
    causal_model_type: str = "attention"  # compass, attention, nam, variational
    causal_hidden_dim: int = 128
    causal_epochs: int = 2000
    causal_batch_size: int = 64
    causal_lr: float = 1e-3
    sparse_weight: float = 0.01
    sparse_weight_decay: float = 0.5  # Decay factor after first iteration
    
    # Parameter optimization
    param_opt_lr: float = 0.01
    param_opt_steps: int = 500
    param_clip_range: Tuple[float, float] = (0.1, 10.0)  # Relative to initial
    
    # Domain randomization
    dr_range: float = 0.3  # Initial DR range (fraction of param value)
    dr_anneal: float = 0.9  # Anneal DR range each iteration
    causality_threshold: float = 0.3  # Threshold for pruning DR params
    
    # Logging
    log_dir: Optional[str] = None
    save_checkpoints: bool = True
    verbose: bool = True


@dataclass
class COMPASSState:
    """State of the COMPASS algorithm."""
    iteration: int = 0
    env_params: Optional[AR4EnvParams] = None
    target_params: Optional[AR4EnvParams] = None  # Ground truth (for sim2sim)
    
    # History
    trajectory_diffs: List[float] = field(default_factory=list)
    param_history: List[np.ndarray] = field(default_factory=list)
    causal_weights_history: List[np.ndarray] = field(default_factory=list)
    
    # Current causal model state
    relevant_params: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'iteration': self.iteration,
            'trajectory_diffs': self.trajectory_diffs,
            'param_history': [p.tolist() for p in self.param_history],
            'relevant_params': self.relevant_params,
        }


class COMPASSPipeline:
    """
    Complete COMPASS pipeline for sim-to-real transfer.
    
    Usage:
        pipeline = COMPASSPipeline(config)
        pipeline.set_environments(sim_env, real_env)
        pipeline.set_policy(policy)
        results = pipeline.run()
    """
    
    def __init__(self, config: COMPASSConfig):
        self.config = config
        self.state = COMPASSState()
        
        # Will be set by user
        self.sim_env = None
        self.real_env = None  # Or real data loader
        self.policy = None
        
        # Causal model (created during run)
        self.causal_model: Optional[BaseCausalModel] = None
        
        # Parameter info
        self.n_params = AR4EnvParams.n_params()
        self.param_names = AR4EnvParams.param_names()
        
        # Logging
        if config.log_dir:
            self.log_dir = Path(config.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = None
    
    def set_environments(self, sim_env, real_env=None):
        """
        Set simulation and real environments.
        
        For sim-to-sim experiments, real_env is another sim with different params.
        For sim-to-real, real_env can be None if using pre-collected data.
        """
        self.sim_env = sim_env
        self.real_env = real_env
    
    def set_policy(self, policy):
        """Set the policy for trajectory collection."""
        self.policy = policy
    
    def set_initial_params(self, params: AR4EnvParams):
        """Set initial simulation parameters."""
        self.state.env_params = params.copy()
        self.state.param_history.append(params.to_vector())
    
    def set_target_params(self, params: AR4EnvParams):
        """Set target (real) parameters for sim-to-sim experiments."""
        self.state.target_params = params
    
    def run(self, real_trajectories: Optional[List[Dict]] = None) -> Dict:
        """
        Run the complete COMPASS pipeline.
        
        Args:
            real_trajectories: Pre-collected real trajectories (optional)
            
        Returns:
            Results dictionary with final parameters and metrics
        """
        if self.state.env_params is None:
            self.state.env_params = AR4EnvParams()
        
        self._log(f"\n{'='*60}")
        self._log(f"COMPASS Pipeline Starting")
        self._log(f"Method: {self.config.causal_model_type}")
        self._log(f"Iterations: {self.config.n_iterations}")
        self._log(f"{'='*60}\n")
        
        # Initialize all parameters as relevant
        self.state.relevant_params = list(range(self.n_params))
        
        for iteration in range(self.config.n_iterations):
            self.state.iteration = iteration
            self._log(f"\n--- Iteration {iteration + 1}/{self.config.n_iterations} ---")
            
            # Step 1: Collect real trajectories (or use provided)
            if real_trajectories is not None:
                real_trajs = real_trajectories
            else:
                real_trajs = self._collect_real_trajectories()
            
            # Step 2: Check convergence
            avg_diff = self._compute_average_trajectory_diff(real_trajs)
            self.state.trajectory_diffs.append(avg_diff)
            self._log(f"Average trajectory difference: {avg_diff:.4f}")
            
            if avg_diff < self.config.convergence_threshold:
                self._log(f"Converged! Trajectory diff below threshold.")
                break
            
            # Step 3: Causality-guided domain randomization
            dr_params, dr_data = self._causality_guided_dr(real_trajs)
            
            # Step 4: Train causal model
            self.causal_model = self._train_causal_model(dr_data)
            
            # Step 5: Update relevant parameters based on causal graph
            self._update_relevant_params()
            
            # Step 6: Optimize environment parameters
            optimized_params = self._optimize_parameters(dr_data)
            
            # Update state
            self.state.env_params = optimized_params
            self.state.param_history.append(optimized_params.to_vector())
            
            # Apply to simulation environment
            if self.sim_env is not None:
                self.sim_env.set_env_params(optimized_params)
            
            # Decay sparsity weight after first iteration
            if iteration == 0:
                self.config.sparse_weight *= self.config.sparse_weight_decay
            
            # Save checkpoint
            if self.config.save_checkpoints and self.log_dir:
                self._save_checkpoint()
        
        # Final results
        results = self._compile_results()
        
        if self.log_dir:
            with open(self.log_dir / 'final_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        self._log(f"\n{'='*60}")
        self._log("COMPASS Pipeline Complete!")
        self._log(f"{'='*60}")
        
        return results
    
    def _collect_real_trajectories(self) -> List[Dict]:
        """Collect trajectories from real environment."""
        if self.real_env is None:
            raise ValueError("Real environment not set")
        
        self._log(f"Collecting {self.config.n_real_rollouts} real trajectories...")
        
        trajectories = []
        for i in range(self.config.n_real_rollouts):
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
            
            traj = self._rollout(self.real_env, self.policy)
            trajectories.append(traj)
        
        return trajectories
    
    def _compute_average_trajectory_diff(self, real_trajs: List[Dict]) -> float:
        """Compute average trajectory difference between sim and real."""
        if self.sim_env is None:
            return float('inf')
        
        total_diff = 0.0
        
        for real_traj in real_trajs:
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
            
            sim_traj = self._rollout(self.sim_env, self.policy)
            diff = self._trajectory_difference(sim_traj, real_traj)
            total_diff += diff.sum()
        
        return total_diff / len(real_trajs)
    
    def _causality_guided_dr(
        self, 
        real_trajs: List[Dict]
    ) -> Tuple[List[AR4EnvParams], Dict]:
        """
        Perform causality-guided domain randomization.
        
        Only randomizes parameters that have been identified as causally relevant.
        """
        self._log(f"Causality-guided DR with {len(self.state.relevant_params)} relevant params...")
        
        base_params = self.state.env_params
        base_vec = base_params.to_vector()
        
        # Calculate DR range (annealed)
        dr_range = self.config.dr_range * (self.config.dr_anneal ** self.state.iteration)
        
        all_param_vecs = []
        all_traj_diffs = []
        all_actions = []
        
        for real_traj in real_trajs:
            # Generate DR samples
            for _ in range(self.config.n_dr_samples):
                # Perturb only relevant parameters
                new_vec = base_vec.copy()
                for idx in self.state.relevant_params:
                    perturbation = np.random.uniform(-dr_range, dr_range)
                    new_vec[idx] = base_vec[idx] * (1 + perturbation)
                
                # Clip to valid range
                new_vec = np.clip(
                    new_vec,
                    base_vec * self.config.param_clip_range[0],
                    base_vec * self.config.param_clip_range[1]
                )
                
                # Special handling for damping (can be negative)
                new_vec[0:6] = base_vec[0:6] * (1 + np.random.uniform(-dr_range, dr_range, 6))
                
                dr_params = AR4EnvParams.from_vector(new_vec)
                
                # Rollout with DR params
                if self.sim_env is not None:
                    self.sim_env.set_env_params(dr_params)
                    if hasattr(self.policy, 'reset'):
                        self.policy.reset()
                    sim_traj = self._rollout(self.sim_env, self.policy)
                    
                    # Compute trajectory difference
                    traj_diff = self._trajectory_difference(sim_traj, real_traj)
                else:
                    # Mock trajectory difference
                    traj_diff = self._mock_trajectory_diff(new_vec, real_traj)
                
                all_param_vecs.append(new_vec)
                all_traj_diffs.append(traj_diff)
                
                if 'actions' in real_traj:
                    all_actions.append(real_traj['actions'].flatten()[:24])  # First 24
                else:
                    all_actions.append(np.zeros(24))
        
        dr_data = {
            'params': np.array(all_param_vecs),
            'traj_diffs': np.array(all_traj_diffs),
            'actions': np.array(all_actions),
        }
        
        return [], dr_data  # Return empty list for params (data already collected)
    
    def _train_causal_model(self, dr_data: Dict) -> BaseCausalModel:
        """Train the causal discovery model."""
        self._log(f"Training {self.config.causal_model_type} causal model...")
        
        n_effects = dr_data['traj_diffs'].shape[1]
        
        # Create model
        model = create_model(
            self.config.causal_model_type,
            input_dim=self.n_params,
            output_dim=n_effects,
            action_dim=0,  # We concatenate actions to params
            hidden_dim=self.config.causal_hidden_dim,
            sparse_weight=self.config.sparse_weight,
            num_phases=4  # For temporal model
        )
        
        # Prepare data
        params_t = torch.FloatTensor(dr_data['params'])
        diffs_t = torch.FloatTensor(dr_data['traj_diffs'])
        
        # Normalize
        params_mean, params_std = params_t.mean(0), params_t.std(0) + 1e-8
        diffs_mean, diffs_std = diffs_t.mean(0), diffs_t.std(0) + 1e-8
        
        params_norm = (params_t - params_mean) / params_std
        diffs_norm = (diffs_t - diffs_mean) / diffs_std
        
        # Store normalization for later
        model.register_buffer('params_mean', params_mean)
        model.register_buffer('params_std', params_std)
        model.register_buffer('diffs_mean', diffs_mean)
        model.register_buffer('diffs_std', diffs_std)
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.causal_lr)
        n_samples = len(params_norm)
        batch_size = self.config.causal_batch_size
        
        model.train()
        for epoch in range(self.config.causal_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                
                pred = model(params_norm[idx])
                loss, info = model.loss_function(pred, diffs_norm[idx])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if epoch % 500 == 0 and self.config.verbose:
                self._log(f"  Epoch {epoch}: loss = {epoch_loss/n_batches:.4f}")
        
        model.eval()
        return model
    
    def _update_relevant_params(self):
        """Update list of relevant parameters based on causal graph."""
        with torch.no_grad():
            soft_weights, _ = self.causal_model.get_causal_weights(
                self.config.causality_threshold
            )
        
        # Max importance across effects
        importance = soft_weights[:self.n_params].max(dim=-1).values.cpu().numpy()
        importance = importance / (importance.max() + 1e-8)
        
        # Select parameters above threshold
        self.state.relevant_params = np.where(
            importance > self.config.causality_threshold
        )[0].tolist()
        
        self._log(f"Relevant parameters: {len(self.state.relevant_params)}/{self.n_params}")
        
        if self.config.verbose and len(self.state.relevant_params) <= 15:
            for idx in self.state.relevant_params:
                self._log(f"  [{idx}] {self.param_names[idx]}: {importance[idx]:.3f}")
        
        # Store causal weights
        self.state.causal_weights_history.append(importance)
    
    def _optimize_parameters(self, dr_data: Dict) -> AR4EnvParams:
        """
        Optimize environment parameters using gradient descent through causal model.
        
        The key insight: we can backprop through the differentiable causal model
        to find parameters that minimize predicted trajectory difference.
        """
        self._log(f"Optimizing parameters ({self.config.param_opt_steps} steps)...")
        
        # Start from current parameters
        base_vec = self.state.env_params.to_vector()
        
        # Normalize using model's statistics
        params_mean = self.causal_model.params_mean.numpy()
        params_std = self.causal_model.params_std.numpy()
        
        # Create optimizable parameter tensor
        params_norm = (base_vec - params_mean) / params_std
        params_t = torch.FloatTensor(params_norm).unsqueeze(0).requires_grad_(True)
        
        # Store initial normalized params for constraint
        init_params_t = params_t.detach().clone()
        
        # Optimizer
        optimizer = torch.optim.Adam([params_t], lr=self.config.param_opt_lr)
        
        # Optimize to minimize predicted trajectory difference
        self.causal_model.eval()
        
        best_params = params_t.detach().clone()
        best_loss = float('inf')
        
        for step in range(self.config.param_opt_steps):
            # Predict trajectory difference
            pred_diff = self.causal_model(params_t)
            
            # Loss = mean predicted difference (minimize)
            loss = pred_diff.abs().mean()
            
            # Add constraint to not deviate too far from initial
            constraint = 0.1 * ((params_t - init_params_t) ** 2).mean()
            total_loss = loss + constraint
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([params_t], max_norm=1.0)
            
            optimizer.step()
            
            # Clamp to reasonable range (in normalized space, roughly [-3, 3])
            with torch.no_grad():
                params_t.clamp_(-3.0, 3.0)
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params_t.detach().clone()
            
            if step % 100 == 0 and self.config.verbose:
                self._log(f"  Step {step}: predicted diff = {loss.item():.4f}")
        
        # Convert back to original scale
        optimized_norm = best_params.squeeze().numpy()
        optimized_vec = optimized_norm * params_std + params_mean
        
        # Clip to valid range relative to base
        min_vals = base_vec * self.config.param_clip_range[0]
        max_vals = base_vec * self.config.param_clip_range[1]
        
        # Handle negative values (damping)
        for i in range(len(optimized_vec)):
            if base_vec[i] < 0:
                min_vals[i], max_vals[i] = max_vals[i], min_vals[i]
        
        optimized_vec = np.clip(optimized_vec, min_vals, max_vals)
        
        return AR4EnvParams.from_vector(optimized_vec)
    
    def _rollout(self, env, policy) -> Dict:
        """Perform a rollout in the given environment."""
        obs_list = []
        action_list = []
        ee_pos_list = []
        
        obs = env.reset()
        
        for _ in range(self.config.max_trajectory_steps):
            if callable(policy):
                action = policy(obs)
            elif hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = np.zeros(6)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            obs_list.append(obs)
            action_list.append(action)
            if 'ee_pos' in info:
                ee_pos_list.append(info['ee_pos'])
            
            if terminated or truncated:
                break
        
        return {
            'observations': np.array(obs_list),
            'actions': np.array(action_list),
            'ee_positions': np.array(ee_pos_list) if ee_pos_list else None,
        }
    
    def _trajectory_difference(self, sim_traj: Dict, real_traj: Dict) -> np.ndarray:
        """Compute trajectory difference vector."""
        # Use end-effector positions if available
        if sim_traj.get('ee_positions') is not None and real_traj.get('ee_positions') is not None:
            sim_ee = sim_traj['ee_positions']
            real_ee = real_traj['ee_positions']
            
            min_len = min(len(sim_ee), len(real_ee))
            diff = np.sum(np.abs(sim_ee[:min_len] - real_ee[:min_len]), axis=0)
            return diff
        
        # Fall back to observations
        sim_obs = sim_traj['observations']
        real_obs = real_traj['observations']
        
        min_len = min(len(sim_obs), len(real_obs))
        # Use first 3 dims (typically position-related)
        diff = np.sum(np.abs(sim_obs[:min_len, :3] - real_obs[:min_len, :3]), axis=0)
        return diff
    
    def _mock_trajectory_diff(self, param_vec: np.ndarray, real_traj: Dict) -> np.ndarray:
        """Generate mock trajectory difference for testing without MuJoCo."""
        # Simple model: diff depends on param deviation from "true" params
        if self.state.target_params is not None:
            target_vec = self.state.target_params.to_vector()
            diff = np.abs(param_vec - target_vec)
            # Aggregate to 3 effect dimensions
            return np.array([
                diff[0:12].sum(),   # Joint params
                diff[12:24].sum(),  # More joint params
                diff[24:].sum(),    # Other params
            ]) * 0.1 + np.random.randn(3) * 0.05
        else:
            return np.random.rand(3)
    
    def _compile_results(self) -> Dict:
        """Compile final results."""
        results = {
            'config': {
                'causal_model_type': self.config.causal_model_type,
                'n_iterations': self.config.n_iterations,
                'n_real_rollouts': self.config.n_real_rollouts,
                'n_dr_samples': self.config.n_dr_samples,
            },
            'final_iteration': self.state.iteration,
            'trajectory_diffs': self.state.trajectory_diffs,
            'final_relevant_params': self.state.relevant_params,
            'n_relevant_params': len(self.state.relevant_params),
        }
        
        # Compute parameter estimation error if target known
        if self.state.target_params is not None:
            target_vec = self.state.target_params.to_vector()
            final_vec = self.state.env_params.to_vector()
            
            # Mean absolute percentage error
            mape = np.mean(np.abs(final_vec - target_vec) / (np.abs(target_vec) + 1e-8))
            results['parameter_mape'] = float(mape)
            
            # Per-param error
            results['param_errors'] = {
                self.param_names[i]: float(np.abs(final_vec[i] - target_vec[i]))
                for i in self.state.relevant_params
            }
        
        return results
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        if self.log_dir is None:
            return
        
        checkpoint = {
            'iteration': self.state.iteration,
            'state': self.state.to_dict(),
            'env_params': self.state.env_params.to_vector().tolist(),
        }
        
        with open(self.log_dir / f'checkpoint_{self.state.iteration}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save causal model
        if self.causal_model is not None:
            torch.save(
                self.causal_model.state_dict(),
                self.log_dir / f'causal_model_{self.state.iteration}.pt'
            )
    
    def _log(self, msg: str):
        """Log message."""
        if self.config.verbose:
            print(msg)
        
        if self.log_dir:
            with open(self.log_dir / 'log.txt', 'a') as f:
                f.write(msg + '\n')


def run_sim2sim_compass(
    target_params: AR4EnvParams,
    initial_params: Optional[AR4EnvParams] = None,
    config: Optional[COMPASSConfig] = None,
    causal_method: str = "attention"
) -> Dict:
    """
    Convenience function to run sim-to-sim COMPASS experiment.
    
    Args:
        target_params: Target ("real") environment parameters
        initial_params: Initial simulation parameters (default: AR4EnvParams())
        config: COMPASS configuration
        causal_method: Causal discovery method
        
    Returns:
        Results dictionary
    """
    if config is None:
        config = COMPASSConfig()
    
    config.causal_model_type = causal_method
    
    if initial_params is None:
        initial_params = AR4EnvParams()
    
    # Create pipeline
    pipeline = COMPASSPipeline(config)
    pipeline.set_initial_params(initial_params)
    pipeline.set_target_params(target_params)
    
    # Create mock "real" trajectories based on target params
    # (In real experiments, these would come from actual rollouts)
    mock_real_trajs = []
    for _ in range(config.n_real_rollouts):
        mock_real_trajs.append({
            'observations': np.random.randn(config.max_trajectory_steps, 12),
            'actions': np.random.randn(config.max_trajectory_steps, 6),
            'ee_positions': np.random.randn(config.max_trajectory_steps, 3),
        })
    
    # Run pipeline
    results = pipeline.run(real_trajectories=mock_real_trajs)
    
    return results
