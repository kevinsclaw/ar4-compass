"""
Intervention-based Causal Discovery Utilities.

Direct causal effect estimation via simulation interventions.
This is the most principled approach as it computes true causal effects.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CausalEffect:
    """Estimated causal effect of a parameter on trajectory."""
    param_idx: int
    param_name: str
    effect_idx: int
    effect_name: str
    effect_magnitude: float
    effect_std: float
    n_samples: int


def estimate_causal_effect_single(
    env,
    param_idx: int,
    effect_idx: int,
    policy: Callable,
    delta: float = 0.1,
    n_samples: int = 50,
    param_getter: Optional[Callable] = None,
    param_setter: Optional[Callable] = None,
) -> CausalEffect:
    """
    Estimate causal effect of a single parameter on a single trajectory component.
    
    Uses the do-calculus formula:
        E[Y | do(X=x+δ)] - E[Y | do(X=x)]
    
    In simulation, we can perform perfect interventions by directly setting parameters.
    
    Args:
        env: MuJoCo environment with modifiable parameters
        param_idx: Index of the parameter to intervene on
        effect_idx: Index of the trajectory component to measure
        policy: Policy function for rollouts
        delta: Intervention magnitude (relative to current value)
        n_samples: Number of rollout samples
        param_getter: Function to get current parameter values
        param_setter: Function to set parameter values
        
    Returns:
        CausalEffect with estimated magnitude and uncertainty
    """
    if param_getter is None:
        param_getter = lambda: env.get_env_params()
    if param_setter is None:
        param_setter = lambda p: env.set_env_params(p)
    
    # Get baseline parameters
    baseline_params = param_getter()
    baseline_value = baseline_params[param_idx]
    
    # Collect baseline trajectories
    baseline_effects = []
    for _ in range(n_samples):
        traj = rollout_trajectory(env, policy)
        baseline_effects.append(compute_trajectory_metric(traj, effect_idx))
    
    # Intervene: set parameter to baseline + delta
    intervened_params = baseline_params.copy()
    intervened_params[param_idx] = baseline_value * (1 + delta)
    param_setter(intervened_params)
    
    # Collect intervened trajectories
    intervened_effects = []
    for _ in range(n_samples):
        traj = rollout_trajectory(env, policy)
        intervened_effects.append(compute_trajectory_metric(traj, effect_idx))
    
    # Restore baseline parameters
    param_setter(baseline_params)
    
    # Compute causal effect
    baseline_mean = np.mean(baseline_effects)
    intervened_mean = np.mean(intervened_effects)
    effect_magnitude = (intervened_mean - baseline_mean) / (baseline_value * delta)
    
    # Estimate uncertainty via bootstrap or standard error
    baseline_std = np.std(baseline_effects)
    intervened_std = np.std(intervened_effects)
    effect_std = np.sqrt(baseline_std**2 + intervened_std**2) / (baseline_value * delta * np.sqrt(n_samples))
    
    return CausalEffect(
        param_idx=param_idx,
        param_name=f"param_{param_idx}",
        effect_idx=effect_idx,
        effect_name=f"effect_{effect_idx}",
        effect_magnitude=effect_magnitude,
        effect_std=effect_std,
        n_samples=n_samples
    )


def estimate_all_causal_effects(
    env,
    policy: Callable,
    n_params: int,
    n_effects: int,
    delta: float = 0.1,
    n_samples: int = 20,
    param_names: Optional[List[str]] = None,
    effect_names: Optional[List[str]] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Estimate causal effects for all parameter-effect pairs.
    
    Args:
        env: Environment
        policy: Policy function
        n_params: Number of parameters
        n_effects: Number of trajectory components
        delta: Intervention magnitude
        n_samples: Samples per intervention
        param_names: Optional parameter names
        effect_names: Optional effect names
        verbose: Show progress bar
        
    Returns:
        Causal effect matrix [n_params, n_effects]
    """
    causal_matrix = np.zeros((n_params, n_effects))
    uncertainty_matrix = np.zeros((n_params, n_effects))
    
    iterator = range(n_params)
    if verbose:
        iterator = tqdm(iterator, desc="Estimating causal effects")
    
    for i in iterator:
        for k in range(n_effects):
            effect = estimate_causal_effect_single(
                env, i, k, policy, delta, n_samples
            )
            causal_matrix[i, k] = effect.effect_magnitude
            uncertainty_matrix[i, k] = effect.effect_std
    
    return causal_matrix, uncertainty_matrix


def intervention_based_screening(
    env,
    policy: Callable,
    n_params: int,
    n_effects: int,
    threshold: float = 0.1,
    n_samples: int = 10,
    verbose: bool = True
) -> List[int]:
    """
    Quick screening to identify potentially relevant parameters.
    
    Uses fewer samples for fast screening, then the selected parameters
    can be analyzed more carefully.
    
    Args:
        env: Environment
        policy: Policy function
        n_params: Number of parameters
        n_effects: Number of trajectory components
        threshold: Minimum effect magnitude to keep
        n_samples: Samples per intervention (lower for speed)
        verbose: Show progress
        
    Returns:
        List of parameter indices that pass screening
    """
    causal_matrix, _ = estimate_all_causal_effects(
        env, policy, n_params, n_effects,
        delta=0.2,  # Larger delta for faster detection
        n_samples=n_samples,
        verbose=verbose
    )
    
    # Take max effect across all trajectory components
    max_effects = np.abs(causal_matrix).max(axis=1)
    
    # Normalize
    max_effects = max_effects / (max_effects.max() + 1e-10)
    
    # Select parameters above threshold
    relevant_params = np.where(max_effects > threshold)[0].tolist()
    
    if verbose:
        print(f"Screening: {len(relevant_params)}/{n_params} parameters pass threshold {threshold}")
    
    return relevant_params


def rollout_trajectory(env, policy, max_steps: int = 100) -> Dict:
    """
    Rollout a trajectory using the given policy.
    
    Returns dict with states, actions, etc.
    """
    states = []
    actions = []
    
    obs = env.reset()
    for _ in range(max_steps):
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        
        states.append(obs)
        actions.append(action)
        obs = next_obs
        
        if done:
            break
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
    }


def compute_trajectory_metric(traj: Dict, effect_idx: int) -> float:
    """
    Compute a trajectory metric for the given effect index.
    
    This should be customized based on what trajectory components you want to track.
    Default: final state value
    """
    states = traj['states']
    
    if effect_idx < states.shape[1]:
        # Use cumulative deviation from initial state
        return np.sum(np.abs(states[:, effect_idx] - states[0, effect_idx]))
    else:
        # Default to trajectory length
        return len(states)


def visualize_causal_effects(
    causal_matrix: np.ndarray,
    param_names: Optional[List[str]] = None,
    effect_names: Optional[List[str]] = None,
    title: str = "Intervention-based Causal Effects"
):
    """Visualize causal effect matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n_params, n_effects = causal_matrix.shape
    
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    if effect_names is None:
        effect_names = [f"effect_{k}" for k in range(n_effects)]
    
    plt.figure(figsize=(max(8, n_effects), max(6, n_params * 0.3)))
    
    # Normalize for visualization
    normalized = causal_matrix / (np.abs(causal_matrix).max() + 1e-10)
    
    sns.heatmap(
        normalized,
        xticklabels=effect_names,
        yticklabels=param_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f"
    )
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()
