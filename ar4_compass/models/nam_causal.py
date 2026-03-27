"""
Neural Additive Models (NAM) for Causal Discovery.

NAM provides inherent interpretability by learning separate contribution 
functions for each parameter-effect pair.

Reference: Agarwal et al., "Neural Additive Models: Interpretable Machine 
Learning with Neural Nets", NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .base_causal import BaseCausalModel, MLP, CUDA


class FeatureNet(nn.Module):
    """
    Small neural network for a single feature's contribution.
    
    Learns f_i(x_i) where x_i is a single input feature.
    """
    
    def __init__(
        self, 
        hidden_dim: int = 64, 
        num_hidden: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = []
        in_features = 1  # Single feature input
        
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1] single feature values
        Returns:
            [B, 1] contribution to output
        """
        return self.network(x)


class NAMCausalModel(BaseCausalModel):
    """
    Neural Additive Model for causal discovery.
    
    Key idea: Model trajectory differences as additive contributions:
        d_k = Σ_i f_{i,k}(ε_i) + b_k
    
    where f_{i,k} is a small neural network for parameter i's effect on 
    trajectory component k.
    
    Advantages:
    1. Inherently interpretable - can visualize f_{i,k}(x) directly
    2. Easy to extract causal importance: ||f_{i,k}||
    3. No need for post-hoc analysis
    4. Can capture non-linear causal relationships
    5. Natural sparsity via L1 on feature network outputs
    
    Interpretability:
    - Plot f_{i,k}(x) to see how param i affects effect k
    - Flat curve = no causal effect
    - Steep curve = strong causal effect
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int = 0,
        feature_hidden_dim: int = 32,
        feature_num_hidden: int = 2,
        dropout: float = 0.0,
        sparse_weight: float = 0.01,
        sparse_norm: float = 1.0,
        output_penalty_weight: float = 0.001,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            sparse_weight=sparse_weight,
            sparse_norm=sparse_norm
        )
        
        self.feature_hidden_dim = feature_hidden_dim
        self.output_penalty_weight = output_penalty_weight
        
        # Create a feature network for each (parameter, effect) pair
        # f[i][k] : R -> R maps param_i to contribution to effect_k
        self.feature_nets = nn.ModuleList([
            nn.ModuleList([
                FeatureNet(feature_hidden_dim, feature_num_hidden, dropout)
                for _ in range(output_dim)
            ])
            for _ in range(self.total_input_dim)
        ])
        
        # Bias term for each output
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Optional: learnable importance weights (can be used for gating)
        self.importance_logits = nn.Parameter(
            torch.zeros(self.total_input_dim, output_dim)
        )
        self.use_importance_gating = kwargs.get('use_importance_gating', False)
        
        # Store feature contributions for interpretability
        self.register_buffer('last_contributions', None)
    
    def forward(
        self, 
        env_params: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass with additive decomposition.
        
        Args:
            env_params: [B, |E|] environment parameters
            actions: [B, A] optional action sequence
            threshold: Optional threshold for feature gating
            
        Returns:
            Predicted trajectory differences [B, K]
        """
        # Concatenate params and actions
        if actions is not None:
            inputs = torch.cat([env_params, actions], dim=-1)
        else:
            inputs = env_params
            
        batch_size = inputs.shape[0]
        
        # Compute contributions from each (input, output) pair
        contributions = torch.zeros(
            batch_size, self.total_input_dim, self.output_dim, 
            device=inputs.device
        )
        
        for i in range(self.total_input_dim):
            x_i = inputs[:, i:i+1]  # [B, 1]
            for k in range(self.output_dim):
                contributions[:, i, k] = self.feature_nets[i][k](x_i).squeeze(-1)
        
        # Store for interpretability
        self.last_contributions = contributions.detach()
        
        # Optional: apply importance gating
        if self.use_importance_gating:
            importance = torch.sigmoid(self.importance_logits)  # [|E|+A, K]
            if threshold is not None:
                importance = (importance > threshold).float()
            contributions = contributions * importance.unsqueeze(0)
        
        # Sum contributions: d_k = Σ_i f_{i,k}(x_i)
        output = contributions.sum(dim=1) + self.bias  # [B, K]
        
        return output
    
    def get_causal_weights(
        self, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get causal weights based on feature network outputs.
        
        Causal weight = average absolute contribution of each feature.
        """
        if self.last_contributions is None:
            # Return uniform weights
            soft = torch.ones(self.total_input_dim, self.output_dim) / self.total_input_dim
            return CUDA(soft), CUDA(soft)
        
        # Average absolute contribution across batch
        # [B, |E|+A, K] -> [|E|+A, K]
        soft_weights = self.last_contributions.abs().mean(dim=0)
        
        # Normalize to [0, 1]
        soft_weights = soft_weights / (soft_weights.max() + 1e-10)
        
        if threshold is not None:
            hard_weights = (soft_weights > threshold).float()
        else:
            hard_weights = soft_weights
        
        return soft_weights, CUDA(hard_weights)
    
    def loss_function(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with output penalty for sparsity.
        
        Output penalty: penalize non-zero contributions to encourage 
        sparse feature usage.
        """
        mse_loss = F.mse_loss(pred, target)
        
        # Output penalty: L1 on feature network outputs
        if self.last_contributions is not None:
            # Only penalize env params, not actions
            env_contributions = self.last_contributions[:, :self.input_dim, :]
            output_penalty = self.output_penalty_weight * torch.mean(
                torch.abs(env_contributions)
            )
        else:
            output_penalty = torch.tensor(0.0)
        
        # Importance sparsity (if using gating)
        if self.use_importance_gating:
            importance = torch.sigmoid(self.importance_logits[:self.input_dim, :])
            sparsity_loss = self.sparse_weight * torch.mean(
                importance ** self.sparse_norm
            )
        else:
            sparsity_loss = torch.tensor(0.0)
        
        total_loss = mse_loss + output_penalty + sparsity_loss
        
        info = {
            'mse': mse_loss.item(),
            'output_penalty': output_penalty.item() if isinstance(output_penalty, torch.Tensor) else output_penalty,
            'sparsity': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            'total': total_loss.item()
        }
        
        return total_loss, info
    
    def get_feature_function(
        self, 
        param_idx: int, 
        effect_idx: int,
        x_range: Tuple[float, float] = (-1, 1),
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned feature function f_{i,k}(x) for visualization.
        
        Args:
            param_idx: Index of the parameter
            effect_idx: Index of the trajectory difference component
            x_range: Range of input values to evaluate
            num_points: Number of points to sample
            
        Returns:
            (x_values, y_values) for plotting
        """
        x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(-1)
        x = CUDA(x)
        
        with torch.no_grad():
            y = self.feature_nets[param_idx][effect_idx](x).squeeze(-1)
        
        return x.cpu().numpy(), y.cpu().numpy()
    
    def visualize_feature_functions(
        self,
        param_names: Optional[List[str]] = None,
        effect_names: Optional[List[str]] = None,
        top_k: int = 10,
        x_range: Tuple[float, float] = (-1, 1)
    ):
        """
        Visualize top-k most important feature functions.
        """
        import matplotlib.pyplot as plt
        
        # Get causal weights to find top-k
        soft_weights, _ = self.get_causal_weights()
        weights_flat = soft_weights[:self.input_dim, :].flatten()
        top_k_idx = torch.topk(weights_flat, min(top_k, len(weights_flat))).indices
        
        if param_names is None:
            param_names = [f"param_{i}" for i in range(self.input_dim)]
        if effect_names is None:
            effect_names = [f"effect_{k}" for k in range(self.output_dim)]
        
        # Create subplots
        n_cols = min(5, top_k)
        n_rows = (top_k + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).flatten()
        
        for ax_idx, flat_idx in enumerate(top_k_idx):
            i = flat_idx // self.output_dim
            k = flat_idx % self.output_dim
            
            x, y = self.get_feature_function(i.item(), k.item(), x_range)
            
            axes[ax_idx].plot(x, y, 'b-', linewidth=2)
            axes[ax_idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[ax_idx].set_title(f"{param_names[i]} → {effect_names[k]}")
            axes[ax_idx].set_xlabel("Normalized param value")
            axes[ax_idx].set_ylabel("Contribution")
            axes[ax_idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_k_idx), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
