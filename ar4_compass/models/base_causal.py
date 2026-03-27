"""
Base class for causal discovery models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import numpy as np


def CUDA(x):
    """Move tensor to CUDA if available. Deprecated - use tensor.to(device) instead."""
    # Don't auto-move to CUDA, let the caller handle device placement
    return x


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 64, 
        num_hidden: int = 2,
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = nn.Dropout(dropout)
        
        layers = []
        in_features = input_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(self.dropout)
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class BaseCausalModel(nn.Module, ABC):
    """
    Abstract base class for causal discovery models.
    
    All causal discovery methods should inherit from this class and implement:
    - forward(): Predict trajectory differences from environment parameters
    - get_causal_weights(): Return the learned causal structure
    - loss_function(): Compute training loss
    """
    
    def __init__(
        self,
        input_dim: int,          # Number of environment parameters |E|
        output_dim: int,         # Number of trajectory difference components K
        action_dim: int = 0,     # Action space dimension (optional)
        sparse_weight: float = 0.01,
        sparse_norm: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.total_input_dim = input_dim + action_dim
        self.sparse_weight = sparse_weight
        self.sparse_norm = sparse_norm
        
        # Normalizer for input preprocessing
        self.register_buffer('input_min', None)
        self.register_buffer('input_max', None)
        
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [-1, 1] range."""
        if self.input_min is None:
            self.input_min = x.min(dim=0, keepdim=True).values
            self.input_max = x.max(dim=0, keepdim=True).values
            # Avoid division by zero
            self.input_max = torch.where(
                torch.abs(self.input_max - self.input_min) < 1e-8,
                self.input_min + 1.0,
                self.input_max
            )
        return 2 * (x - self.input_min) / (self.input_max - self.input_min) - 1
    
    def denormalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize inputs back to original range."""
        return (x + 1) * (self.input_max - self.input_min) / 2 + self.input_min
    
    @abstractmethod
    def forward(
        self, 
        env_params: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Predict trajectory differences from environment parameters.
        
        Args:
            env_params: Environment parameters [B, |E|]
            actions: Optional action sequence [B, A]
            threshold: Optional threshold for causal graph sparsification
            
        Returns:
            Predicted trajectory differences [B, K]
        """
        pass
    
    @abstractmethod
    def get_causal_weights(self, threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the learned causal structure.
        
        Args:
            threshold: Optional threshold for binarization
            
        Returns:
            Tuple of (soft_weights, hard_weights) both of shape [|E|, K]
        """
        pass
    
    @abstractmethod
    def loss_function(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            pred: Predicted trajectory differences [B, K]
            target: Ground truth trajectory differences [B, K]
            
        Returns:
            Tuple of (loss, info_dict)
        """
        pass
    
    def get_relevant_params(self, threshold: float = 0.3) -> List[int]:
        """
        Get indices of parameters with causal relevance above threshold.
        
        Args:
            threshold: Minimum causal weight to be considered relevant
            
        Returns:
            List of parameter indices
        """
        soft_weights, _ = self.get_causal_weights(threshold)
        # Max over output dimensions
        max_weights = soft_weights.max(dim=1).values
        return (max_weights > threshold).nonzero(as_tuple=True)[0].tolist()
    
    def visualize_causal_graph(self, param_names: Optional[List[str]] = None, 
                               effect_names: Optional[List[str]] = None):
        """Generate visualization of the causal graph."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        soft_weights, _ = self.get_causal_weights()
        weights_np = soft_weights.cpu().detach().numpy()
        
        if param_names is None:
            param_names = [f"param_{i}" for i in range(self.input_dim)]
        if effect_names is None:
            effect_names = [f"effect_{i}" for i in range(self.output_dim)]
            
        plt.figure(figsize=(12, max(8, len(param_names) * 0.3)))
        sns.heatmap(
            weights_np, 
            xticklabels=effect_names,
            yticklabels=param_names,
            annot=True, 
            fmt=".2f",
            cmap="YlOrRd",
            square=True
        )
        plt.title("Causal Weights: Parameters → Trajectory Differences")
        plt.tight_layout()
        return plt.gcf()
