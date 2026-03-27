"""
Causal Discovery Models for Sim-to-Real Transfer.

This module provides multiple approaches to causal discovery:

1. COMPASSOriginal: Original COMPASS with Gumbel-Softmax binary graph
2. AttentionCausalModel: Attention-based with continuous causal weights  
3. NAMCausalModel: Neural Additive Models for interpretable causality
4. VariationalCausalModel: Bayesian approach with uncertainty quantification
5. TemporalCausalModel: Phase-aware temporal causality
"""

from .base_causal import BaseCausalModel, MLP, CUDA
from .compass_original import COMPASSOriginal
from .attention_causal import AttentionCausalModel
from .nam_causal import NAMCausalModel
from .variational_causal import VariationalCausalModel
from .temporal_causal import TemporalCausalModel, TaskPhase


# Model registry for easy instantiation
MODEL_REGISTRY = {
    'compass': COMPASSOriginal,
    'attention': AttentionCausalModel,
    'nam': NAMCausalModel,
    'variational': VariationalCausalModel,
    'temporal': TemporalCausalModel,
}


def create_model(model_type: str, **kwargs) -> BaseCausalModel:
    """
    Factory function to create causal discovery models.
    
    Args:
        model_type: One of 'compass', 'attention', 'nam', 'variational', 'temporal'
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated causal model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = [
    'BaseCausalModel',
    'MLP',
    'CUDA',
    'COMPASSOriginal',
    'AttentionCausalModel', 
    'NAMCausalModel',
    'VariationalCausalModel',
    'TemporalCausalModel',
    'TaskPhase',
    'MODEL_REGISTRY',
    'create_model',
]
