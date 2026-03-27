"""
Attention-based Causal Discovery Model.

Instead of a fixed binary causal graph, this model learns soft causal weights
via cross-attention between effect queries and cause keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List

from .base_causal import BaseCausalModel, MLP, CUDA


class AttentionCausalModel(BaseCausalModel):
    """
    Attention-based causal discovery model.
    
    Key idea: Use cross-attention where:
    - Queries (Q) = Effect embeddings (trajectory difference components)
    - Keys (K) = Cause embeddings (environment parameters)
    - Values (V) = Encoded parameter values
    
    The attention weights naturally represent causal strength:
    - Continuous values in [0, 1]
    - Learnable importance weighting
    - Interpretable as "how much does param_i affect effect_k"
    
    Advantages over Gumbel-Softmax:
    1. Continuous causal strength (not just binary)
    2. Better gradient flow (no discrete sampling)
    3. Natural sparsity via attention entropy regularization
    4. Multi-head attention can capture different causal aspects
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int = 0,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        sparse_weight: float = 0.01,
        sparse_norm: float = 1.0,
        entropy_weight: float = 0.001,  # Encourage sparse attention
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            sparse_weight=sparse_weight,
            sparse_norm=sparse_norm
        )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.entropy_weight = entropy_weight
        
        # Parameter (cause) encoder
        self.param_embedding = nn.Embedding(self.total_input_dim, embed_dim)
        self.param_encoder = MLP(
            embed_dim + 1,  # embedding + value
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout
        )
        
        # Effect (trajectory diff) embeddings - learnable queries
        self.effect_embedding = nn.Embedding(output_dim, embed_dim)
        
        # Multi-head cross attention: effects attend to parameters
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Optional: self-attention among effects to model correlations
        self.effect_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction head
        self.output_head = MLP(
            embed_dim,
            1,
            hidden_dim,
            num_hidden=1,
            dropout=dropout
        )
        
        # Layer norms
        self.ln_param = nn.LayerNorm(embed_dim)
        self.ln_effect = nn.LayerNorm(embed_dim)
        self.ln_cross = nn.LayerNorm(embed_dim)
        
        # Store attention weights for interpretability
        self.register_buffer('last_attn_weights', None)
    
    def forward(
        self, 
        env_params: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with attention-based causal discovery.
        
        Args:
            env_params: [B, |E|] environment parameters
            actions: [B, A] optional action sequence
            threshold: Optional threshold for attention sparsification
            return_attention: If True, also return attention weights
            
        Returns:
            Predicted trajectory differences [B, K]
            (optionally) attention weights [B, K, |E|+A]
        """
        # Concatenate params and actions
        if actions is not None:
            inputs = torch.cat([env_params, actions], dim=-1)
        else:
            inputs = env_params
            
        batch_size = inputs.shape[0]
        
        # === Encode Parameters (Causes) ===
        # Get parameter embeddings: [|E|+A, D]
        param_idx = CUDA(torch.arange(self.total_input_dim).long())
        param_emb = self.param_embedding(param_idx)
        
        # Expand for batch and concatenate values: [B, |E|+A, D+1]
        param_emb_batch = param_emb.unsqueeze(0).expand(batch_size, -1, -1)
        param_with_values = torch.cat([
            param_emb_batch, 
            inputs.unsqueeze(-1)
        ], dim=-1)
        
        # Encode: [B, |E|+A, D]
        param_encoded = self.param_encoder(param_with_values)
        param_encoded = self.ln_param(param_encoded)
        
        # === Effect Queries ===
        # Get effect embeddings: [K, D]
        effect_idx = CUDA(torch.arange(self.output_dim).long())
        effect_emb = self.effect_embedding(effect_idx)
        
        # Expand for batch: [B, K, D]
        effect_queries = effect_emb.unsqueeze(0).expand(batch_size, -1, -1)
        effect_queries = self.ln_effect(effect_queries)
        
        # === Cross Attention: Effects attend to Parameters ===
        # Q = effect queries [B, K, D]
        # K, V = encoded parameters [B, |E|+A, D]
        attended_features, attn_weights = self.cross_attention(
            query=effect_queries,
            key=param_encoded,
            value=param_encoded,
            need_weights=True,
            average_attn_weights=True  # Average over heads
        )
        # attended_features: [B, K, D]
        # attn_weights: [B, K, |E|+A]
        
        # Store attention weights
        self.last_attn_weights = attn_weights.detach()
        
        # Optional: Apply threshold for sparse attention
        if threshold is not None:
            attn_mask = (attn_weights > threshold).float()
            # Re-compute with masked attention (approximate)
            attended_features = attended_features * attn_mask.unsqueeze(-1).mean(dim=-1, keepdim=True)
        
        # Residual connection
        attended_features = self.ln_cross(attended_features + effect_queries)
        
        # === Optional: Self-attention among effects ===
        # effect_features, _ = self.effect_self_attention(
        #     attended_features, attended_features, attended_features
        # )
        # attended_features = attended_features + effect_features
        
        # === Predict trajectory differences ===
        output = self.output_head(attended_features).squeeze(-1)  # [B, K]
        
        if return_attention:
            return output, attn_weights
        return output
    
    def get_causal_weights(
        self, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get causal weights from stored attention.
        
        Returns:
            (soft_weights, hard_weights) both [|E|+A, K]
            Note: Transposed to match COMPASS format
        """
        if self.last_attn_weights is None:
            # Return uniform weights if no forward pass yet
            soft = torch.ones(self.total_input_dim, self.output_dim) / self.total_input_dim
            hard = torch.ones_like(soft)
            return CUDA(soft), CUDA(hard)
        
        # Average over batch: [K, |E|+A] -> transpose to [|E|+A, K]
        soft_weights = self.last_attn_weights.mean(dim=0).T
        
        if threshold is not None:
            hard_weights = (soft_weights > threshold).float()
        else:
            # Use mean as threshold
            hard_weights = (soft_weights > soft_weights.mean()).float()
        
        return soft_weights, CUDA(hard_weights)
    
    def loss_function(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with attention entropy regularization.
        
        The entropy term encourages sparse attention:
        - Low entropy = focused attention = sparse causal structure
        - High entropy = uniform attention = dense causal structure
        """
        mse_loss = F.mse_loss(pred, target)
        
        # Attention entropy regularization (encourage sparsity)
        if self.last_attn_weights is not None:
            # Only regularize env param attention, not actions
            env_attn = self.last_attn_weights[:, :, :self.input_dim]
            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(
                env_attn * torch.log(env_attn + 1e-10), 
                dim=-1
            ).mean()
            # We want LOW entropy (sparse), so minimize negative entropy
            # Actually we want to encourage some sparsity, so penalize high entropy
            entropy_loss = self.entropy_weight * entropy
        else:
            entropy_loss = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
        
        # L1 sparsity on attention weights
        if self.last_attn_weights is not None:
            env_attn = self.last_attn_weights[:, :, :self.input_dim]
            sparsity_loss = self.sparse_weight * torch.mean(
                env_attn ** self.sparse_norm
            )
        else:
            sparsity_loss = torch.tensor(0.0)
        
        total_loss = mse_loss + entropy_loss + sparsity_loss
        
        info = {
            'mse': mse_loss.item(),
            'entropy': entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            'sparsity': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            'total': total_loss.item()
        }
        
        return total_loss, info
    
    def get_top_k_causes(self, k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Get top-k causal relationships.
        
        Returns:
            List of (param_idx, effect_idx, weight) tuples
        """
        soft_weights, _ = self.get_causal_weights()
        weights_flat = soft_weights[:self.input_dim, :].flatten()
        
        top_k_idx = torch.topk(weights_flat, k).indices
        
        results = []
        for idx in top_k_idx:
            param_idx = idx // self.output_dim
            effect_idx = idx % self.output_dim
            weight = weights_flat[idx].item()
            results.append((param_idx.item(), effect_idx.item(), weight))
        
        return results
