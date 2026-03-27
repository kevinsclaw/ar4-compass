"""
Original COMPASS implementation with Gumbel-Softmax causal graph.

Reference: Huang et al., "What Went Wrong? Closing the Sim-to-Real Gap via 
Differentiable Causal Discovery", CoRL 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base_causal import BaseCausalModel, MLP, CUDA


def temp_sigmoid(x: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Temperature-scaled sigmoid."""
    return torch.sigmoid(x / temp)


class COMPASSOriginal(BaseCausalModel):
    """
    Original COMPASS model with Gumbel-Softmax binary causal graph.
    
    Architecture:
    1. Independent encoder for each input dimension
    2. Learnable binary causal graph (Gumbel-Softmax)
    3. Masked feature aggregation
    4. Shared decoder for each output dimension
    
    The causal graph G ∈ {0,1}^{|E|×K} indicates which environment 
    parameters causally influence which trajectory difference components.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int = 0,
        emb_dim: int = 32,
        hidden_dim: int = 256,
        causal_dim: int = 32,
        num_hidden: int = 2,
        sparse_weight: float = 0.01,
        sparse_norm: float = 1.0,
        tau: float = 1.0,
        use_full: bool = False,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            sparse_weight=sparse_weight,
            sparse_norm=sparse_norm
        )
        
        self.emb_dim = emb_dim
        self.causal_dim = causal_dim
        self.tau = tau
        self.use_full = use_full
        
        # Encoder: processes each input dimension independently
        # Input: [emb_dim + 1] (embedding + value)
        # Output: [causal_dim]
        self.encoder = MLP(
            emb_dim + 1, 
            causal_dim, 
            hidden_dim, 
            num_hidden
        )
        self.encoder_idx_emb = nn.Embedding(self.total_input_dim, emb_dim)
        
        # Decoder: predicts each output dimension
        # Input: [causal_dim + emb_dim]
        # Output: [1]
        self.decoder = MLP(
            causal_dim + emb_dim, 
            1, 
            hidden_dim, 
            num_hidden
        )
        self.decoder_idx_emb = nn.Embedding(output_dim, emb_dim)
        
        # Learnable causal graph parameters (logits)
        # Initialized to 3 to start with high probability of connection
        self.mask_logits = nn.Parameter(
            3 * torch.ones(self.total_input_dim, output_dim), 
            requires_grad=True
        )
        
        # Current mask (updated during forward pass)
        self.register_buffer('mask', torch.ones(self.total_input_dim, output_dim))
    
    def forward(
        self, 
        env_params: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            env_params: [B, |E|] environment parameters
            actions: [B, A] optional action sequence
            threshold: If provided, use hard thresholding instead of Gumbel-Softmax
            
        Returns:
            Predicted trajectory differences [B, K]
        """
        # Concatenate params and actions if provided
        if actions is not None:
            inputs = torch.cat([env_params, actions], dim=-1)
        else:
            inputs = env_params
            
        assert inputs.shape[-1] == self.total_input_dim, \
            f"Expected input dim {self.total_input_dim}, got {inputs.shape[-1]}"
        
        batch_size = inputs.shape[0]
        
        # Add value dimension: [B, S+A] -> [B, S+A, 1]
        inputs = inputs.unsqueeze(-1)
        
        # Get encoder embeddings: [S+A, E]
        device = inputs.device
        encoder_idx = self.encoder_idx_emb(
            torch.arange(0, self.total_input_dim, device=device).long()
        )
        # Repeat for batch: [B, S+A, E]
        batch_encoder_idx = encoder_idx.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate value with embedding: [B, S+A, E+1]
        inputs_feature = torch.cat([inputs, batch_encoder_idx], dim=-1)
        
        # Encode each input dimension: [B, S+A, C]
        latent_feature = self.encoder(inputs_feature)
        
        # Get causal mask
        if not self.use_full:
            _, self.mask = self.get_causal_weights(threshold)
        else:
            self.mask = torch.ones_like(self.mask_logits)
        
        # Apply mask: [B, S+A, C] × [S+A, K] -> [B, K, C]
        masked_feature = torch.einsum('bnc, nk -> bkc', latent_feature, self.mask)
        
        # Get decoder embeddings: [K, E]
        decoder_idx = self.decoder_idx_emb(
            torch.arange(0, self.output_dim, device=device).long()
        )
        # Repeat for batch: [B, K, E]
        batch_decoder_idx = decoder_idx.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate masked features with decoder embeddings: [B, K, C+E]
        decoder_input = torch.cat([masked_feature, batch_decoder_idx], dim=-1)
        
        # Decode: [B, K, C+E] -> [B, K]
        output = self.decoder(decoder_input).squeeze(-1)
        
        return output
    
    def get_causal_weights(
        self, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get soft and hard causal weights.
        
        Args:
            threshold: If provided, use hard thresholding; else use Gumbel-Softmax
            
        Returns:
            (soft_weights, hard_weights) both [|E|+A, K]
        """
        # Soft weights via sigmoid
        soft_weights = temp_sigmoid(self.mask_logits, temp=1.0)
        
        if threshold is not None:
            # Hard thresholding
            hard_weights = (soft_weights > threshold).float()
        else:
            # Gumbel-Softmax sampling
            # Build Bernoulli: [S+A, K, 2] where dim 0 is P(0) and dim 1 is P(1)
            mask_bernoulli = torch.stack([
                (1 - soft_weights).log(),
                soft_weights.log()
            ], dim=-1)
            
            # Sample with Gumbel-Softmax
            hard_weights = F.gumbel_softmax(
                mask_bernoulli, 
                tau=self.tau, 
                hard=True, 
                dim=-1
            )
            hard_weights = hard_weights[..., 1]  # Take P(1) channel
        
        return soft_weights, CUDA(hard_weights)
    
    def loss_function(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MSE loss with sparsity regularization.
        
        Only applies sparsity to environment parameters, not actions.
        """
        mse_loss = F.mse_loss(pred, target)
        
        # Sparsity only on env params (not actions)
        env_mask_probs = torch.sigmoid(self.mask_logits[:self.input_dim, :])
        sparsity_loss = self.sparse_weight * torch.mean(
            env_mask_probs ** self.sparse_norm
        )
        
        total_loss = mse_loss + sparsity_loss
        
        info = {
            'mse': mse_loss.item(),
            'sparsity': sparsity_loss.item(),
            'total': total_loss.item(),
            'avg_mask_prob': env_mask_probs.mean().item()
        }
        
        return total_loss, info
    
    def get_param_causal_weights(self) -> torch.Tensor:
        """Get causal weights for environment parameters only (excluding actions)."""
        soft_weights, _ = self.get_causal_weights()
        return soft_weights[:self.input_dim, :]
