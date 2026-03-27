"""
Variational Causal Discovery Model.

Treats the causal graph as a latent variable and learns its posterior 
distribution, enabling uncertainty quantification in causal structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from .base_causal import BaseCausalModel, MLP, CUDA


class VariationalCausalModel(BaseCausalModel):
    """
    Variational approach to causal discovery.
    
    Key idea: Treat the causal graph G as a latent variable:
        p(d_τ | ε) = ∫ p(d_τ | G, ε) p(G) dG
    
    Learn the posterior q(G | data) and optimize via ELBO:
        ELBO = E_q[log p(d_τ | G, ε)] - KL(q(G) || p(G))
    
    Advantages:
    1. Uncertainty quantification in causal structure
    2. Principled sparsity via sparse prior
    3. Better generalization through Bayesian averaging
    4. Can encode domain knowledge through prior
    
    The posterior q(G) is parameterized as independent Bernoulli distributions:
        q(G_ij) = Bernoulli(σ(logit_ij))
    
    We use the concrete/Gumbel-Softmax relaxation for differentiability.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int = 0,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        prior_sparsity: float = 0.1,  # Prior probability of edge
        beta: float = 1.0,  # KL weight (β-VAE style)
        tau: float = 0.5,  # Gumbel-Softmax temperature
        hard: bool = False,  # Use hard samples during forward
        sparse_weight: float = 0.0,  # Additional L1 (usually 0 when using KL)
        sparse_norm: float = 1.0,
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
        self.prior_sparsity = prior_sparsity
        self.beta = beta
        self.tau = tau
        self.hard = hard
        
        # Prior: sparse Bernoulli
        # log p(G_ij = 1) = log(prior_sparsity)
        self.register_buffer(
            'prior_logit', 
            torch.tensor(math.log(prior_sparsity / (1 - prior_sparsity)))
        )
        
        # Posterior parameters: q(G_ij) = Bernoulli(σ(posterior_logits_ij))
        # Initialize to match prior
        self.posterior_logits = nn.Parameter(
            self.prior_logit.item() * torch.ones(self.total_input_dim, output_dim)
        )
        
        # Optional: make posterior depend on data (amortized inference)
        self.use_amortized = kwargs.get('use_amortized', False)
        if self.use_amortized:
            self.posterior_encoder = MLP(
                self.total_input_dim + output_dim,  # params + effects
                self.total_input_dim * output_dim,  # all edge logits
                hidden_dim,
                num_hidden
            )
        
        # Parameter encoder
        self.param_embedding = nn.Embedding(self.total_input_dim, embed_dim)
        self.param_encoder = MLP(embed_dim + 1, embed_dim, hidden_dim, num_hidden)
        
        # Effect decoder
        self.effect_embedding = nn.Embedding(output_dim, embed_dim)
        self.effect_decoder = MLP(embed_dim * 2, 1, hidden_dim, num_hidden)
        
        # Store samples for analysis
        self.register_buffer('last_graph_sample', None)
        self.register_buffer('last_graph_probs', None)
    
    def sample_graph(
        self, 
        batch_size: int = 1,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Sample causal graph from posterior using Gumbel-Softmax.
        
        Args:
            batch_size: Number of samples
            temperature: Gumbel-Softmax temperature (lower = more discrete)
            hard: Use straight-through estimator for hard samples
            
        Returns:
            Sampled graph [B, |E|+A, K] or [|E|+A, K] if batch_size=1
        """
        tau = temperature if temperature is not None else self.tau
        use_hard = hard if hard is not None else self.hard
        
        # Get posterior probabilities
        probs = torch.sigmoid(self.posterior_logits)  # [|E|+A, K]
        self.last_graph_probs = probs.detach()
        
        if batch_size == 1 and not self.training:
            # During inference, use MAP estimate (no sampling)
            return (probs > 0.5).float()
        
        # Gumbel-Softmax sampling
        # Create Bernoulli logits: [|E|+A, K, 2]
        logits = torch.stack([
            torch.log(1 - probs + 1e-10),  # log P(0)
            torch.log(probs + 1e-10)       # log P(1)
        ], dim=-1)
        
        # Expand for batch
        logits = logits.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Sample with Gumbel-Softmax
        samples = F.gumbel_softmax(logits, tau=tau, hard=use_hard, dim=-1)
        samples = samples[..., 1]  # Take P(1) channel: [B, |E|+A, K]
        
        self.last_graph_sample = samples.detach()
        
        return samples.squeeze(0) if batch_size == 1 else samples
    
    def forward(
        self, 
        env_params: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Forward pass with graph sampling.
        
        Args:
            env_params: [B, |E|] environment parameters
            actions: [B, A] optional action sequence
            threshold: If provided, use deterministic thresholding
            n_samples: Number of graph samples for Monte Carlo
            
        Returns:
            Predicted trajectory differences [B, K]
        """
        if actions is not None:
            inputs = torch.cat([env_params, actions], dim=-1)
        else:
            inputs = env_params
            
        batch_size = inputs.shape[0]
        
        # === Encode Parameters ===
        param_idx = CUDA(torch.arange(self.total_input_dim).long())
        param_emb = self.param_embedding(param_idx)  # [|E|+A, D]
        param_emb_batch = param_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        param_with_values = torch.cat([
            param_emb_batch,
            inputs.unsqueeze(-1)
        ], dim=-1)  # [B, |E|+A, D+1]
        
        param_encoded = self.param_encoder(param_with_values)  # [B, |E|+A, D]
        
        # === Sample or threshold causal graph ===
        if threshold is not None:
            # Deterministic: use threshold
            graph = (torch.sigmoid(self.posterior_logits) > threshold).float()
            graph = graph.unsqueeze(0).expand(batch_size, -1, -1)  # [B, |E|+A, K]
        else:
            # Stochastic: sample from posterior
            graph = self.sample_graph(batch_size)  # [B, |E|+A, K]
        
        # === Apply graph mask and decode ===
        effect_idx = CUDA(torch.arange(self.output_dim).long())
        effect_emb = self.effect_embedding(effect_idx)  # [K, D]
        
        outputs = []
        for k in range(self.output_dim):
            # Get masked parameter features for effect k
            mask_k = graph[:, :, k:k+1]  # [B, |E|+A, 1]
            masked_features = (param_encoded * mask_k).sum(dim=1)  # [B, D]
            
            # Concatenate with effect embedding
            effect_k_emb = effect_emb[k].unsqueeze(0).expand(batch_size, -1)
            decoder_input = torch.cat([masked_features, effect_k_emb], dim=-1)
            
            output_k = self.effect_decoder(decoder_input)  # [B, 1]
            outputs.append(output_k)
        
        output = torch.cat(outputs, dim=-1)  # [B, K]
        return output
    
    def get_causal_weights(
        self, 
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get posterior probabilities as causal weights.
        """
        soft_weights = torch.sigmoid(self.posterior_logits)
        
        if threshold is not None:
            hard_weights = (soft_weights > threshold).float()
        else:
            hard_weights = (soft_weights > 0.5).float()
        
        return soft_weights, CUDA(hard_weights)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        KL(q(G) || p(G)) = Σ_ij KL(Bernoulli(q_ij) || Bernoulli(p))
        """
        q = torch.sigmoid(self.posterior_logits)
        p = self.prior_sparsity
        
        # KL for Bernoulli: q*log(q/p) + (1-q)*log((1-q)/(1-p))
        kl = q * (torch.log(q + 1e-10) - math.log(p + 1e-10)) + \
             (1 - q) * (torch.log(1 - q + 1e-10) - math.log(1 - p + 1e-10))
        
        # Only count env params, not actions
        return kl[:self.input_dim, :].sum()
    
    def loss_function(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ELBO loss.
        
        ELBO = E_q[log p(y|x,G)] - β * KL(q(G) || p(G))
        """
        # Reconstruction loss (negative log likelihood)
        mse_loss = F.mse_loss(pred, target)
        
        # KL divergence
        kl_loss = self.kl_divergence()
        
        # ELBO (we minimize, so negate)
        elbo = mse_loss + self.beta * kl_loss
        
        # Optional additional sparsity
        if self.sparse_weight > 0:
            q = torch.sigmoid(self.posterior_logits[:self.input_dim, :])
            sparsity = self.sparse_weight * torch.mean(q ** self.sparse_norm)
        else:
            sparsity = torch.tensor(0.0)
        
        total_loss = elbo + sparsity
        
        info = {
            'mse': mse_loss.item(),
            'kl': kl_loss.item(),
            'elbo': elbo.item(),
            'sparsity': sparsity.item() if isinstance(sparsity, torch.Tensor) else sparsity,
            'total': total_loss.item(),
            'posterior_mean': torch.sigmoid(self.posterior_logits).mean().item()
        }
        
        return total_loss, info
    
    def get_uncertainty(self) -> torch.Tensor:
        """
        Get uncertainty in causal structure (entropy of posterior).
        
        High entropy = uncertain about edge existence
        Low entropy = confident about edge existence
        """
        q = torch.sigmoid(self.posterior_logits)
        entropy = -q * torch.log(q + 1e-10) - (1-q) * torch.log(1-q + 1e-10)
        return entropy
