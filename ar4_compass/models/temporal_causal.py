"""
Temporal Phase-Aware Causal Discovery Model.

Different task phases (approach, grasp, lift, place) may have different 
causal structures - this model learns phase-specific causal graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from enum import Enum

from .base_causal import BaseCausalModel, MLP, CUDA
from .compass_original import COMPASSOriginal


class TaskPhase(Enum):
    """Standard manipulation task phases."""
    APPROACH = 0
    GRASP = 1
    LIFT = 2
    TRANSPORT = 3
    PLACE = 4
    RETREAT = 5


class PhaseClassifier(nn.Module):
    """
    Classify trajectory segments into task phases.
    
    Can be trained supervised (with phase labels) or learned end-to-end.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_phases: int = 4,
        hidden_dim: int = 64,
        use_lstm: bool = True
    ):
        super().__init__()
        self.num_phases = num_phases
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.encoder = nn.LSTM(
                state_dim, 
                hidden_dim, 
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            self.classifier = nn.Linear(hidden_dim * 2, num_phases)
        else:
            self.encoder = MLP(state_dim, hidden_dim, hidden_dim, num_hidden=2)
            self.classifier = nn.Linear(hidden_dim, num_phases)
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Classify each timestep into a phase.
        
        Args:
            trajectory: [B, T, state_dim] trajectory states
            
        Returns:
            phase_logits: [B, T, num_phases]
        """
        if self.use_lstm:
            encoded, _ = self.encoder(trajectory)
        else:
            encoded = self.encoder(trajectory)
        
        logits = self.classifier(encoded)
        return logits
    
    def get_phases(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Get hard phase assignments."""
        logits = self.forward(trajectory)
        return logits.argmax(dim=-1)


class TemporalCausalModel(BaseCausalModel):
    """
    Phase-aware temporal causal discovery.
    
    Key idea: Different task phases have different causal structures:
    
    - APPROACH: Joint damping and inertia matter most (free space motion)
    - GRASP: Friction and gripper stiffness matter most (contact)
    - LIFT: Gravity compensation and payload inertia matter most
    - PLACE: Contact parameters matter most
    
    This model learns separate causal graphs for each phase and combines
    them based on the current phase classification.
    
    Advantages:
    1. Captures time-varying causality
    2. More accurate for multi-phase tasks
    3. Better interpretability (know which params matter when)
    4. Can focus parameter tuning on specific phases
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int = 0,
        num_phases: int = 4,
        state_dim: int = 12,  # For phase classification
        embed_dim: int = 32,
        hidden_dim: int = 256,
        causal_dim: int = 32,
        num_hidden: int = 2,
        sparse_weight: float = 0.01,
        sparse_norm: float = 1.0,
        phase_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            sparse_weight=sparse_weight,
            sparse_norm=sparse_norm
        )
        
        self.num_phases = num_phases
        self.state_dim = state_dim
        self.phase_names = phase_names or [f"phase_{i}" for i in range(num_phases)]
        
        # Phase classifier
        self.phase_classifier = PhaseClassifier(
            state_dim=state_dim,
            num_phases=num_phases,
            hidden_dim=hidden_dim // 2
        )
        
        # Separate causal model for each phase
        self.phase_models = nn.ModuleList([
            COMPASSOriginal(
                input_dim=input_dim,
                output_dim=output_dim,
                action_dim=action_dim,
                emb_dim=embed_dim,
                hidden_dim=hidden_dim,
                causal_dim=causal_dim,
                num_hidden=num_hidden,
                sparse_weight=sparse_weight,
                sparse_norm=sparse_norm
            )
            for _ in range(num_phases)
        ])
        
        # Optional: shared encoder across phases
        self.share_encoder = kwargs.get('share_encoder', False)
        if self.share_encoder:
            shared_encoder = self.phase_models[0].encoder
            for model in self.phase_models[1:]:
                model.encoder = shared_encoder
        
        # Store phase-wise predictions
        self.register_buffer('last_phase_weights', None)
        self.register_buffer('last_phases', None)
    
    def forward(
        self, 
        env_params: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        phase_labels: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass with phase-aware causal models.
        
        Args:
            env_params: [B, |E|] environment parameters
            actions: [B, A] optional action sequence  
            trajectory: [B, T, state_dim] trajectory for phase classification
            phase_labels: [B] or [B, T] ground truth phases (optional)
            threshold: Threshold for causal graph
            
        Returns:
            Predicted trajectory differences [B, K]
        """
        batch_size = env_params.shape[0]
        
        # === Determine phases ===
        if phase_labels is not None:
            # Use provided phase labels
            if phase_labels.dim() == 1:
                phases = phase_labels  # [B]
            else:
                # If per-timestep labels, use mode (most common)
                phases = phase_labels.mode(dim=1).values  # [B]
            phase_probs = F.one_hot(phases, self.num_phases).float()  # [B, P]
        elif trajectory is not None:
            # Classify phases from trajectory
            phase_logits = self.phase_classifier(trajectory)  # [B, T, P]
            # Average over time
            phase_probs = F.softmax(phase_logits.mean(dim=1), dim=-1)  # [B, P]
            phases = phase_probs.argmax(dim=-1)
        else:
            # No phase info - use uniform mixture
            phase_probs = torch.ones(batch_size, self.num_phases, device=env_params.device)
            phase_probs = phase_probs / self.num_phases
            phases = torch.zeros(batch_size, dtype=torch.long, device=env_params.device)
        
        self.last_phases = phases.detach()
        self.last_phase_weights = phase_probs.detach()
        
        # === Get predictions from each phase model ===
        phase_outputs = []
        for phase_idx, model in enumerate(self.phase_models):
            output = model(env_params, actions, threshold)  # [B, K]
            phase_outputs.append(output)
        
        phase_outputs = torch.stack(phase_outputs, dim=1)  # [B, P, K]
        
        # === Combine predictions based on phase probabilities ===
        # Weighted combination: Σ_p P(phase=p) * output_p
        output = torch.einsum('bp, bpk -> bk', phase_probs, phase_outputs)
        
        return output
    
    def get_causal_weights(
        self, 
        threshold: Optional[float] = None,
        phase: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get causal weights, optionally for a specific phase.
        
        Args:
            threshold: Threshold for binarization
            phase: If provided, return weights for this phase only
            
        Returns:
            (soft_weights, hard_weights) both [|E|+A, K] or [P, |E|+A, K]
        """
        if phase is not None:
            return self.phase_models[phase].get_causal_weights(threshold)
        
        # Return all phase weights
        soft_list = []
        hard_list = []
        for model in self.phase_models:
            soft, hard = model.get_causal_weights(threshold)
            soft_list.append(soft)
            hard_list.append(hard)
        
        soft_weights = torch.stack(soft_list, dim=0)  # [P, |E|+A, K]
        hard_weights = torch.stack(hard_list, dim=0)
        
        return soft_weights, hard_weights
    
    def get_phase_specific_params(
        self, 
        phase: int, 
        threshold: float = 0.3
    ) -> List[int]:
        """Get parameter indices relevant for a specific phase."""
        soft, _ = self.phase_models[phase].get_causal_weights(threshold)
        max_weights = soft[:self.input_dim].max(dim=1).values
        return (max_weights > threshold).nonzero(as_tuple=True)[0].tolist()
    
    def loss_function(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        phase_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with phase-specific sparsity.
        """
        mse_loss = F.mse_loss(pred, target)
        
        # Sparsity for each phase model
        total_sparsity = torch.tensor(0.0, device=pred.device)
        phase_sparsities = []
        
        for idx, model in enumerate(self.phase_models):
            _, info = model.loss_function(pred, target)
            phase_sparsities.append(info['sparsity'])
            total_sparsity = total_sparsity + info['sparsity']
        
        total_sparsity = total_sparsity / self.num_phases
        
        # Optional: phase classification loss if labels provided
        if phase_labels is not None and self.last_phase_weights is not None:
            phase_ce = F.cross_entropy(
                self.last_phase_weights.log(), 
                phase_labels
            )
        else:
            phase_ce = torch.tensor(0.0)
        
        total_loss = mse_loss + total_sparsity + 0.1 * phase_ce
        
        info = {
            'mse': mse_loss.item(),
            'sparsity': total_sparsity.item(),
            'phase_ce': phase_ce.item() if isinstance(phase_ce, torch.Tensor) else phase_ce,
            'total': total_loss.item()
        }
        
        # Add per-phase sparsities
        for idx, sp in enumerate(phase_sparsities):
            info[f'sparsity_{self.phase_names[idx]}'] = sp
        
        return total_loss, info
    
    def visualize_phase_causal_graphs(
        self,
        param_names: Optional[List[str]] = None,
        effect_names: Optional[List[str]] = None,
        threshold: float = 0.3
    ):
        """Visualize causal graphs for each phase."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if param_names is None:
            param_names = [f"p_{i}" for i in range(self.input_dim)]
        if effect_names is None:
            effect_names = [f"e_{k}" for k in range(self.output_dim)]
        
        fig, axes = plt.subplots(1, self.num_phases, figsize=(5*self.num_phases, 8))
        
        for phase_idx, (ax, model) in enumerate(zip(axes, self.phase_models)):
            soft, _ = model.get_causal_weights(threshold)
            weights = soft[:self.input_dim].cpu().detach().numpy()
            
            sns.heatmap(
                weights,
                ax=ax,
                xticklabels=effect_names,
                yticklabels=param_names,
                cmap="YlOrRd",
                vmin=0, vmax=1,
                annot=True,
                fmt=".2f"
            )
            ax.set_title(f"Phase: {self.phase_names[phase_idx]}")
        
        plt.tight_layout()
        return fig
