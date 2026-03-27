# AR4-COMPASS: Advanced Causal Discovery for Sim-to-Real Transfer

> **Extending COMPASS with Novel Causal Discovery Methods for 6-DOF Manipulator Sim-to-Real Transfer**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-green.svg)](https://mujoco.org/)

## 🎯 Overview

This project extends the [COMPASS](https://github.com/XilunZhangRobo/COMPASS-Sim2Real) framework with **novel causal discovery methods** specifically designed for robotic manipulator sim-to-real transfer. We focus on the [AR4 6-DOF robot arm](https://www.anninrobotics.com/) as our target platform.

### Key Innovations

| Method | Description | Status |
|--------|-------------|--------|
| **Attention-based Causal Discovery** | Replace Gumbel-Softmax with learned attention for continuous causal weights | 🔄 In Progress |
| **Neural Additive Models (NAM)** | Interpretable per-parameter contribution functions | 🔄 In Progress |
| **Variational Causal Graph** | Bayesian approach with uncertainty quantification | 📋 Planned |
| **Intervention-based Discovery** | Direct causal effect estimation via simulation interventions | 📋 Planned |
| **Phase-Aware Temporal Causality** | Task-phase-specific causal structures for manipulation | 📋 Planned |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AR4-COMPASS Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Intervention│ → │   Causal    │ → │  Parameter  │     │
│  │Pre-screening│    │  Discovery  │    │Optimization │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ↓                  ↓                  ↓             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Causal Discovery Methods               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │Original │ │Attention│ │  NAM    │ │Temporal │   │   │
│  │  │COMPASS  │ │ -based  │ │         │ │ Phase   │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  AR4 MuJoCo Env                      │   │
│  │  • 6-DOF joints  • Pick-and-Place  • Gripper        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Comparison of Causal Discovery Methods

### Original COMPASS (Gumbel-Softmax Binary Graph)

**Pros:**
- Simple and effective
- Differentiable through Gumbel-Softmax
- Sparse by design

**Cons:**
- Binary edges only (0/1)
- No causal strength quantification
- Static structure across trajectory
- Temperature scheduling sensitive

### Our Improvements

#### 1. Attention-based Causal Discovery

```python
# Soft causal weights via cross-attention
Q = effect_embeddings      # [K, d] - trajectory differences
K = V = param_embeddings   # [|E|, d] - environment parameters

causal_weights = softmax(Q @ K.T / sqrt(d)) @ V
# causal_weights ∈ [0, 1]^{K × |E|} - continuous causal strength
```

**Advantages:**
- Continuous causal strength (not just binary)
- Learnable importance weighting
- Better gradient flow
- Natural sparsity via attention entropy regularization

#### 2. Neural Additive Models (NAM)

```python
# Additive decomposition: d_k = Σ_i f_{i,k}(ε_i)
effect_k = sum(f[i][k](param_i) for i in range(n_params))
```

**Advantages:**
- Inherently interpretable
- Per-parameter contribution curves
- Easy to visualize causal relationships
- No need for post-hoc analysis

#### 3. Variational Causal Graph

```python
# Treat causal graph as latent variable
G ~ q(G | data)           # Posterior
prior = Bernoulli(0.1)    # Sparse prior

ELBO = E_q[log p(d_τ | G, ε)] - KL(q(G) || prior)
```

**Advantages:**
- Uncertainty quantification
- Principled sparsity via prior
- Better generalization

#### 4. Intervention-based Discovery

```python
# Direct causal effect via do-calculus
effect = E[Y | do(X=x+δ)] - E[Y | do(X=x)]
       ≈ (rollout(params + δe_i) - rollout(params)) / δ
```

**Advantages:**
- Theoretically grounded (true causality)
- No learning required
- Perfect interventions in simulation
- Can bootstrap other methods

#### 5. Phase-Aware Temporal Causality

```python
# Different causal structures for different task phases
phases = ['approach', 'grasp', 'lift', 'place']
G = {phase: CausalGraph() for phase in phases}

# Approach: joint_damping, inertia matter most
# Grasp: friction, gripper_stiffness matter most
# Lift: gravity_comp, payload_inertia matter most
```

**Advantages:**
- Captures time-varying causality
- More accurate for multi-phase tasks
- Better interpretability for manipulation

## 🤖 AR4 Robot Parameters

### Tunable Parameter Space (~50 dimensions)

| Category | Parameters | Count |
|----------|-----------|-------|
| Joint Dynamics | `damping`, `frictionloss`, `armature` | 18 |
| Link Inertia | `ixx`, `iyy`, `izz` per link | 18 |
| Actuator | `ctrlrange`, `forcerange`, `gear` | 12 |
| Contact | `friction`, `stiffness`, `damping` | 9+ |
| Gripper | `stiffness`, `damping`, `friction` | 4+ |

### State Space Factorization (K dimensions)

| Factorization | Components | K |
|---------------|------------|---|
| End-effector only | `x, y, z, roll, pitch, yaw` | 6 |
| Joint angles | `θ_1, ..., θ_6` | 6 |
| Combined | End-effector + Joints | 12 |
| With velocity | Pos + Vel | 24 |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ar4-compass.git
cd ar4-compass

# Create conda environment
conda env create -f environment.yml
conda activate ar4-compass

# Install AR4 MuJoCo model
pip install -e .
```

### Run Experiments

```bash
# Sim-to-Sim with original COMPASS
python scripts/main.py --method compass --env ar4_pick_place

# Sim-to-Sim with Attention-based
python scripts/main.py --method attention --env ar4_pick_place

# Sim-to-Sim with NAM
python scripts/main.py --method nam --env ar4_pick_place

# Run all methods comparison
python scripts/compare_methods.py --env ar4_pick_place
```

## 📁 Project Structure

```
ar4-compass/
├── README.md
├── environment.yml
├── setup.py
├── configs/
│   ├── ar4_params.yaml          # AR4 parameter definitions
│   └── experiment_configs.yaml   # Experiment configurations
├── ar4_compass/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── ar4_mujoco_env.py    # AR4 MuJoCo environment
│   │   └── ar4_pick_place.py    # Pick-and-place task
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_causal.py       # Base causal model class
│   │   ├── compass_original.py  # Original COMPASS
│   │   ├── attention_causal.py  # Attention-based
│   │   ├── nam_causal.py        # Neural Additive Models
│   │   ├── variational_causal.py # Variational approach
│   │   └── temporal_causal.py   # Phase-aware temporal
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── intervention.py      # Intervention-based discovery
│   │   ├── trajectory.py        # Trajectory processing
│   │   └── visualization.py     # Plotting utilities
│   └── algorithms/
│       ├── __init__.py
│       ├── domain_randomization.py
│       └── parameter_optimization.py
├── scripts/
│   ├── main.py                  # Main training script
│   ├── compare_methods.py       # Method comparison
│   └── visualize_causal.py      # Causal graph visualization
├── experiments/
│   └── configs/                 # Experiment-specific configs
└── tests/
    └── test_models.py
```

## 📈 Experiment Plan

### Phase 1: Sim-to-Sim Validation
- [ ] Implement AR4 MuJoCo environment
- [ ] Create "target" environment with modified parameters
- [ ] Validate each causal discovery method
- [ ] Compare convergence speed and accuracy

### Phase 2: Method Comparison
- [ ] Trajectory alignment metrics
- [ ] Parameter estimation accuracy
- [ ] Computational efficiency
- [ ] Interpretability analysis

### Phase 3: Pick-and-Place Task
- [ ] Success rate comparison
- [ ] Generalization to unseen objects
- [ ] Phase-aware analysis

### Phase 4: (Optional) Sim-to-Real
- [ ] Real AR4 data collection
- [ ] Transfer learning evaluation

## 📚 References

1. **COMPASS**: Huang et al., "What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery", CoRL 2023
2. **Neural Additive Models**: Agarwal et al., "Neural Additive Models: Interpretable Machine Learning with Neural Nets", NeurIPS 2021
3. **Attention Mechanism**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
4. **Variational Inference for Graphs**: Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS Workshop 2016

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [COMPASS-Sim2Real](https://github.com/XilunZhangRobo/COMPASS-Sim2Real) - Original COMPASS implementation
- [AR4 ROS2](https://github.com/ycheng517/ar4_ros_driver) - AR4 robot description
- [MuJoCo](https://mujoco.org/) - Physics simulation engine
