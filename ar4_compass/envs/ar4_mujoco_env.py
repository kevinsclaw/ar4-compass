"""
AR4 MuJoCo Environment for COMPASS Sim-to-Real experiments.

This environment wraps the AR4 6-DOF robot arm MuJoCo model and provides:
- Tunable environment parameters (damping, friction, inertia, etc.)
- Rollout functionality with configurable policies
- Trajectory difference computation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not installed. AR4 environment will not work.")


# Default paths
AR4_MJCF_PATH = Path("/home/ubuntu/ar4/ar4_mujoco_sim/mjcf/scene.xml")


@dataclass
class AR4EnvConfig:
    """Configuration for AR4 environment."""
    mjcf_path: str = str(AR4_MJCF_PATH)
    timestep: float = 0.002  # 500 Hz
    frame_skip: int = 4  # Control at 125 Hz
    max_episode_steps: int = 200
    
    # Observation settings
    obs_include_qpos: bool = True
    obs_include_qvel: bool = True
    obs_include_ee_pos: bool = True
    obs_include_ee_quat: bool = True
    
    # Action settings
    action_scale: float = 1.0
    
    # Rendering
    render_mode: Optional[str] = None  # None, "human", "rgb_array"
    camera_name: Optional[str] = "front_camera"


@dataclass 
class AR4EnvParams:
    """
    Tunable environment parameters for AR4.
    
    These are the parameters that can be modified for domain randomization
    and sim-to-real transfer experiments.
    """
    # Joint parameters (6 joints)
    joint_damping: np.ndarray = field(default_factory=lambda: np.array([-10.0] * 6))
    joint_frictionloss: np.ndarray = field(default_factory=lambda: np.array([0.1] * 6))
    joint_armature: np.ndarray = field(default_factory=lambda: np.array([0.1] * 6))
    
    # Actuator parameters
    actuator_force_scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    actuator_ctrl_delay: float = 0.0  # seconds
    
    # Contact parameters  
    contact_friction_sliding: float = 0.8
    contact_friction_torsional: float = 0.02
    contact_friction_rolling: float = 0.01
    
    # Link mass scaling (relative to default)
    link_mass_scale: np.ndarray = field(default_factory=lambda: np.ones(7))  # base + 6 links
    
    # Sensing noise
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert parameters to a flat vector for causal discovery."""
        return np.concatenate([
            self.joint_damping,           # 0-5
            self.joint_frictionloss,      # 6-11
            self.joint_armature,          # 12-17
            self.actuator_force_scale,    # 18-23
            [self.actuator_ctrl_delay],   # 24
            [self.contact_friction_sliding],    # 25
            [self.contact_friction_torsional],  # 26
            [self.contact_friction_rolling],    # 27
            self.link_mass_scale,         # 28-34
            [self.position_noise_std],    # 35
            [self.velocity_noise_std],    # 36
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'AR4EnvParams':
        """Create parameters from a flat vector."""
        return cls(
            joint_damping=vec[0:6],
            joint_frictionloss=vec[6:12],
            joint_armature=vec[12:18],
            actuator_force_scale=vec[18:24],
            actuator_ctrl_delay=vec[24],
            contact_friction_sliding=vec[25],
            contact_friction_torsional=vec[26],
            contact_friction_rolling=vec[27],
            link_mass_scale=vec[28:35],
            position_noise_std=vec[35],
            velocity_noise_std=vec[36],
        )
    
    @classmethod
    def param_names(cls) -> List[str]:
        """Get names of all parameters."""
        names = []
        names += [f"joint_{i+1}@damping" for i in range(6)]
        names += [f"joint_{i+1}@frictionloss" for i in range(6)]
        names += [f"joint_{i+1}@armature" for i in range(6)]
        names += [f"actuator_{i+1}@force_scale" for i in range(6)]
        names += ["actuator@ctrl_delay"]
        names += ["contact@friction_sliding", "contact@friction_torsional", "contact@friction_rolling"]
        names += [f"link_{i}@mass_scale" for i in range(7)]
        names += ["sensing@position_noise", "sensing@velocity_noise"]
        return names
    
    @classmethod
    def n_params(cls) -> int:
        """Number of tunable parameters."""
        return 37  # 6+6+6+6+1+3+7+2
    
    def copy(self) -> 'AR4EnvParams':
        """Create a deep copy."""
        return AR4EnvParams(
            joint_damping=self.joint_damping.copy(),
            joint_frictionloss=self.joint_frictionloss.copy(),
            joint_armature=self.joint_armature.copy(),
            actuator_force_scale=self.actuator_force_scale.copy(),
            actuator_ctrl_delay=self.actuator_ctrl_delay,
            contact_friction_sliding=self.contact_friction_sliding,
            contact_friction_torsional=self.contact_friction_torsional,
            contact_friction_rolling=self.contact_friction_rolling,
            link_mass_scale=self.link_mass_scale.copy(),
            position_noise_std=self.position_noise_std,
            velocity_noise_std=self.velocity_noise_std,
        )


class AR4MuJoCoEnv:
    """
    AR4 MuJoCo Environment for sim-to-real experiments.
    
    Features:
    - Load AR4 robot model
    - Get/set environment parameters
    - Run rollouts with arbitrary policies
    - Compute trajectory differences
    """
    
    # Joint and actuator names in the model
    JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
    ACTUATOR_NAMES = [f"joint_{i}_ctrl" for i in range(1, 7)]
    LINK_NAMES = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
    
    def __init__(self, config: Optional[AR4EnvConfig] = None):
        """
        Initialize AR4 MuJoCo environment.
        
        Args:
            config: Environment configuration
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("mujoco package is required. Install with: pip install mujoco")
        
        self.config = config or AR4EnvConfig()
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(self.config.mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Store original model parameters for resetting
        self._original_params = self._get_model_params()
        
        # Current environment parameters
        self.env_params = AR4EnvParams()
        
        # Rendering
        self.viewer = None
        self.renderer = None
        
        # Get joint and actuator indices
        self._joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                          for name in self.JOINT_NAMES]
        self._actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                             for name in self.ACTUATOR_NAMES]
        self._link_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                              for name in self.LINK_NAMES]
        
        # End-effector (link_6)
        self._ee_body_id = self._link_body_ids[-1]
        
        # Spaces
        self.n_joints = 6
        self.action_dim = 6
        self.obs_dim = self._compute_obs_dim()
        
        # Step counter
        self._step_count = 0
    
    def _compute_obs_dim(self) -> int:
        """Compute observation dimension based on config."""
        dim = 0
        if self.config.obs_include_qpos:
            dim += self.n_joints
        if self.config.obs_include_qvel:
            dim += self.n_joints
        if self.config.obs_include_ee_pos:
            dim += 3
        if self.config.obs_include_ee_quat:
            dim += 4
        return dim
    
    def _get_model_params(self) -> Dict[str, np.ndarray]:
        """Extract current model parameters."""
        return {
            'dof_damping': self.model.dof_damping.copy(),
            'dof_frictionloss': self.model.dof_frictionloss.copy(),
            'dof_armature': self.model.dof_armature.copy(),
            'body_mass': self.model.body_mass.copy(),
            'actuator_forcerange': self.model.actuator_forcerange.copy(),
        }
    
    def set_env_params(self, params: AR4EnvParams):
        """
        Apply environment parameters to the MuJoCo model.
        
        Args:
            params: Environment parameters to apply
        """
        self.env_params = params.copy()
        
        # Apply joint parameters
        for i, joint_id in enumerate(self._joint_ids):
            dof_id = self.model.jnt_dofadr[joint_id]
            self.model.dof_damping[dof_id] = params.joint_damping[i]
            self.model.dof_frictionloss[dof_id] = params.joint_frictionloss[i]
            self.model.dof_armature[dof_id] = params.joint_armature[i]
        
        # Apply link mass scaling
        for i, body_id in enumerate(self._link_body_ids):
            original_mass = self._original_params['body_mass'][body_id]
            self.model.body_mass[body_id] = original_mass * params.link_mass_scale[i]
        
        # Note: Actuator force scaling and contact parameters are applied during simulation
    
    def get_env_params(self) -> AR4EnvParams:
        """Get current environment parameters."""
        return self.env_params.copy()
    
    def get_env_params_vector(self) -> np.ndarray:
        """Get environment parameters as a flat vector."""
        return self.env_params.to_vector()
    
    def set_env_params_vector(self, vec: np.ndarray):
        """Set environment parameters from a flat vector."""
        self.set_env_params(AR4EnvParams.from_vector(vec))
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment.
        
        Args:
            seed: Random seed (optional)
            
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions (optional: randomize)
        # Default: all zeros (home position)
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self._step_count = 0
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment.
        
        Args:
            action: Joint torque commands [6]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action and apply force scaling
        action = np.clip(action, -1, 1) * self.config.action_scale
        action = action * self.env_params.actuator_force_scale
        
        # Apply control
        self.data.ctrl[:] = action
        
        # Simulate
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward (simple distance-based for now)
        reward = 0.0
        
        # Check termination
        terminated = False
        truncated = self._step_count >= self.config.max_episode_steps
        
        info = {
            'ee_pos': self.get_ee_position(),
            'joint_pos': self.get_joint_positions(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        obs_parts = []
        
        if self.config.obs_include_qpos:
            qpos = self.get_joint_positions()
            if self.env_params.position_noise_std > 0:
                qpos += np.random.randn(len(qpos)) * self.env_params.position_noise_std
            obs_parts.append(qpos)
        
        if self.config.obs_include_qvel:
            qvel = self.get_joint_velocities()
            if self.env_params.velocity_noise_std > 0:
                qvel += np.random.randn(len(qvel)) * self.env_params.velocity_noise_std
            obs_parts.append(qvel)
        
        if self.config.obs_include_ee_pos:
            obs_parts.append(self.get_ee_position())
        
        if self.config.obs_include_ee_quat:
            obs_parts.append(self.get_ee_orientation())
        
        return np.concatenate(obs_parts)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get joint positions [6]."""
        return np.array([self.data.qpos[self.model.jnt_qposadr[jid]] 
                        for jid in self._joint_ids])
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get joint velocities [6]."""
        return np.array([self.data.qvel[self.model.jnt_dofadr[jid]] 
                        for jid in self._joint_ids])
    
    def get_ee_position(self) -> np.ndarray:
        """Get end-effector position [3]."""
        return self.data.xpos[self._ee_body_id].copy()
    
    def get_ee_orientation(self) -> np.ndarray:
        """Get end-effector orientation as quaternion [4]."""
        return self.data.xquat[self._ee_body_id].copy()
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.config.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        elif self.config.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera=self.config.camera_name)
            return self.renderer.render()
        return None
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def rollout_trajectory(
    env: AR4MuJoCoEnv,
    policy,
    max_steps: int = 200,
    deterministic: bool = True
) -> Dict[str, np.ndarray]:
    """
    Rollout a trajectory using the given policy.
    
    Args:
        env: AR4 environment
        policy: Policy function or object with predict() method
        max_steps: Maximum trajectory length
        deterministic: Use deterministic actions
        
    Returns:
        Dictionary with trajectory data
    """
    obs_list = []
    action_list = []
    ee_pos_list = []
    joint_pos_list = []
    
    obs = env.reset()
    obs_list.append(obs)
    ee_pos_list.append(env.get_ee_position())
    joint_pos_list.append(env.get_joint_positions())
    
    for _ in range(max_steps):
        # Get action from policy
        if callable(policy):
            action = policy(obs)
        elif hasattr(policy, 'predict'):
            action, _ = policy.predict(obs, deterministic=deterministic)
        else:
            raise ValueError("Policy must be callable or have predict() method")
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_list.append(action)
        obs_list.append(obs)
        ee_pos_list.append(info['ee_pos'])
        joint_pos_list.append(info['joint_pos'])
        
        if terminated or truncated:
            break
    
    return {
        'observations': np.array(obs_list),
        'actions': np.array(action_list),
        'ee_positions': np.array(ee_pos_list),
        'joint_positions': np.array(joint_pos_list),
    }


def compute_trajectory_difference(
    traj_sim: Dict[str, np.ndarray],
    traj_real: Dict[str, np.ndarray],
    components: List[str] = ['ee_positions']
) -> np.ndarray:
    """
    Compute trajectory difference between sim and real.
    
    Args:
        traj_sim: Simulated trajectory
        traj_real: Real (or target) trajectory
        components: Which components to compare
        
    Returns:
        Trajectory difference vector [K]
    """
    differences = []
    
    for comp in components:
        if comp not in traj_sim or comp not in traj_real:
            continue
        
        sim_data = traj_sim[comp]
        real_data = traj_real[comp]
        
        # Align lengths
        min_len = min(len(sim_data), len(real_data))
        sim_data = sim_data[:min_len]
        real_data = real_data[:min_len]
        
        # Compute per-dimension cumulative error
        if sim_data.ndim == 1:
            diff = np.sum(np.abs(sim_data - real_data))
            differences.append(diff)
        else:
            for d in range(sim_data.shape[1]):
                diff = np.sum(np.abs(sim_data[:, d] - real_data[:, d]))
                differences.append(diff)
    
    return np.array(differences)


# Effect names for trajectory difference components
TRAJECTORY_EFFECT_NAMES = [
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
]


def random_policy(obs: np.ndarray) -> np.ndarray:
    """Simple random policy for testing."""
    return np.random.uniform(-0.5, 0.5, size=6)


def sine_wave_policy(obs: np.ndarray, t: float = 0, freq: float = 0.5) -> np.ndarray:
    """Sine wave policy for reproducible trajectories."""
    return 0.3 * np.sin(2 * np.pi * freq * t + np.arange(6) * np.pi / 3)
