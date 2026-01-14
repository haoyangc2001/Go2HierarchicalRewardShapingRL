import torch
import os
from typing import Optional
from legged_gym.envs.go2.go2_env import GO2Robot
from legged_gym.envs.go2.high_level_navigation_env import HighLevelNavigationEnv, HighLevelNavigationConfig
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyRunner


class HierarchicalGO2Env:
    """
    Hierarchical GO2 environment that combines low-level locomotion control with a high-level navigation policy.
    - Low level: pretrained locomotion policy (velocity commands -> joint actions)
    - High level: navigation policy to be trained (observations -> velocity commands)
    """

    def __init__(self, cfg, low_level_model_path: str, args=None, device='cuda:0'):
        """
        Args:
            cfg: Environment configuration
            low_level_model_path: Path to the low-level policy checkpoint
            args: CLI arguments used to build the environment
            device: Compute device
        """
        self.cfg = cfg
        self.args = args
        self.device = device
        self.low_level_model_path = low_level_model_path

        # Create the underlying GO2 environment
        self.base_env = self._create_base_env()
        if hasattr(self.base_env.cfg, "rewards_ext"):
            terminate_on_reach_avoid = True
            if hasattr(self.cfg, "reward_shaping"):
                terminate_on_reach_avoid = bool(
                    getattr(self.cfg.reward_shaping, "terminate_on_safety_violation", True)
                    or getattr(self.cfg.reward_shaping, "terminate_on_success", True)
                )
            setattr(self.base_env.cfg.rewards_ext, "terminate_on_reach_avoid", terminate_on_reach_avoid)

        # Load the low-level locomotion policy
        self.low_level_policy = self._load_low_level_policy()

        # Build the high-level navigation wrapper
        self.high_level_config = HighLevelNavigationConfig()
        self._update_high_level_config()
        self.high_level_env = HighLevelNavigationEnv(self.base_env, self.high_level_config)
        self.low_level_action_repeat = getattr(self.cfg.env, 'high_level_action_repeat', 1)

        # Environment properties exposed to the algorithm
        self.num_envs = self.base_env.num_envs
        self.num_obs = self.high_level_env.num_high_level_obs  # high-level observation dimension
        self.num_actions = self.high_level_env.num_high_level_actions  # high-level action dimension
        self.device = self.base_env.device

    def _create_base_env(self):
        """Instantiate the original GO2 environment."""
        # Build the base environment using the standard registry entry
        env, _ = task_registry.make_env(name="go2", args=self.args)
        return env

    def _get_dummy_args(self):
        """Create dummy CLI arguments for environment initialization."""
        # Return the real args when they are provided, otherwise build a placeholder object

        if self.args is not None:
            return self.args

        class DummyArgs:
            def __init__(self, device, cfg):
                self.headless = False
                self.rl_device = str(device)
                self.sim_device = str(device)
                self.graphics_device_id = 0
                self.num_envs = getattr(cfg.env, 'num_envs', 2)
                self.physics_engine = "physx"
                self.use_gpu = True
                self.use_gpu_pipeline = True
                self.subscenes = 0
                self.num_threads = 0
                self.sim_device_type = "cuda" if "cuda" in str(device) else "cpu"
                self.compute_device_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
                self.sim_device_id = self.compute_device_id

        return DummyArgs(self.device, self.cfg)

    def _load_low_level_policy(self):
        """Load the pretrained low-level policy."""
        if not os.path.exists(self.low_level_model_path):
            raise FileNotFoundError(f"Low-level policy checkpoint not found: {self.low_level_model_path}")

        # Create a PPO runner to restore the low-level policy
        train_cfg = self._get_low_level_train_cfg()
        train_cfg_dict = class_to_dict(train_cfg)
        ppo_runner = OnPolicyRunner(self.base_env, train_cfg_dict, device=self.device)

        # Load weights and return an inference callable
        print(f"Loading low-level policy checkpoint: {self.low_level_model_path}")
        ppo_runner.load(self.low_level_model_path)

        return ppo_runner.get_inference_policy(device=self.device)

    def _get_low_level_train_cfg(self):
        """Retrieve the configuration used to train the low-level policy."""
        _, train_cfg = task_registry.get_cfgs(name="go2")
        train_cfg.runner.resume = False  # do not automatically resume
        return train_cfg

    def _update_high_level_config(self):
        """Copy reach-avoid specific parameters from the environment config."""
        if hasattr(self.cfg, 'rewards_ext'):
            # Support both single and multiple obstacle configurations
            if hasattr(self.cfg.rewards_ext, 'unsafe_spheres_pos'):
                self.high_level_config.unsafe_spheres_pos = self.cfg.rewards_ext.unsafe_spheres_pos
            if hasattr(self.cfg.rewards_ext, 'unsafe_sphere_pos'):
                self.high_level_config.unsafe_sphere_pos = self.cfg.rewards_ext.unsafe_sphere_pos
            self.high_level_config.unsafe_sphere_radius = self.cfg.rewards_ext.unsafe_sphere_radius
            self.high_level_config.target_radius = self.cfg.rewards_ext.target_sphere_radius
            self.high_level_config.target_sphere_pos = self.cfg.rewards_ext.target_sphere_pos
        if hasattr(self.cfg, "enable_manual_lidar"):
            self.high_level_config.enable_manual_lidar = self.cfg.enable_manual_lidar
        if hasattr(self.cfg, "lidar_max_range"):
            self.high_level_config.lidar_max_range = self.cfg.lidar_max_range
        if hasattr(self.cfg, "lidar_num_bins"):
            self.high_level_config.lidar_num_bins = self.cfg.lidar_num_bins

    def reset(self):
        """Reset the environment and return high-level observations and metrics."""
        high_level_obs, g_values, h_values = self.high_level_env.reset()
        return high_level_obs, g_values, h_values

    def step(self, high_level_actions):
        """
        Run one high-level interaction step.

        Args:
            high_level_actions: [num_envs, 3] high-level velocity commands [vx, vy, vyaw]
        Returns:
            observations: [num_envs, num_obs] high-level observations
            g_values: [num_envs] reach metric values
            h_values: [num_envs] avoid metric values
            dones: [num_envs] termination flags
            infos: Additional diagnostic information
        """
        # 1. Update desired velocity commands
        self.high_level_env.update_velocity_commands(high_level_actions)
        desired_velocity_commands = self.base_env.commands[:, :3].clone()

        # 2. Execute the low-level policy multiple times to honor the commands
        base_obs = None
        base_infos = None
        avoid_metric = None
        reach_metric = None
        aggregated_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.low_level_action_repeat):
            self.base_env.commands[:, :3] = desired_velocity_commands
            self.base_env.compute_observations()
            current_base_obs = self.base_env.get_observations()
            with torch.no_grad():
                low_level_actions = self.low_level_policy(current_base_obs)

            base_obs, privileged_obs, _, step_dones, base_infos, avoid_metric, reach_metric = self.base_env.step(
                low_level_actions
            )

            aggregated_dones |= step_dones.bool()

        # 4. Compute the updated high-level observation
        self.high_level_env._compute_high_level_observations()
        high_level_obs = self.high_level_env.get_observations()

        # 5. Convert reach/avoid metrics into g/h values
        g_values, h_values = self.high_level_env.compute_g_h_values(avoid_metric, reach_metric)

        # 6. Assemble info dictionary for logging/debugging
        infos = {
            'base_infos': base_infos,
            'avoid_metric': avoid_metric,
            'reach_metric': reach_metric,
            'base_obs': base_obs
        }

        return high_level_obs, g_values, h_values, aggregated_dones, infos

    def get_observations(self):
        """Return the current high-level observation buffer."""
        return self.high_level_env.get_observations()

    def close(self):
        """Release resources held by the environment."""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


def create_hierarchical_go2_env(env_cfg, low_level_model_path: str, device='cuda:0'):
    """
    Helper function to construct the hierarchical GO2 environment.

    Args:
        env_cfg: Environment configuration
        low_level_model_path: Path to the pretrained low-level policy
        device: Compute device

    Returns:
        HierarchicalGO2Env: Instantiated hierarchical environment
    """
    return HierarchicalGO2Env(env_cfg, low_level_model_path, device)
