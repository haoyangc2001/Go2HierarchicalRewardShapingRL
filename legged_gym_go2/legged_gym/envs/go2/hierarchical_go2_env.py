import torch
import os
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
        self.reward_cfg = getattr(self.cfg, "reward_shaping", None)
        if self.reward_cfg is not None and hasattr(self.base_env.cfg, "rewards_ext"):
            if hasattr(self.reward_cfg, "goal_reached_dist"):
                self.base_env.cfg.rewards_ext.goal_reached_dist = float(self.reward_cfg.goal_reached_dist)
            if hasattr(self.reward_cfg, "collision_dist"):
                self.base_env.cfg.rewards_ext.collision_dist = float(self.reward_cfg.collision_dist)

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
        self.prev_target_distance = None
        self.prev_commands = None

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
            if hasattr(self.cfg.rewards_ext, "boundary_margin"):
                self.high_level_config.boundary_margin = self.cfg.rewards_ext.boundary_margin
        if hasattr(self.cfg, "enable_manual_lidar"):
            self.high_level_config.enable_manual_lidar = self.cfg.enable_manual_lidar
        if hasattr(self.cfg, "lidar_max_range"):
            self.high_level_config.lidar_max_range = self.cfg.lidar_max_range
        if hasattr(self.cfg, "lidar_num_bins"):
            self.high_level_config.lidar_num_bins = self.cfg.lidar_num_bins
        if hasattr(self.cfg, "target_lidar_num_bins"):
            self.high_level_config.target_lidar_num_bins = self.cfg.target_lidar_num_bins
        if hasattr(self.cfg, "target_lidar_max_range"):
            self.high_level_config.target_lidar_max_range = self.cfg.target_lidar_max_range
        if hasattr(self.cfg, "reach_metric_scale"):
            self.high_level_config.reach_metric_scale = self.cfg.reach_metric_scale
        if hasattr(self.cfg, "terrain"):
            terrain_length = getattr(self.cfg.terrain, "terrain_length", None)
            terrain_width = getattr(self.cfg.terrain, "terrain_width", None)
            if terrain_length and terrain_width:
                self.high_level_config.boundary_half_extents = (
                    float(terrain_length) * 0.5,
                    float(terrain_width) * 0.5,
                )

    def reset(self):
        """Reset the environment and return high-level observations."""
        high_level_obs = self.high_level_env.reset()
        self.prev_target_distance = self.high_level_env.extract_target_distance(high_level_obs).clone()
        self.prev_commands = torch.zeros(
            self.num_envs, self.num_actions, device=self.device, dtype=torch.float
        )
        return high_level_obs

    def step(self, high_level_actions):
        """
        Run one high-level interaction step.

        Args:
            high_level_actions: [num_envs, 3] high-level velocity commands [vx, vy, vyaw]
        Returns:
            observations: [num_envs, num_obs] high-level observations
            rewards: [num_envs] reward values
            dones: [num_envs] termination flags
            infos: Additional diagnostic information
        """
        # 1. Update desired velocity commands
        self.high_level_env.update_velocity_commands(high_level_actions)
        desired_velocity_commands = self.base_env.commands[:, :3].clone()

        # 2. Execute the low-level policy multiple times to honor the commands
        base_infos = None
        aggregated_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reach_buf = None
        min_hazard_buf = None
        boundary_buf = None
        obstacle_surface_buf = None
        base_lin_vel_buf = None
        time_outs_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.low_level_action_repeat):
            self.base_env.commands[:, :3] = desired_velocity_commands
            self.base_env.compute_observations()
            current_base_obs = self.base_env.get_observations()
            with torch.no_grad():
                low_level_actions = self.low_level_policy(current_base_obs)

            _, _, _, step_dones, base_infos = self.base_env.step(
                low_level_actions
            )

            step_dones = step_dones.bool()
            step_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if isinstance(base_infos, dict) and "time_outs" in base_infos:
                step_time_outs = base_infos["time_outs"].to(self.device).bool()

            reach_metric = self.base_env.reach_metric
            if reach_buf is None:
                reach_buf = reach_metric.clone()
                min_hazard_buf = self.base_env.min_hazard_distance.clone()
                boundary_buf = self.base_env.boundary_distance.clone()
                obstacle_surface_buf = self.base_env.obstacle_surface_distance.clone()
                base_lin_vel_buf = self.base_env.base_lin_vel.clone()
                time_outs_buf = step_time_outs.clone()
            else:
                update_mask = ~done_mask
                reach_buf = torch.where(update_mask, reach_metric, reach_buf)
                min_hazard_buf = torch.where(
                    update_mask, self.base_env.min_hazard_distance, min_hazard_buf
                )
                boundary_buf = torch.where(update_mask, self.base_env.boundary_distance, boundary_buf)
                obstacle_surface_buf = torch.where(
                    update_mask, self.base_env.obstacle_surface_distance, obstacle_surface_buf
                )
                base_lin_vel_buf = torch.where(
                    update_mask.unsqueeze(1), self.base_env.base_lin_vel, base_lin_vel_buf
                )
                time_outs_buf = torch.where(update_mask, step_time_outs, time_outs_buf)

            done_mask |= step_dones
            aggregated_dones |= step_dones

        # 4. Compute the updated high-level observation
        self.high_level_env._compute_high_level_observations()
        high_level_obs = self.high_level_env.get_observations()

        # 5. Derive distances from lidar observations; use reach_metric for rewards.
        target_distance_est = self.high_level_env.extract_target_distance(high_level_obs)
        hazard_distance_est = self.high_level_env.extract_hazard_distance(high_level_obs)
        reach_metric = reach_buf if reach_buf is not None else self.base_env.reach_metric.clone()
        min_hazard_true = (
            min_hazard_buf if min_hazard_buf is not None else self.base_env.min_hazard_distance.clone()
        )
        if reach_buf is not None:
            target_distance_est = torch.where(aggregated_dones, reach_buf, target_distance_est)
        if min_hazard_buf is not None:
            hazard_distance_est = torch.where(aggregated_dones, min_hazard_buf, hazard_distance_est)

        hazard_distance_for_reward = min_hazard_true if min_hazard_true is not None else hazard_distance_est

        reset_mask = self.base_env.episode_length_buf == 0
        reward, done_flags, reached, success, collision, terminated, truncated, components = self._compute_reward(
            high_level_obs=high_level_obs,
            desired_commands=desired_velocity_commands,
            target_distance=reach_metric,
            hazard_distance=hazard_distance_for_reward,
            base_dones=aggregated_dones,
            time_outs=time_outs_buf,
            reset_mask=reset_mask,
        )

        # Update history buffers for the next step
        self.prev_target_distance = reach_metric.detach().clone()
        self.prev_commands = desired_velocity_commands.detach().clone()

        # 6. Assemble info dictionary for logging/debugging
        infos = {
            "time_outs": time_outs_buf,
            "reached": reached,
            "success": success,
            "collision": collision,
            "terminated": terminated,
            "truncated": truncated,
            "target_distance": reach_metric,
            "target_distance_est": target_distance_est,
            "reach_metric": reach_metric,
            "min_hazard_distance": hazard_distance_est,
            "min_hazard_distance_est": hazard_distance_est,
            "min_hazard_distance_true": min_hazard_true,
            "boundary_distance": boundary_buf,
            "obstacle_surface_distance": obstacle_surface_buf,
            "base_lin_vel": base_lin_vel_buf,
            "desired_commands": desired_velocity_commands,
            "progress": components["progress"],
            "alignment": components["alignment"],
            "command_alignment": components["command_alignment"],
            "obstacle_penalty": components["obstacle_penalty"],
            "command_speed": components["command_speed"],
            "body_speed": components["body_speed"],
            "command_delta": components["command_delta"],
            "reward_clip_frac": components["reward_clip_frac"],
        }

        return high_level_obs, reward, done_flags, infos

    def _compute_reward(
        self,
        high_level_obs: torch.Tensor,
        desired_commands: torch.Tensor,
        target_distance: torch.Tensor,
        hazard_distance: torch.Tensor,
        base_dones: torch.Tensor,
        time_outs: torch.Tensor,
        reset_mask: torch.Tensor,
    ):
        cfg = self.reward_cfg
        if self.prev_target_distance is None:
            self.prev_target_distance = target_distance.detach().clone()
        if self.prev_commands is None:
            self.prev_commands = desired_commands.detach().clone()
        goal_reached_dist = float(getattr(cfg, "goal_reached_dist", 0.3))
        collision_dist = float(getattr(cfg, "collision_dist", 0.35))
        obstacle_avoid_dist = float(getattr(cfg, "obstacle_avoid_dist", 1.0))
        progress_scale = float(getattr(cfg, "progress_scale", 10.0))
        alignment_scale = float(getattr(cfg, "alignment_scale", 1.0))
        obstacle_penalty_scale = float(getattr(cfg, "obstacle_penalty_scale", 1.0))
        yaw_rate_scale = float(getattr(cfg, "yaw_rate_scale", 0.0))
        action_smooth_scale = float(getattr(cfg, "action_smooth_scale", 0.0))
        body_speed_scale = float(getattr(cfg, "body_speed_scale", 0.0))
        command_alignment_scale = float(getattr(cfg, "command_alignment_scale", 0.0))
        idle_speed = float(getattr(cfg, "idle_speed_threshold", 0.05))
        idle_dist = float(getattr(cfg, "idle_distance_threshold", 0.5))
        idle_penalty_scale = float(getattr(cfg, "idle_penalty_scale", 0.0))
        success_reward = float(getattr(cfg, "success_reward", 100.0))
        collision_penalty = float(getattr(cfg, "collision_penalty", 100.0))
        timeout_penalty = float(getattr(cfg, "timeout_penalty", 0.0))

        body_vel_xy = self.high_level_env.extract_body_vel_xy(high_level_obs)
        body_speed = torch.norm(body_vel_xy, dim=1)
        target_dir = high_level_obs[:, 6:8]
        alignment = torch.sum(body_vel_xy * target_dir, dim=1)
        command_alignment = torch.sum(desired_commands[:, :2] * target_dir, dim=1)
        yaw_rate = high_level_obs[:, 4] / self.high_level_env.ang_vel_scale

        hazard_penalty = torch.clamp(
            (obstacle_avoid_dist - hazard_distance) / obstacle_avoid_dist, min=0.0
        )
        command_speed = torch.norm(desired_commands[:, :2], dim=1)

        effective_prev_dist = torch.where(reset_mask, target_distance, self.prev_target_distance)
        progress = effective_prev_dist - target_distance
        effective_prev_cmd = torch.where(
            reset_mask.unsqueeze(1), desired_commands, self.prev_commands
        )
        command_delta = torch.norm(desired_commands - effective_prev_cmd, dim=1)

        reached = target_distance <= goal_reached_dist
        hazard_collision = hazard_distance <= collision_dist
        # Keep done flags aligned with the base environment resets to avoid desyncs.
        done_flags = base_dones.clone()

        terminated = done_flags & ~time_outs
        truncated = time_outs & ~terminated
        failure = terminated & ~reached
        collision = (hazard_collision | failure) & done_flags

        active_mask = (~done_flags).float()
        progress = progress * (~(terminated | truncated)).float()
        alignment = alignment * active_mask
        command_alignment = command_alignment * active_mask
        hazard_penalty = hazard_penalty * active_mask
        body_speed = body_speed * active_mask
        command_speed = command_speed * active_mask
        yaw_penalty = torch.abs(yaw_rate) * active_mask
        command_delta = command_delta * active_mask

        reward = (
            progress_scale * progress
            + alignment_scale * alignment
            + command_alignment_scale * command_alignment
            - obstacle_penalty_scale * hazard_penalty
            - yaw_rate_scale * yaw_penalty
            - action_smooth_scale * command_delta
        )
        if body_speed_scale > 0.0:
            reward = reward + body_speed_scale * body_speed
        if idle_penalty_scale > 0.0:
            idle_mask = (body_speed < idle_speed) & (target_distance > idle_dist)
            reward = reward - idle_penalty_scale * idle_mask.float()

        success = reached & done_flags & ~collision
        reward = torch.where(success, reward + success_reward, reward)
        reward = torch.where(collision, reward - collision_penalty, reward)
        reward = torch.where(truncated, reward - timeout_penalty, reward)

        reward_scale = float(getattr(cfg, "reward_scale", 1.0))
        reward = reward * reward_scale
        reward_clip = float(getattr(cfg, "reward_clip", 0.0))
        if reward_clip > 0.0:
            clipped = torch.clamp(reward, -reward_clip, reward_clip)
            reward_clip_frac = (clipped != reward).float()
            reward = clipped
        else:
            reward_clip_frac = torch.zeros_like(reward)

        components = {
            "progress": progress,
            "alignment": alignment,
            "command_alignment": command_alignment,
            "obstacle_penalty": hazard_penalty,
            "command_speed": command_speed,
            "body_speed": body_speed,
            "command_delta": command_delta,
            "reward_clip_frac": reward_clip_frac,
        }

        return reward, done_flags, reached, success, collision, terminated, truncated, components

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
