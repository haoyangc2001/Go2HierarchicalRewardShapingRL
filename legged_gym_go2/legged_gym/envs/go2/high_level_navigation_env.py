import torch
import numpy as np
import math
from legged_gym.envs.go2.go2_env import GO2Robot
from legged_gym.utils.math import quat_apply
from legged_gym.utils.helpers import class_to_dict


class HighLevelNavigationEnv:
    """
    High-level navigation wrapper.
    - Observations: [cos_heading, sin_heading, body vx, body vy, yaw_rate, dist_to_target, target_dir_body]
      plus optional lidar bins
    - Actions: [vx, vy, vyaw] velocity commands
    - Invokes the low-level GO2 environment to execute motion
    """

    def __init__(self, base_env: GO2Robot, cfg):
        self.base_env = base_env
        self.cfg = cfg
        self.device = base_env.device
        self.num_envs = base_env.num_envs

        # Forward vector used to compute heading
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float)

        # Precompute region centers relative to each environment origin（兼容多障碍物）
        unsafe_positions = getattr(self.cfg, "unsafe_spheres_pos", None)
        if unsafe_positions is not None and len(unsafe_positions) > 0:
            unsafe_positions = torch.tensor(unsafe_positions, device=self.device, dtype=torch.float)
        else:
            unsafe_positions = torch.tensor(
                [self.cfg.unsafe_sphere_pos], device=self.device, dtype=torch.float
            )
        self._unsafe_pos_base = unsafe_positions  # [num_obstacles, 3]
        self.num_obstacles = self._unsafe_pos_base.shape[0]
        self._target_pos_base = torch.tensor(
            self.cfg.target_sphere_pos, device=self.device, dtype=torch.float
        )

        # Observation bookkeeping
        self._base_obs_dim = 8  # base features (heading, body vel, yaw rate, target dist & direction)
        self.use_manual_lidar = getattr(self.cfg, "enable_manual_lidar", True)
        self.lidar_num_bins = getattr(self.cfg, "lidar_num_bins", 16)
        self.target_lidar_num_bins = getattr(self.cfg, "target_lidar_num_bins", self.lidar_num_bins)
        self.target_lidar_max_range = getattr(self.cfg, "target_lidar_max_range", getattr(self.cfg, "lidar_max_range", 10.0))
        lidar_feature_dim = self.lidar_num_bins if self.use_manual_lidar else 0
        target_feature_dim = self.target_lidar_num_bins if self.target_lidar_num_bins > 0 else 0
        self.num_high_level_obs = self._base_obs_dim + target_feature_dim + lidar_feature_dim
        self.num_high_level_actions = 3  # [vx, vy, vyaw]
        boundary_extents = getattr(self.cfg, "boundary_half_extents", (3.0, 3.0))
        self.boundary_half_extents = torch.tensor(boundary_extents, device=self.device, dtype=torch.float)
        self.boundary_margin = getattr(self.cfg, "boundary_margin", 0.25)
        if self.use_manual_lidar and self.lidar_num_bins > 0:
            bin_size = 2 * math.pi / float(self.lidar_num_bins)
            angles = torch.arange(self.lidar_num_bins, device=self.device, dtype=torch.float) * bin_size - math.pi + 0.5 * bin_size
            self._lidar_dir_body = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        else:
            self._lidar_dir_body = None

        # High-level observation buffer
        self.high_level_obs_buf = torch.zeros(
            self.num_envs, self.num_high_level_obs,
            dtype=torch.float, device=self.device
        )

        # 动作限制
        self.action_scale = getattr(cfg, 'action_scale', [1.0, 1.0, 1.0])  # [vx, vy, vyaw] scaling
        self.action_scale = torch.tensor(self.action_scale, device=self.device, dtype=torch.float)

    def reset(self):
        """
        Reset the environment and return initial observations plus g/h values.

        Returns:
            high_level_obs: [num_envs, 7] 高层观测（含目标/危险区距离）
            g_values: [num_envs] initial g-function values
            h_values: [num_envs] initial h-function values
        """
        # 重置底层环境
        base_obs = self.base_env.reset()

        # 计算高层观测
        self._compute_high_level_observations()

        # Compute initial g/h values
        initial_avoid_metric, initial_reach_metric = self._get_current_metrics()
        initial_g_values = self._compute_g_function(initial_reach_metric)
        initial_h_values = self._compute_h_function(initial_avoid_metric)

        return self.high_level_obs_buf, initial_g_values, initial_h_values

    def update_velocity_commands(self, high_level_actions):
        """
        将高层动作转换为速度命令并更新到底层环境

        Args:
            high_level_actions: [num_envs, 3] 速度命令 [vx, vy, vyaw]
        """
        # 将高层动作转换为底层环境的速度命令
        high_level_actions = torch.clip(high_level_actions, -1.0, 1.0)
        velocity_commands = high_level_actions * self.action_scale

        # 更新底层环境的速度命令
        self.base_env.commands[:, 0] = velocity_commands[:, 0] * 0.6  # vx
        self.base_env.commands[:, 1] = velocity_commands[:, 1] * 0.2 # vy
        self.base_env.commands[:, 2] = velocity_commands[:, 2] * 0.8# vyaw

    def compute_g_h_values(self, avoid_metric, reach_metric):
        """
        Compute g/h values based on safety and reach metrics

        Args:
            avoid_metric: 避障指标
            reach_metric: 到达指标
        Returns:
            g_values: [num_envs] g-function values
            h_values: [num_envs] h-function values
        """
        g_values = self._compute_g_function(reach_metric)
        h_values = self._compute_h_function(avoid_metric)
        return g_values, h_values

    def _compute_high_level_observations(self):
        """Compute high-level observations including target/unsafe distances."""
        # Fetch robot position in world coordinates
        robot_pos = self.base_env.base_pos  # [num_envs, 3]

        # 计算朝向
        forward_vec_expanded = self.forward_vec.unsqueeze(0).expand(self.num_envs, -1)  # [num_envs, 3]
        forward = quat_apply(self.base_env.base_quat, forward_vec_expanded)  # [num_envs, 3]
        heading_cos = forward[:, 0]  # x分量
        heading_sin = forward[:, 1]  # y分量

        # 组织高层观测
        self.high_level_obs_buf[:, 0] = heading_cos       # cos(heading)
        self.high_level_obs_buf[:, 1] = heading_sin       # sin(heading)

        base_lin_vel_body = self.base_env.base_lin_vel  # body-frame linear velocity
        lin_vel_scale = 2.0  # match low-level obs_scales.lin_vel
        self.high_level_obs_buf[:, 2] = torch.clamp(base_lin_vel_body[:, 0] * lin_vel_scale, -1.0, 1.0)
        self.high_level_obs_buf[:, 3] = torch.clamp(base_lin_vel_body[:, 1] * lin_vel_scale, -1.0, 1.0)

        base_ang_vel_body = self.base_env.base_ang_vel  # body-frame angular velocity
        ang_vel_scale = 0.25  # match low-level obs_scales.ang_vel
        self.high_level_obs_buf[:, 4] = torch.clamp(base_ang_vel_body[:, 2] * ang_vel_scale, -1.0, 1.0)

        env_origins = self.base_env.env_origins
        target_centers = env_origins + self._target_pos_base
        obstacle_centers = env_origins.unsqueeze(1) + self._unsafe_pos_base.unsqueeze(0)
        obstacle_rel = obstacle_centers - robot_pos.unsqueeze(1)
        rel_pos_xy = robot_pos[:, :2] - env_origins[:, :2]

        target_rel = target_centers - robot_pos
        target_rel_xy = target_rel[:, :2]
        target_distance = torch.norm(target_rel_xy, dim=1)
        lidar_max_range = getattr(self.cfg, "lidar_max_range", 10.0)
        target_distance_norm = torch.clamp(target_distance / lidar_max_range, 0.0, 1.0)
        heading_cos_exp = heading_cos.unsqueeze(1)
        heading_sin_exp = heading_sin.unsqueeze(1)
        target_rel_x_body = target_rel_xy[:, 0] * heading_cos + target_rel_xy[:, 1] * heading_sin
        target_rel_y_body = -target_rel_xy[:, 0] * heading_sin + target_rel_xy[:, 1] * heading_cos
        target_body_vec = torch.stack((target_rel_x_body, target_rel_y_body), dim=-1)
        target_body_norm = torch.norm(target_body_vec, dim=1, keepdim=True).clamp_min(1e-6)
        target_dir_body = target_body_vec / target_body_norm

        dist_to_unsafe_all = torch.norm(obstacle_rel, dim=2)
        dist_to_unsafe = torch.min(dist_to_unsafe_all, dim=1)[0]

        # Normalize distance similar to lidar features to keep scale consistent
        self.high_level_obs_buf[:, 5] = target_distance_norm
        self.high_level_obs_buf[:, 6] = target_dir_body[:, 0]
        self.high_level_obs_buf[:, 7] = target_dir_body[:, 1]

        # Smooth target proximity encoding across body-frame bins
        target_start = self._base_obs_dim
        if self.target_lidar_num_bins > 0:
            target_surface_dist = torch.clamp(target_distance - self.cfg.target_radius, min=0.0)
            target_intensity = 1.0 - torch.clamp(target_surface_dist / self.target_lidar_max_range, 0.0, 1.0)
            target_angles = torch.atan2(target_dir_body[:, 1], target_dir_body[:, 0])
            normalized_angles = (target_angles + math.pi) / (2 * math.pi)
            scaled_bins = normalized_angles * self.target_lidar_num_bins
            floored = torch.floor(scaled_bins)
            frac = (scaled_bins - floored).clamp(0.0, 1.0)
            lower_bins = torch.remainder(floored.long(), self.target_lidar_num_bins)
            upper_bins = (lower_bins + 1) % self.target_lidar_num_bins
            lower_weights = (1.0 - frac) * target_intensity
            upper_weights = frac * target_intensity
            target_lidar_buf = torch.zeros(self.num_envs, self.target_lidar_num_bins, device=self.device)
            env_indices = torch.arange(self.num_envs, device=self.device)
            target_lidar_buf[env_indices, lower_bins] = torch.maximum(
                target_lidar_buf[env_indices, lower_bins], lower_weights
            )
            target_lidar_buf[env_indices, upper_bins] = torch.maximum(
                target_lidar_buf[env_indices, upper_bins], upper_weights
            )
            self.high_level_obs_buf[:, target_start:target_start + self.target_lidar_num_bins] = target_lidar_buf
        else:
            target_lidar_buf = None

        if self.use_manual_lidar and self.lidar_num_bins > 0:
            rel_xy = obstacle_rel[:, :, :2]
            heading_cos_exp = heading_cos.unsqueeze(1)
            heading_sin_exp = heading_sin.unsqueeze(1)
            rel_x_body = rel_xy[:, :, 0] * heading_cos_exp + rel_xy[:, :, 1] * heading_sin_exp
            rel_y_body = -rel_xy[:, :, 0] * heading_sin_exp + rel_xy[:, :, 1] * heading_cos_exp
            planar_dist = torch.sqrt(rel_x_body ** 2 + rel_y_body ** 2 + 1e-9)
            surface_dist = torch.clamp(planar_dist - self.cfg.unsafe_sphere_radius, min=0.0)
            lidar_max_range = getattr(self.cfg, "lidar_max_range", 10.0)
            normalized_dist = torch.clamp(surface_dist / lidar_max_range, 0.0, 1.0)
            intensity = 1.0 - normalized_dist
            bin_size = 2 * math.pi / float(self.lidar_num_bins)
            angles = torch.atan2(rel_y_body, rel_x_body)
            bin_indices = torch.floor((angles + math.pi) / bin_size).long()
            bin_indices = torch.clamp(bin_indices, min=0, max=self.lidar_num_bins - 1)

            lidar_buf = torch.zeros(self.num_envs, self.lidar_num_bins, device=self.device)
            env_indices = torch.arange(self.num_envs, device=self.device)
            if self.num_obstacles > 0:
                for obs_idx in range(self.num_obstacles):
                    bins = bin_indices[:, obs_idx]
                    vals = intensity[:, obs_idx]
                    lidar_buf[env_indices, bins] = torch.maximum(lidar_buf[env_indices, bins], vals)
            boundary_intensity = self._compute_boundary_lidar(rel_pos_xy, heading_cos, heading_sin)
            if boundary_intensity is not None:
                lidar_buf = torch.maximum(lidar_buf, boundary_intensity)
            lidar_start = target_start + (self.target_lidar_num_bins if self.target_lidar_num_bins > 0 else 0)
            self.high_level_obs_buf[:, lidar_start:lidar_start + self.lidar_num_bins] = lidar_buf


    def _compute_g_function(self, reach_metric):
        """
        计算目标函数g(x)
        Args:
            reach_metric: 到目标中心的距离
        Returns:
            g_values: [num_envs] g-function values
        """
        # 判断是否在目标集合内
        distance_from_boundary = reach_metric - self.cfg.target_radius
        in_goal = distance_from_boundary <= 0

        # Compute g-function values using config parameters
        goal_value = torch.full_like(reach_metric, self.cfg.g_target_value)
        outside_value = self.cfg.g_distance_scale * distance_from_boundary
        g_values = torch.where(in_goal, goal_value, outside_value)

        return g_values

    def _compute_h_function(self, avoid_metric):
        """
        计算安全函数h(x)
        Args:
            avoid_metric: safety metric (positive inside unsafe region)
        Returns:
            h_values: [num_envs] h-function values
        """
        # Determine whether the state is unsafe
        in_unsafe = avoid_metric > 0

        # Compute h-function values using config parameters
        unsafe_value = torch.full_like(avoid_metric, self.cfg.h_unsafe_value)
        safe_value = torch.full_like(avoid_metric, self.cfg.h_safe_value)
        h_values = torch.where(in_unsafe, unsafe_value, safe_value)

        return h_values

    def _get_current_metrics(self):
        """
        Retrieve current avoidance and reach metrics
        This triggers the base environment to recompute safety metrics
        Returns:
            current_avoid_metric: 当前状态的避障指标
            current_reach_metric: 当前状态的到达指标
        """
        # Call the base environment safety metric helper
        self.base_env._compute_safety_metrics()
        return self.base_env.avoid_metric.clone(), self.base_env.reach_metric.clone()

    def get_observations(self):
        """返回当前高层观测"""
        return self.high_level_obs_buf

    def get_base_observations(self):
        """返回底层观测"""
        return self.base_env.get_observations()

    def _compute_boundary_lidar(self, rel_pos_xy, heading_cos, heading_sin):
        if not self.use_manual_lidar or self.lidar_num_bins <= 0 or self._lidar_dir_body is None:
            return None
        half_length = self.boundary_half_extents[0]
        half_width = self.boundary_half_extents[1]
        dir_body = self._lidar_dir_body  # [num_bins, 2]
        dir_world_x = heading_cos.unsqueeze(1) * dir_body[:, 0].unsqueeze(0) - heading_sin.unsqueeze(1) * dir_body[:, 1].unsqueeze(0)
        dir_world_y = heading_sin.unsqueeze(1) * dir_body[:, 0].unsqueeze(0) + heading_cos.unsqueeze(1) * dir_body[:, 1].unsqueeze(0)
        rel_x = rel_pos_xy[:, 0].unsqueeze(1)
        rel_y = rel_pos_xy[:, 1].unsqueeze(1)
        inf = torch.full_like(dir_world_x, float("inf"))
        eps = 1e-6

        tx_pos = torch.where(dir_world_x > eps, (half_length - rel_x) / dir_world_x, inf)
        tx_neg = torch.where(dir_world_x < -eps, (-half_length - rel_x) / dir_world_x, inf)
        ty_pos = torch.where(dir_world_y > eps, (half_width - rel_y) / dir_world_y, inf)
        ty_neg = torch.where(dir_world_y < -eps, (-half_width - rel_y) / dir_world_y, inf)

        t_candidates = torch.stack((tx_pos, tx_neg, ty_pos, ty_neg), dim=-1)
        t_candidates = torch.where(t_candidates > 0, t_candidates, inf.unsqueeze(-1))
        boundary_dist = torch.min(t_candidates, dim=-1).values
        lidar_max_range = getattr(self.cfg, "lidar_max_range", 10.0)
        surface_dist = torch.clamp(boundary_dist, min=0.0)
        normalized = torch.clamp(surface_dist / lidar_max_range, 0.0, 1.0)
        intensity = 1.0 - normalized
        intensity = torch.where(torch.isinf(boundary_dist), torch.zeros_like(intensity), intensity)
        return intensity


class HighLevelNavigationConfig:
    """高层导航环境配置"""

    def __init__(self):
        # 动作缩放
        self.action_scale = [1.0, 1.0, 1.0]  # [vx, vy, vyaw]

        # Target and unsafe region configuration (loaded from base_env.cfg)
        self.target_radius = 0.4
        self.target_sphere_pos = [4.0, 0.0, 0.4]
        self.unsafe_sphere_pos = [2.0, 0.0, 0.4]  # Single obstacle (backward compatibility)
        self.unsafe_spheres_pos = None  # Multiple obstacles (if configured)
        self.unsafe_sphere_radius = 0.3

        # reach-avoid函数参数
        self.g_target_value = -300.0      # g value inside target region
        self.g_distance_scale = 100.0     # scaling factor for distance-based g value
        self.g_distance_offset = 0.0      # g value offset (applied after scaling)

        self.h_safe_value = -300.0         # h value in safe region
        self.h_unsafe_value = 300.0        # h value in unsafe region

        # 手动“激光雷达”相关参数
        self.enable_manual_lidar = True
        self.lidar_max_range = 10.0
        self.lidar_num_bins = 16
        self.boundary_half_extents = (3.0, 3.0)
        self.boundary_margin = 0.25



