from legged_gym.envs.base.legged_robot import LeggedRobot
import numpy as np
import math
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch


class GO2Robot(LeggedRobot):

    def reset(self):
        """ Reset all robots"""
        # print("test1")
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs

    def step(self, actions):
        """
        Step wrapper for the GO2 environment.
        """
        return super().step(actions)

    def check_termination(self):
        """Extend termination conditions with safety violations and goal reaching."""
        super().check_termination()
        self._compute_safety_metrics()

        if getattr(self.cfg.rewards_ext, "terminate_on_reach_avoid", False):
            collision_dist = getattr(self.cfg.rewards_ext, "collision_dist", None)
            if collision_dist is not None:
                # Align termination with reward collision logic.
                safety_violation = self.min_hazard_distance < collision_dist
            else:
                safety_violation = self.avoid_metric > 0.0
            goal_reached_dist = getattr(
                self.cfg.rewards_ext, "goal_reached_dist", self.cfg.rewards_ext.target_sphere_radius
            )
            reached = self.reach_metric <= goal_reached_dist
            self.reset_buf |= safety_violation | reached

    def compute_observations(self):
        """ Computes observations  
        """
        self.obs_buf = torch.cat((  
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _init_buffers(self):
        super()._init_buffers()
        # Initialize buffers for our custom safety metrics
        self.avoid_metric = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reach_metric = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # Tracks distance to the nearest hazard surface (obstacle/boundary) for reward shaping.
        self.min_hazard_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # Tracks obstacle and boundary distances separately for diagnostics.
        self.obstacle_surface_distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.boundary_distance = torch.full((self.num_envs,), float("inf"), device=self.device, dtype=torch.float)
        # get rigid body states
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)

    def _reset_root_states(self, env_ids):
        """Overrides base reset to sample initial xy within the full 6x6 arena."""
        if len(env_ids) == 0:
            return

        super()._reset_root_states(env_ids)

        terrain_length = getattr(self.cfg.terrain, "terrain_length", 6.0)
        terrain_width = getattr(self.cfg.terrain, "terrain_width", 6.0)
        half_length = max(terrain_length * 0.5, 0.0)
        half_width = max(terrain_width * 0.5, 0.0)
        spawn_range_x = half_length
        spawn_range_y = half_width

        env_ids = env_ids.to(self.device)
        num_ids = len(env_ids)
        env_origins = self.env_origins[env_ids]

        spawn_clearance = getattr(self.cfg.rewards_ext, "spawn_clearance", None)
        if spawn_clearance is None:
            spawn_clearance = getattr(self.cfg.rewards_ext, "collision_dist", 0.4)
        spawn_clearance = float(spawn_clearance)
        max_tries = int(getattr(self.cfg.rewards_ext, "spawn_max_tries", 10))

        rand_x = torch_rand_float(-spawn_range_x, spawn_range_x, (num_ids, 1), device=self.device)
        rand_y = torch_rand_float(-spawn_range_y, spawn_range_y, (num_ids, 1), device=self.device)
        pos_xy = torch.cat((rand_x, rand_y), dim=1)

        unsafe_spheres_pos_base = None
        if hasattr(self.cfg.rewards_ext, "unsafe_spheres_pos") and len(self.cfg.rewards_ext.unsafe_spheres_pos) > 0:
            unsafe_spheres_pos_base = torch.tensor(
                self.cfg.rewards_ext.unsafe_spheres_pos,
                device=self.device,
                dtype=torch.float,
            )
        else:
            unsafe_pos = getattr(self.cfg.rewards_ext, "unsafe_sphere_pos", None)
            if unsafe_pos is not None:
                unsafe_spheres_pos_base = torch.tensor(
                    [unsafe_pos], device=self.device, dtype=torch.float
                )

        for _ in range(max_tries):
            invalid = torch.zeros(num_ids, device=self.device, dtype=torch.bool)
            if half_length > 0.0 and half_width > 0.0:
                boundary_gap_x = half_length - torch.abs(pos_xy[:, 0])
                boundary_gap_y = half_width - torch.abs(pos_xy[:, 1])
                boundary_distance = torch.min(boundary_gap_x, boundary_gap_y)
                invalid |= boundary_distance < spawn_clearance
            if unsafe_spheres_pos_base is not None:
                obstacle_centers = env_origins.unsqueeze(1) + unsafe_spheres_pos_base.unsqueeze(0)
                obstacle_rel = pos_xy.unsqueeze(1) - obstacle_centers[:, :, :2]
                obstacle_dist = torch.norm(obstacle_rel, dim=2)
                obstacle_surface = obstacle_dist - float(self.cfg.rewards_ext.unsafe_sphere_radius)
                invalid |= obstacle_surface.min(dim=1).values < spawn_clearance
            if not invalid.any():
                break
            resample = invalid.nonzero(as_tuple=False).squeeze(-1)
            if resample.numel() == 0:
                break
            pos_xy[resample, 0] = torch_rand_float(
                -spawn_range_x, spawn_range_x, (resample.numel(), 1), device=self.device
            ).squeeze(-1)
            pos_xy[resample, 1] = torch_rand_float(
                -spawn_range_y, spawn_range_y, (resample.numel(), 1), device=self.device
            ).squeeze(-1)

        self.root_states[env_ids, 0] = pos_xy[:, 0] + env_origins[:, 0]
        self.root_states[env_ids, 1] = pos_xy[:, 1] + env_origins[:, 1]

        # randomize yaw so rollouts explore diverse headings
        random_yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).squeeze(-1)
        half_yaw = 0.5 * random_yaw
        delta_quat = torch.zeros(len(env_ids), 4, device=self.device)
        delta_quat[:, 2] = torch.sin(half_yaw)
        delta_quat[:, 3] = torch.cos(half_yaw)
        base_quat = self.base_init_state[3:7].unsqueeze(0).repeat(len(env_ids), 1)
        randomized_quat = quat_mul(delta_quat, base_quat)
        self.root_states[env_ids, 3:7] = randomized_quat

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # def _draw_debug_visualization(self):
    #     """
    #     专门用于绘制调试信息的函数.
    #     这个版本修正了 'add_lines' 的参数类型，以匹配 Isaac Gym API 的要求。
    #     """
    #     # 只有在有 viewer (即非 headless 模式) 时才执行
    #     if self.viewer:
    #         # 清除上一帧绘制的所有线条
    #         self.gym.clear_lines(self.viewer)
            
    #         # 定义颜色
    #         unsafe_color = gymapi.Vec3(1.0, 0.1, 0.1)  # 红色
    #         target_color = gymapi.Vec3(0.1, 1.0, 0.1)  # 绿色

    #         for i in range(self.num_envs):
    #             # --- 处理不安全区域的可视化 ---
    #             unsafe_pos_tensor = torch.tensor(self.cfg.rewards_ext.unsafe_sphere_pos, device=self.device) + self.env_origins[i]
    #             unsafe_pos = gymapi.Vec3(unsafe_pos_tensor[0], unsafe_pos_tensor[1], unsafe_pos_tensor[2])
    #             radius = self.cfg.rewards_ext.unsafe_sphere_radius

    #             # 创建三维十字标记的顶点 (Python列表)
    #             verts_list = [
    #                 gymapi.Vec3(unsafe_pos.x - radius, unsafe_pos.y, unsafe_pos.z), gymapi.Vec3(unsafe_pos.x + radius, unsafe_pos.y, unsafe_pos.z),
    #                 gymapi.Vec3(unsafe_pos.x, unsafe_pos.y - radius, unsafe_pos.z), gymapi.Vec3(unsafe_pos.x, unsafe_pos.y + radius, unsafe_pos.z),
    #                 gymapi.Vec3(unsafe_pos.x, unsafe_pos.y, unsafe_pos.z - radius), gymapi.Vec3(unsafe_pos.x, unsafe_pos.y, unsafe_pos.z + radius)
    #             ]
                
    #             # *** 关键步骤：将Python列表转换为NumPy数组 ***
    #             verts_np = np.array(verts_list, dtype=gymapi.Vec3)
    #             colors_np = np.array([unsafe_color] * len(verts_list), dtype=gymapi.Vec3)
                
    #             # 使用正确类型的参数调用API
    #             self.gym.add_lines(self.viewer, self.envs[i], len(verts_list) // 2, verts_np, colors_np)

    #             # --- 处理目标区域的可视化 ---
    #             target_pos_tensor = torch.tensor(self.cfg.rewards_ext.target_sphere_pos, device=self.device) + self.env_origins[i]
    #             target_pos = gymapi.Vec3(target_pos_tensor[0], target_pos_tensor[1], target_pos_tensor[2])
    #             radius = self.cfg.rewards_ext.target_sphere_radius

    #             # 创建三维十字标记的顶点 (Python列表)
    #             verts_list = [
    #                 gymapi.Vec3(target_pos.x - radius, target_pos.y, target_pos.z), gymapi.Vec3(target_pos.x + radius, target_pos.y, target_pos.z),
    #                 gymapi.Vec3(target_pos.x, target_pos.y - radius, target_pos.z), gymapi.Vec3(target_pos.x, target_pos.y + radius, target_pos.z),
    #                 gymapi.Vec3(target_pos.x, target_pos.y, target_pos.z - radius), gymapi.Vec3(target_pos.x, target_pos.y, target_pos.z + radius)
    #             ]

    #             # *** 关键步骤：将Python列表转换为NumPy数组 ***
    #             verts_np = np.array(verts_list, dtype=gymapi.Vec3)
    #             colors_np = np.array([target_color] * len(verts_list), dtype=gymapi.Vec3)
                
    #             # 使用正确类型的参数调用API
    #             self.gym.add_lines(self.viewer, self.envs[i], len(verts_list) // 2, verts_np, colors_np)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
  
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec


    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    def _reward_hip_pos(self):
        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        return (0.8-torch.abs(self.commands[:,1])) *  torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_num= torch.sum(contact,dim=1)
        reward = torch.where(contact_num <=2, 0, 1)
        return reward* (torch.norm(self.commands[:, :2], dim=1) > 0.1)
    def _reward_orientation_eular(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        #quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return orientation#(quat_mismatch + orientation) / 2.
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 100), dim=1)
    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    
    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update
    def _reward_no_fly(self):
        # reward one-foot contact when moving
        contacts = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        double_contact = torch.sum(1.*contacts, dim=1)==2
        return double_contact*(torch.norm(self.commands[:, :2], dim=1) > self.cfg.rewards.command_dead)#1.*((self.time_to_stand_still <= self.static_delay) & single_contact)
    
    def _compute_safety_metrics(self):
        """ Computes the avoid and reach metrics and stores them on self. """
        # Avoid metric: penalty for being inside ANY unsafe sphere (use minimum distance)

        # Check if using multiple obstacles or single obstacle
        radius_scale = getattr(self.cfg.rewards_ext, "unsafe_radius_h_eval_scale", 1.0)
        effective_radius = self.cfg.rewards_ext.unsafe_sphere_radius * radius_scale

        if hasattr(self.cfg.rewards_ext, 'unsafe_spheres_pos') and len(self.cfg.rewards_ext.unsafe_spheres_pos) > 0:
            # Multiple obstacles mode
            unsafe_spheres_pos_base = torch.tensor(
                self.cfg.rewards_ext.unsafe_spheres_pos,
                device=self.device,
                dtype=torch.float
            )  # [num_obstacles, 3]
            num_obstacles = unsafe_spheres_pos_base.shape[0]

            # Add environment origins: [num_envs, 1, 3] + [1, num_obstacles, 3] -> [num_envs, num_obstacles, 3]
            unsafe_spheres_pos = self.env_origins.unsqueeze(1) + unsafe_spheres_pos_base.unsqueeze(0)

            # Compute distance from agent to each obstacle: [num_envs, num_obstacles]
            dist_to_unsafe_all = torch.norm(
                self.base_pos.unsqueeze(1) - unsafe_spheres_pos,
                dim=2
            )

            # Take minimum distance (closest obstacle determines avoid metric)
            dist_to_unsafe = torch.min(dist_to_unsafe_all, dim=1)[0]  # [num_envs]
        else:
            # Single obstacle mode (backward compatibility)
            unsafe_pos_base = torch.tensor(
                self.cfg.rewards_ext.unsafe_sphere_pos,
                device=self.device,
                dtype=torch.float
            )
            unsafe_pos = unsafe_pos_base.unsqueeze(0) + self.env_origins  # [num_envs, 3]
            dist_to_unsafe = torch.norm(self.base_pos - unsafe_pos, dim=1)

        # The 'avoid' metric is a cost that is > 0 inside the sphere.
        avoid_metric = torch.clamp(effective_radius - dist_to_unsafe, min=0.)
        # Compute distance to the obstacle surface for reward shaping (can be negative if inside).
        reward_radius_scale = getattr(self.cfg.rewards_ext, "unsafe_radius_reward_scale", 1.0)
        reward_radius = self.cfg.rewards_ext.unsafe_sphere_radius * reward_radius_scale
        obstacle_surface_distance = dist_to_unsafe - reward_radius

        terrain_length = getattr(self.cfg.terrain, "terrain_length", 6.0)
        terrain_width = getattr(self.cfg.terrain, "terrain_width", 6.0)
        half_length = max(terrain_length * 0.5, 0.0)
        half_width = max(terrain_width * 0.5, 0.0)
        boundary_distance = torch.full((self.num_envs,), float("inf"), device=self.device, dtype=torch.float)
        if half_length > 0.0 and half_width > 0.0:
            rel_pos_xy = self.base_pos[:, :2] - self.env_origins[:, :2]
            gap_x = half_length - torch.abs(rel_pos_xy[:, 0])
            gap_y = half_width - torch.abs(rel_pos_xy[:, 1])
            boundary_distance = torch.min(gap_x, gap_y)
            boundary_margin = getattr(self.cfg.rewards_ext, "boundary_margin", 0.25)
            boundary_metric = torch.clamp(boundary_margin - boundary_distance, min=0.0)
            avoid_metric = torch.maximum(avoid_metric, boundary_metric)
            outside = boundary_distance < 0.0
            if outside.any():
                self.reset_buf[outside] = 1

        # Use the nearest hazard (obstacle surface or boundary) for reward shaping.
        self.min_hazard_distance = torch.minimum(obstacle_surface_distance, boundary_distance)
        self.obstacle_surface_distance = obstacle_surface_distance
        self.boundary_distance = boundary_distance

        # Reach metric: distance to the target sphere center (only x,y plane)
        target_pos_base = torch.tensor(self.cfg.rewards_ext.target_sphere_pos, device=self.device, dtype=torch.float)
        # 为每个环境添加环境原点偏移
        target_pos = target_pos_base.unsqueeze(0) + self.env_origins  # [num_envs, 3]
        # 只计算x,y平面的距离，忽略z轴
        base_pos_xy = self.base_pos[:, :2]  # [num_envs, 2]
        target_pos_xy = target_pos[:, :2]   # [num_envs, 2]
        reach_metric = torch.norm(base_pos_xy - target_pos_xy, dim=1)

        # Store metrics as class attributes
        self.avoid_metric = avoid_metric
        self.reach_metric = reach_metric
