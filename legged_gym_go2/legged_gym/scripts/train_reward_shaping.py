#!/usr/bin/env python3
import os
import time
from datetime import datetime
from typing import Tuple

import isaacgym
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.scripts.train_reach_avoid import (
    HierarchicalVecEnv,
    create_env,
)
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCritic


# 计算奖励整形并生成到达/安全标记。
def _compute_reward_shaping(
    reach_metric: torch.Tensor,  # 到目标的距离（当前步）。
    prev_goal_dist: torch.Tensor,  # 上一步到目标的距离。
    min_hazard_distance: torch.Tensor,  # 到最近危险区域表面的距离。
    velocity_commands: torch.Tensor,  # 当前速度命令（已按高层限制裁剪/缩放）。
    target_dir_body: torch.Tensor,  # 目标方向（机体坐标系，单位向量）。
    time_outs: torch.Tensor,  # 超时终止标记。
    reward_cfg,  # 奖励配置。
    rewards_ext,  # 奖励扩展配置。
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    # 参考环境的奖励定义：终止给大信号，否则用动作和最近障碍距离。
    goal_reached_dist = float(
        getattr(reward_cfg, "goal_reached_dist", rewards_ext.target_sphere_radius)
    )
    collision_dist = float(getattr(reward_cfg, "collision_dist", 0.35))
    obstacle_avoid_dist = float(getattr(reward_cfg, "obstacle_avoid_dist", 1.0))
    forward_scale = float(getattr(reward_cfg, "forward_reward_scale", 0.5))
    yaw_scale = float(getattr(reward_cfg, "yaw_penalty_scale", 0.5))
    obstacle_scale = float(getattr(reward_cfg, "obstacle_penalty_scale", 0.5))
    progress_scale = float(getattr(reward_cfg, "goal_progress_scale", 1.0))

    # 终止分类：collision / success / time_out。
    reached = reach_metric <= goal_reached_dist
    collision = min_hazard_distance < collision_dist
    terminated = reached | collision
    truncated = time_outs & (~terminated)

    # 基础奖励项：朝目标方向的速度投影 + 角度误差惩罚 + 近障惩罚（终止逻辑保持不变）。
    # 公式: forward_scale * dot(v_cmd_xy, target_dir_body)
    #     - yaw_scale * angle_error(command_dir, target_dir_body) * I[|v_cmd_xy| > 1e-3]
    #     - obstacle_scale * r3(min_hazard_distance)
    min_laser = torch.clamp(min_hazard_distance, min=0.0)
    r3 = torch.clamp((obstacle_avoid_dist - min_laser) / obstacle_avoid_dist, min=0.0)
    target_dir = target_dir_body / target_dir_body.norm(dim=1, keepdim=True).clamp_min(1e-6)
    command_xy = velocity_commands[:, :2]
    command_speed = torch.norm(command_xy, dim=1)
    command_dir = command_xy / command_speed.unsqueeze(1).clamp_min(1e-6)
    target_proj = torch.sum(command_xy * target_dir, dim=1)
    cos_angle = torch.sum(command_dir * target_dir, dim=1).clamp(-1.0, 1.0)
    angle_error = torch.acos(cos_angle)
    angle_mask = (command_speed > 1e-3).float()
    reward = forward_scale * target_proj - yaw_scale * angle_error * angle_mask - obstacle_scale * r3

    # 目标进度奖励：鼓励距离减少，终止步不计入进度。
    progress = prev_goal_dist - reach_metric
    progress = progress * (~(terminated | truncated)).float()
    reward = reward + progress_scale * progress

    # 终止奖励：到达/碰撞/超时。
    reward = torch.where(
        reached,
        reward + float(getattr(reward_cfg, "success_reward", 100.0)),
        reward,
    )
    reward = torch.where(
        collision,
        reward - float(getattr(reward_cfg, "collision_penalty", 100.0)),
        reward,
    )
    reward = torch.where(
        truncated,
        reward - float(getattr(reward_cfg, "timeout_penalty", 0.0)),
        reward,
    )

    reward_scale = float(getattr(reward_cfg, "reward_scale", 1.0))
    reward = reward * reward_scale
    reward_clip = float(getattr(reward_cfg, "reward_clip", 0.0))
    if reward_clip > 0.0:
        reward = torch.clamp(reward, -reward_clip, reward_clip)

    components = {
        "target_proj": target_proj,
        "angle_error": angle_error,
        "obstacle_penalty": r3,
        "progress": progress,
        "command_speed": command_speed,
    }

    return reward, reached, collision, terminated, truncated, components


# 从环境信息中提取 PPO 需要的字段。
def _build_ppo_infos(
    infos: dict  # 环境 step 返回的 infos 字典。
) -> dict:  # 返回整理后的 PPO infos 字典。
    # 函数作用: 只保留 PPO 训练需要的超时标记字段。
    # 输入 infos: 环境返回的信息字典。
    # 输出: 仅包含 PPO 关注字段的字典。
    ppo_infos = {}  # 初始化返回字典。
    # 确保 infos 为字典类型。
    if isinstance(infos, dict):
        # 从 infos 中提取 base_infos 子字典。
        base_infos = infos.get("base_infos")
        # 若 base_infos 合法且包含 time_outs 字段, 则拷贝给 PPO。
        if isinstance(base_infos, dict) and "time_outs" in base_infos:
            ppo_infos["time_outs"] = base_infos["time_outs"]
    # 返回整理后的 PPO infos。
    return ppo_infos


def train_reward_shaping(args) -> None:
    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()

    train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]
    train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]

    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    if train_cfg.runner.experiment_name == "high_level_go2":
        train_cfg.runner.experiment_name = "high_level_go2_reward_shaping"

    device = torch.device(args.rl_device)
    env = create_env(env_cfg, train_cfg, args, device)

    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std,
        action_squash="tanh",
    ).to(device)

    alg = PPO(
        actor_critic=actor_critic,
        device=device,
        **train_cfg.algorithm.__dict__,
    )
    alg.init_storage(
        num_envs=env.num_envs,
        num_transitions_per_env=train_cfg.algorithm.num_steps_per_env,
        actor_obs_shape=(env.num_obs,),
        critic_obs_shape=(env.num_obs,),
        action_shape=(env.num_actions,),
    )

    start_iteration = 0
    log_dir = None

    if getattr(train_cfg.runner, "resume", False):
        resume_path = getattr(train_cfg.runner, "resume_path", "")
        if resume_path and os.path.isfile(resume_path):
            log_dir = os.path.dirname(resume_path)
            print(f"resuming from checkpoint: {resume_path}")
            print(f"  using existing log directory: {log_dir}")

            checkpoint = torch.load(resume_path, map_location=device)
            actor_state = checkpoint.get("actor_critic")
            if actor_state is not None:
                actor_critic.load_state_dict(actor_state)
            opt_state = checkpoint.get("optimizer")
            if opt_state is not None:
                alg.optimizer.load_state_dict(opt_state)
            start_iteration = checkpoint.get("iteration", 0)
            print(f"  continuing from iteration {start_iteration}")
        else:
            raise FileNotFoundError(
                f"Resume enabled but checkpoint not found: {resume_path}. Please provide a valid checkpoint path."
            )

    if log_dir is None:
        log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        base_log_root = os.path.join(repo_root, "logs", "logs")
        log_dir = os.path.join(base_log_root, train_cfg.runner.experiment_name, log_timestamp)
        os.makedirs(log_dir, exist_ok=True)
        print(f"created new log directory: {log_dir}")
    else:
        os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "training.log")
    log_fp = open(log_file, "a", encoding="utf-8")
    print(f"training log file: {log_file}")

    reward_cfg = env_cfg.reward_shaping
    rewards_ext = env.env.base_env.cfg.rewards_ext

    print("Reward shaping PPO training")
    print(f"  envs       : {env.num_envs}")
    print(f"  obs dim    : {env.num_obs}")
    print(f"  action dim : {env.num_actions}")
    print(f"  horizon    : {train_cfg.algorithm.num_steps_per_env}")
    print(f"  device     : {device}")
    print(f"  log dir    : {log_dir}")

    obs, g_vals, h_vals = env.reset()
    obs = obs.to(device)
    g_vals = g_vals.to(device)
    h_vals = h_vals.to(device)
    prev_goal_dist = env.env.base_env.reach_metric.clone().to(device)
    horizon = train_cfg.algorithm.num_steps_per_env

    max_iterations = train_cfg.runner.max_iterations
    save_interval = train_cfg.runner.save_interval
    success_rate = 0.0
    interval_start = time.time()

    for iteration in range(start_iteration, max_iterations):
        episode_collision = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        episode_reached = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        episode_steps = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        success_count = 0
        reached_count = 0
        collision_count = 0
        timeout_count = 0
        episode_count = 0
        success_steps_sum = 0.0

        reward_sum = 0.0
        goal_dist_sum = 0.0
        min_hazard_sum = 0.0
        action_sat_sum = 0.0
        proj_sum = 0.0
        angle_error_sum = 0.0
        obstacle_penalty_sum = 0.0
        progress_sum = 0.0
        command_speed_sum = 0.0

        for step in range(horizon):
            actions = alg.act(obs, obs)
            next_obs, next_g, next_h, dones, infos = env.step(actions)

            next_obs = next_obs.to(device)
            next_g = next_g.to(device)
            next_h = next_h.to(device)
            dones = dones.to(device)

            # 基于危险距离与进度构造奖励（包含门控与终止项）。
            min_hazard_distance = env.env.base_env.min_hazard_distance.to(device)
            reach_metric = infos["reach_metric"].to(device)
            time_outs = torch.zeros_like(reach_metric, dtype=torch.bool)
            base_infos = infos.get("base_infos", {}) if isinstance(infos, dict) else {}
            if isinstance(base_infos, dict) and "time_outs" in base_infos:
                time_outs = base_infos["time_outs"].to(device).bool()

            # 奖励使用执行中的速度命令，并结合目标方向投影，避免与动作裁剪/缩放不一致。
            velocity_commands = env.env.base_env.commands[:, :3].to(device)
            target_dir_body = obs[:, 6:8].to(device)

            reward, reached, collision, _terminated, _truncated, components = _compute_reward_shaping(
                reach_metric=reach_metric,
                prev_goal_dist=prev_goal_dist,
                min_hazard_distance=min_hazard_distance,
                velocity_commands=velocity_commands,
                target_dir_body=target_dir_body,
                time_outs=time_outs,
                reward_cfg=reward_cfg,
                rewards_ext=rewards_ext,
            )

            done_flags = dones
            if getattr(reward_cfg, "terminate_on_safety_violation", True):
                done_flags = torch.logical_or(done_flags, collision)
            if getattr(reward_cfg, "terminate_on_success", True):
                done_flags = torch.logical_or(done_flags, reached)

            episode_steps += 1
            episode_reached |= reached
            episode_collision |= collision

            done_envs = done_flags
            if done_envs.any():
                # 成功率：先到达目标且期间未发生碰撞。
                success_mask = done_envs & episode_reached & ~episode_collision
                success_count += success_mask.sum().item()
                reached_count += (done_envs & reached).sum().item()
                collision_count += (done_envs & collision).sum().item()
                timeout_count += (done_envs & _truncated).sum().item()
                episode_count += done_envs.sum().item()
                if success_mask.any():
                    success_steps_sum += episode_steps[success_mask].float().sum().item()
                episode_collision[done_envs] = False
                episode_reached[done_envs] = False
                episode_steps[done_envs] = 0

            reward_sum += reward.mean().item()
            goal_dist_sum += reach_metric.mean().item()
            min_hazard_sum += min_hazard_distance.mean().item()
            action_sat_sum += (actions.abs() > 0.95).float().mean().item()
            proj_sum += components["target_proj"].mean().item()
            angle_error_sum += components["angle_error"].mean().item()
            obstacle_penalty_sum += components["obstacle_penalty"].mean().item()
            progress_sum += components["progress"].mean().item()
            command_speed_sum += components["command_speed"].mean().item()
            alg.process_env_step(reward, done_flags, _build_ppo_infos(infos))

            obs = next_obs
            g_vals = next_g
            h_vals = next_h
            prev_goal_dist = reach_metric.clone()

        alg.compute_returns(obs)
        value_loss, policy_loss, approx_kl, clip_fraction = alg.update()

        if episode_count > 0:
            success_rate = success_count / float(episode_count)
        else:
            success_rate = 0.0
        if success_count > 0:
            execution_cost = success_steps_sum / float(success_count)
        else:
            execution_cost = 0.0
        avg_reward = reward_sum / float(horizon)
        avg_goal_dist = goal_dist_sum / float(horizon)
        avg_min_hazard = min_hazard_sum / float(horizon)
        avg_action_sat = action_sat_sum / float(horizon)
        avg_proj = proj_sum / float(horizon)
        avg_angle_error = angle_error_sum / float(horizon)
        avg_obstacle_penalty = obstacle_penalty_sum / float(horizon)
        avg_progress = progress_sum / float(horizon)
        avg_command_speed = command_speed_sum / float(horizon)
        if episode_count > 0:
            reach_rate = reached_count / float(episode_count)
            collision_rate = collision_count / float(episode_count)
            timeout_rate = timeout_count / float(episode_count)
        else:
            reach_rate = 0.0
            collision_rate = 0.0
            timeout_rate = 0.0
        action_std = float(alg.actor_critic.std.mean().item())

        if (iteration + 1) % 1 == 0:
            elapsed = time.time() - interval_start
            log_line = (
                f"iter {iteration + 1:05d} | success {success_rate:.3f} | reach {reach_rate:.3f} | "
                f"collision {collision_rate:.3f} | timeout {timeout_rate:.3f} | cost {execution_cost:.1f} | "
                f"avg_reward {avg_reward:.3f} | proj {avg_proj:.3f} | angle {avg_angle_error:.3f} | "
                f"progress {avg_progress:.3f} | obstacle {avg_obstacle_penalty:.3f} | "
                f"goal_dist {avg_goal_dist:.3f} | min_hazard {avg_min_hazard:.3f} | "
                f"cmd_speed {avg_command_speed:.3f} | action_sat {avg_action_sat:.3f} | "
                f"action_std {action_std:.3f} | policy_loss {policy_loss:.5f} | value_loss {value_loss:.5f} | "
                f"approx_kl {approx_kl:.5f} | clip_frac {clip_fraction:.3f} | "
                f"elapsed {elapsed:.2f}s"
            )
            print(log_line)
            log_fp.write(log_line + "\n")
            log_fp.flush()
            interval_start = time.time()

        if (iteration + 1) % save_interval == 0:
            save_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
            torch.save(
                {
                    "actor_critic": alg.actor_critic.state_dict(),
                    "optimizer": alg.optimizer.state_dict(),
                    "iteration": iteration + 1,
                    "success_rate": success_rate,
                    "execution_cost": execution_cost,
                    "low_level_model_path": train_cfg.runner.low_level_model_path,
                },
                save_path,
            )
            print(f"  saved checkpoint: {save_path}")

        if iteration + 1 < max_iterations:
            obs, g_vals, h_vals = env.reset()
            obs = obs.to(device)
            g_vals = g_vals.to(device)
            h_vals = h_vals.to(device)
            prev_goal_dist = env.env.base_env.reach_metric.clone().to(device)

    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(
        {
            "actor_critic": alg.actor_critic.state_dict(),
            "optimizer": alg.optimizer.state_dict(),
            "iteration": max_iterations,
            "success_rate": success_rate,
            "low_level_model_path": train_cfg.runner.low_level_model_path,
        },
        final_path,
    )
    print(f"training complete. final checkpoint: {final_path}")

    env.close()


if __name__ == "__main__":
    args = get_args()
    args.headless = True
    args.compute_device_id = 6
    args.sim_device_id = 6
    args.rl_device = "cuda:6"
    args.sim_device = "cuda:6"
    train_reward_shaping(args)
