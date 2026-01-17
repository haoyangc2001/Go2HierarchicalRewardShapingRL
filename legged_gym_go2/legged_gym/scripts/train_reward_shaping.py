#!/usr/bin/env python3
import os
import time
import math
from datetime import datetime

import isaacgym
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.utils.hierarchical_env_utils import create_env
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict, update_cfg_from_args
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCritic


def _build_ppo_infos(infos: dict) -> dict:
    ppo_infos = {}
    if isinstance(infos, dict) and "time_outs" in infos:
        ppo_infos["time_outs"] = infos["time_outs"]
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

    algo_cfg = class_to_dict(train_cfg.algorithm)
    ppo_kwargs = {
        key: algo_cfg[key]
        for key in (
            "num_learning_epochs",
            "num_mini_batches",
            "clip_param",
            "gamma",
            "lam",
            "value_loss_coef",
            "entropy_coef",
            "learning_rate",
            "max_grad_norm",
            "use_clipped_value_loss",
            "schedule",
            "desired_kl",
            "value_clip_param",
            "min_lr",
            "max_lr",
        )
        if key in algo_cfg
    }
    alg = PPO(
        actor_critic=actor_critic,
        device=device,
        **ppo_kwargs,
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
        base_log_root = "/home/caohy/repositories/MCRA_RL/logs"
        log_dir = os.path.join(base_log_root, train_cfg.runner.experiment_name, log_timestamp)
        os.makedirs(log_dir, exist_ok=True)
        print(f"created new log directory: {log_dir}")
    else:
        os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "training.log")
    log_fp = open(log_file, "a", encoding="utf-8")
    print(f"training log file: {log_file}")

    reward_cfg = env_cfg.reward_shaping

    print("Reward shaping PPO training")
    print(f"  envs       : {env.num_envs}")
    print(f"  obs dim    : {env.num_obs}")
    print(f"  action dim : {env.num_actions}")
    print(f"  horizon    : {train_cfg.algorithm.num_steps_per_env}")
    print(f"  device     : {device}")
    print(f"  log dir    : {log_dir}")

    obs = env.reset()
    obs = obs.to(device)
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
        boundary_collision_count = 0
        obstacle_collision_count = 0
        timeout_count = 0
        episode_count = 0
        success_steps_sum = 0.0

        reward_sum = 0.0
        goal_dist_sum = 0.0
        min_hazard_sum = 0.0
        hazard_p10_sum = 0.0
        hazard_p50_sum = 0.0
        hazard_p90_sum = 0.0
        action_sat_sum = 0.0
        proj_sum = 0.0
        cmd_align_sum = 0.0
        progress_sum = 0.0
        obstacle_penalty_sum = 0.0
        command_speed_sum = 0.0
        body_speed_sum = 0.0
        speed_ratio_sum = 0.0
        speed_ratio_active_sum = 0.0
        speed_ratio_active_count = 0
        cmd_delta_sum = 0.0
        cmd_delta_count = 0
        cmd_zero_frac_sum = 0.0
        reward_clip_sum = 0.0
        boundary_violation_sum = 0.0
        episode_len_sum = 0.0
        episode_len_sq_sum = 0.0
        episode_len_count = 0.0
        done_frac_sum = 0.0

        if hasattr(env, "env") and hasattr(env.env, "base_env"):
            start_goal_dist = env.env.base_env.reach_metric.mean().item()
        else:
            start_goal_dist = 0.0

        for step in range(horizon):
            actions = alg.act(obs, obs)
            next_obs, rewards, dones, infos = env.step(actions)

            next_obs = next_obs.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            time_outs = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            reached = infos.get("reached", torch.zeros_like(dones, dtype=torch.bool))
            target_distance = infos.get("target_distance", torch.zeros_like(rewards))
            min_hazard_distance = infos.get(
                "min_hazard_distance_true",
                infos.get("min_hazard_distance", torch.zeros_like(rewards)),
            )
            progress = infos.get("progress", torch.zeros_like(rewards))
            alignment = infos.get("alignment", torch.zeros_like(rewards))
            command_alignment = infos.get("command_alignment", torch.zeros_like(rewards))
            obstacle_penalty = infos.get("obstacle_penalty", torch.zeros_like(rewards))
            command_speed = infos.get("command_speed", torch.zeros_like(rewards))
            body_speed = infos.get("body_speed", torch.zeros_like(rewards))
            command_delta = infos.get("command_delta", torch.zeros_like(rewards))
            reward_clip_frac = infos.get("reward_clip_frac", torch.zeros_like(rewards))
            boundary_distance = infos.get("boundary_distance")
            obstacle_surface_distance = infos.get("obstacle_surface_distance")

            hazard_collision = min_hazard_distance < float(reward_cfg.collision_dist)

            done_flags = dones
            episode_steps += 1
            episode_reached |= reached
            episode_collision |= hazard_collision

            if done_flags.any():
                success_mask = done_flags & episode_reached & ~episode_collision
                success_count += success_mask.sum().item()
                reached_count += (done_flags & reached).sum().item()
                collision_count += (done_flags & hazard_collision).sum().item()
                timeout_count += (done_flags & time_outs).sum().item()
                episode_count += done_flags.sum().item()
                ep_lengths = episode_steps[done_flags].float()
                episode_len_sum += ep_lengths.sum().item()
                episode_len_sq_sum += (ep_lengths ** 2).sum().item()
                episode_len_count += done_flags.sum().item()
                if success_mask.any():
                    success_steps_sum += episode_steps[success_mask].float().sum().item()
                episode_collision[done_flags] = False
                episode_reached[done_flags] = False
                episode_steps[done_flags] = 0

            reward_sum += rewards.mean().item()
            goal_dist_sum += target_distance.mean().item()
            min_hazard_sum += min_hazard_distance.mean().item()
            hazard_quantiles = torch.quantile(
                min_hazard_distance,
                torch.tensor([0.1, 0.5, 0.9], device=device),
            )
            hazard_p10_sum += hazard_quantiles[0].item()
            hazard_p50_sum += hazard_quantiles[1].item()
            hazard_p90_sum += hazard_quantiles[2].item()
            action_sat_sum += (actions.abs() > 0.95).float().mean().item()
            proj_sum += alignment.mean().item()
            cmd_align_sum += command_alignment.mean().item()
            progress_sum += progress.mean().item()
            obstacle_penalty_sum += obstacle_penalty.mean().item()
            command_speed_sum += command_speed.mean().item()
            body_speed_sum += body_speed.mean().item()
            speed_ratio = body_speed / command_speed.clamp_min(1e-6)
            speed_ratio_sum += speed_ratio.mean().item()
            active_mask = command_speed > 0.1
            if active_mask.any():
                speed_ratio_active_sum += speed_ratio[active_mask].mean().item()
                speed_ratio_active_count += 1
            cmd_delta_sum += command_delta.mean().item()
            cmd_delta_count += 1
            cmd_zero_frac_sum += (command_speed < 0.1).float().mean().item()
            reward_clip_sum += reward_clip_frac.mean().item()
            done_frac_sum += done_flags.float().mean().item()

            if boundary_distance is not None and obstacle_surface_distance is not None:
                hazard_is_boundary = boundary_distance <= obstacle_surface_distance
                boundary_collision = hazard_collision & hazard_is_boundary
                obstacle_collision = hazard_collision & ~hazard_is_boundary
                boundary_violation = boundary_distance < 0.0
                boundary_violation_sum += boundary_violation.float().mean().item()
                boundary_collision_count += (done_flags & boundary_collision).sum().item()
                obstacle_collision_count += (done_flags & obstacle_collision).sum().item()

            alg.process_env_step(rewards, done_flags, _build_ppo_infos(infos))

            obs = next_obs

        alg.compute_returns(obs)
        values = alg.storage.values
        returns = alg.storage.returns
        advantages_raw = returns - values
        v_mean = values.mean().item()
        v_std = values.std().item()
        r_mean = returns.mean().item()
        r_std = returns.std().item()
        adv_mean = advantages_raw.mean().item()
        adv_std = advantages_raw.std().item()
        value_loss, policy_loss, approx_kl, clip_fraction = alg.update()
        update_stats = getattr(alg, "last_stats", {})

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
        avg_cmd_align = cmd_align_sum / float(horizon)
        avg_progress = progress_sum / float(horizon)
        avg_obstacle_penalty = obstacle_penalty_sum / float(horizon)
        avg_command_speed = command_speed_sum / float(horizon)
        avg_body_speed = body_speed_sum / float(horizon)
        avg_speed_ratio = speed_ratio_sum / float(horizon)
        avg_speed_ratio_active = (
            speed_ratio_active_sum / float(speed_ratio_active_count)
            if speed_ratio_active_count > 0
            else 0.0
        )
        avg_cmd_delta = cmd_delta_sum / float(cmd_delta_count) if cmd_delta_count > 0 else 0.0
        avg_cmd_zero_frac = cmd_zero_frac_sum / float(horizon)
        avg_reward_clip = reward_clip_sum / float(horizon)
        avg_hazard_p10 = hazard_p10_sum / float(horizon)
        avg_hazard_p50 = hazard_p50_sum / float(horizon)
        avg_hazard_p90 = hazard_p90_sum / float(horizon)
        avg_boundary_violation = boundary_violation_sum / float(horizon)
        avg_done_frac = done_frac_sum / float(horizon)
        if episode_len_count > 0:
            avg_episode_len = episode_len_sum / episode_len_count
            var_episode_len = max(episode_len_sq_sum / episode_len_count - avg_episode_len ** 2, 0.0)
            std_episode_len = math.sqrt(var_episode_len)
        else:
            avg_episode_len = 0.0
            std_episode_len = 0.0
        if episode_count > 0:
            reach_rate = reached_count / float(episode_count)
            collision_rate = collision_count / float(episode_count)
            timeout_rate = timeout_count / float(episode_count)
            boundary_collision_rate = boundary_collision_count / float(episode_count)
            obstacle_collision_rate = obstacle_collision_count / float(episode_count)
        else:
            reach_rate = 0.0
            collision_rate = 0.0
            timeout_rate = 0.0
            boundary_collision_rate = 0.0
            obstacle_collision_rate = 0.0
        action_std = float(alg.actor_critic.std.mean().item())

        if (iteration + 1) % 1 == 0:
            elapsed = time.time() - interval_start
            entropy = float(update_stats.get("entropy", 0.0))
            lr = float(update_stats.get("lr", 0.0))
            grad_norm = float(update_stats.get("grad_norm", 0.0))
            value_clip_frac = float(update_stats.get("value_clip_frac", 0.0))
            log_line = (
                f"iter {iteration + 1:05d} | success {success_rate:.3f} | reach {reach_rate:.3f} | "
                f"collision {collision_rate:.3f} | boundary_collision_rate {boundary_collision_rate:.3f} | "
                f"obstacle_collision_rate {obstacle_collision_rate:.3f} | timeout {timeout_rate:.3f} | "
                f"cost {execution_cost:.1f} | "
                f"avg_reward {avg_reward:.3f} | proj {avg_proj:.3f} | cmd_align {avg_cmd_align:.3f} | "
                f"progress {avg_progress:.6f} | obstacle {avg_obstacle_penalty:.3f} | "
                f"goal_dist {avg_goal_dist:.3f} | min_hazard {avg_min_hazard:.3f} | "
                f"cmd_speed {avg_command_speed:.3f} | body_speed {avg_body_speed:.3f} | "
                f"speed_ratio {avg_speed_ratio:.3f} | speed_ratio_active {avg_speed_ratio_active:.3f} | "
                f"cmd_delta {avg_cmd_delta:.3f} | "
                f"cmd_zero {avg_cmd_zero_frac:.3f} | action_sat {avg_action_sat:.3f} | "
                f"action_std {action_std:.3f} | policy_loss {policy_loss:.5f} | value_loss {value_loss:.5f} | "
                f"approx_kl {approx_kl:.5f} | clip_frac {clip_fraction:.3f} | "
                f"entropy {entropy:.5f} | lr {lr:.6f} | grad_norm {grad_norm:.3f} | "
                f"value_clip_frac {value_clip_frac:.3f} | "
                f"Vmean {v_mean:.3f} | Vstd {v_std:.3f} | Rmean {r_mean:.3f} | Rstd {r_std:.3f} | "
                f"adv_mean {adv_mean:.3f} | adv_std {adv_std:.3f} | reward_clip {avg_reward_clip:.3f} | "
                f"hazard_p10 {avg_hazard_p10:.3f} | hazard_p50 {avg_hazard_p50:.3f} | hazard_p90 {avg_hazard_p90:.3f} | "
                f"boundary_violation {avg_boundary_violation:.3f} | done_frac {avg_done_frac:.3f} | "
                f"ep_len_mean {avg_episode_len:.1f} | "
                f"ep_len_std {std_episode_len:.1f} | init_goal_dist {start_goal_dist:.3f} | "
                f"elapsed {elapsed:.2f}s | "
                f"\n"
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
            obs = env.reset()
            obs = obs.to(device)

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
    args.compute_device_id = 3
    args.sim_device_id = 3
    args.rl_device = "cuda:3"
    args.sim_device = "cuda:3"
    train_reward_shaping(args)
