#!/usr/bin/env python3
import os
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
        base_log_root = "/home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping"
        log_dir = os.path.join(base_log_root, log_timestamp)
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
    for iteration in range(start_iteration, max_iterations):
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
        progress_sum = 0.0
        progress_sq_sum = 0.0
        safety_penalty_sum = 0.0
        safety_penalty_sq_sum = 0.0
        smooth_penalty_sum = 0.0
        smooth_penalty_sq_sum = 0.0
        target_speed_reward_sum = 0.0
        target_speed_reward_sq_sum = 0.0
        reward_clip_sum = 0.0
        episode_len_sum = 0.0
        episode_len_count = 0.0
        cmd_corr_count = 0
        cmd_corr_sum_x = 0.0
        cmd_corr_sum_y = 0.0
        cmd_corr_sum_x2 = 0.0
        cmd_corr_sum_y2 = 0.0
        cmd_corr_sum_xy = 0.0

        for step in range(horizon):
            actions = alg.act(obs, obs)
            step_out = env.step(actions)
            if len(step_out) == 4:
                next_obs, rewards, dones, infos = step_out
            elif len(step_out) == 5:
                next_obs, _, rewards, dones, infos = step_out
            else:
                raise ValueError(f"Unexpected env.step return size: {len(step_out)}")

            next_obs = next_obs.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            time_outs = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            reached = infos.get("reached", torch.zeros_like(dones, dtype=torch.bool))
            success = infos.get("success", torch.zeros_like(dones, dtype=torch.bool))
            collision = infos.get("collision")
            target_distance = infos.get("target_distance", torch.zeros_like(rewards))
            min_hazard_distance = infos.get(
                "min_hazard_distance_true",
                infos.get("min_hazard_distance", torch.zeros_like(rewards)),
            )
            progress = infos.get("progress", torch.zeros_like(rewards))
            safety_penalty = infos.get("safety_penalty", torch.zeros_like(rewards))
            smooth_penalty = infos.get("smooth_penalty", torch.zeros_like(rewards))
            target_speed_reward = infos.get("target_speed_reward", torch.zeros_like(rewards))
            reward_clip_frac = infos.get("reward_clip_frac", torch.zeros_like(rewards))
            command_speed = infos.get("command_speed", torch.zeros_like(rewards))
            boundary_distance = infos.get("boundary_distance")
            obstacle_surface_distance = infos.get("obstacle_surface_distance")

            hazard_collision = min_hazard_distance < float(reward_cfg.collision_dist)
            if collision is None:
                collision = hazard_collision

            done_flags = dones
            episode_steps += 1

            if done_flags.any():
                success_mask = done_flags & success
                success_count += success_mask.sum().item()
                reached_count += (done_flags & reached).sum().item()
                collision_count += (done_flags & collision).sum().item()
                timeout_count += (done_flags & time_outs).sum().item()
                episode_count += done_flags.sum().item()
                ep_lengths = episode_steps[done_flags].float()
                episode_len_sum += ep_lengths.sum().item()
                episode_len_count += done_flags.sum().item()
                if success_mask.any():
                    success_steps_sum += episode_steps[success_mask].float().sum().item()
                episode_steps[done_flags] = 0

            reward_sum += rewards.mean().item()
            goal_dist_sum += target_distance.mean().item()
            min_hazard_sum += min_hazard_distance.mean().item()
            progress_sum += progress.mean().item()
            progress_sq_sum += (progress ** 2).mean().item()
            safety_penalty_sum += safety_penalty.mean().item()
            safety_penalty_sq_sum += (safety_penalty ** 2).mean().item()
            smooth_penalty_sum += smooth_penalty.mean().item()
            smooth_penalty_sq_sum += (smooth_penalty ** 2).mean().item()
            target_speed_reward_sum += target_speed_reward.mean().item()
            target_speed_reward_sq_sum += (target_speed_reward ** 2).mean().item()
            reward_clip_sum += reward_clip_frac.mean().item()
            cmd_corr_count += command_speed.numel()
            cmd_corr_sum_x += command_speed.sum().item()
            cmd_corr_sum_y += progress.sum().item()
            cmd_corr_sum_x2 += (command_speed ** 2).sum().item()
            cmd_corr_sum_y2 += (progress ** 2).sum().item()
            cmd_corr_sum_xy += (command_speed * progress).sum().item()

            if boundary_distance is not None and obstacle_surface_distance is not None:
                hazard_is_boundary = boundary_distance <= obstacle_surface_distance
                boundary_collision = hazard_collision & hazard_is_boundary
                obstacle_collision = hazard_collision & ~hazard_is_boundary
                boundary_collision_count += (done_flags & boundary_collision).sum().item()
                obstacle_collision_count += (done_flags & obstacle_collision).sum().item()

            alg.process_env_step(rewards, done_flags, _build_ppo_infos(infos))

            obs = next_obs

        alg.compute_returns(obs)
        update_out = alg.update()
        if isinstance(update_out, dict):
            value_loss = float(update_out.get("value", 0.0))
            policy_loss = float(update_out.get("surrogate", 0.0))
            approx_kl = float(update_out.get("approx_kl", float("nan")))
            clip_fraction = float(update_out.get("clip_frac", float("nan")))
        elif isinstance(update_out, (list, tuple)):
            if len(update_out) == 2:
                value_loss, policy_loss = update_out
                approx_kl = float(getattr(alg, "last_approx_kl", float("nan")))
                clip_fraction = float(getattr(alg, "last_clip_frac", float("nan")))
            elif len(update_out) >= 4:
                value_loss, policy_loss, approx_kl, clip_fraction = update_out[:4]
            else:
                raise ValueError(f"Unexpected PPO update output size: {len(update_out)}")
        else:
            raise ValueError(f"Unexpected PPO update output type: {type(update_out)}")
        lr = float(getattr(alg, "learning_rate", 0.0))
        num_minibatches = int(getattr(alg, "last_num_minibatches", 0))
        num_updates = int(getattr(alg, "last_num_updates", 0))
        num_skipped = int(getattr(alg, "last_num_skipped", 0))
        num_skipped_kl = int(getattr(alg, "last_num_skipped_kl", 0))
        num_skipped_nonfinite = int(getattr(alg, "last_num_skipped_nonfinite", 0))
        adv_mean = float(getattr(alg, "last_adv_mean", float("nan")))
        adv_std = float(getattr(alg, "last_adv_std", float("nan")))
        ratio_mean = float(getattr(alg, "last_ratio_mean", float("nan")))
        ratio_std = float(getattr(alg, "last_ratio_std", float("nan")))
        ratio_abs_mean = float(getattr(alg, "last_ratio_abs_mean", float("nan")))
        ratio_min = float(getattr(alg, "last_ratio_min", float("nan")))
        ratio_max = float(getattr(alg, "last_ratio_max", float("nan")))
        if num_minibatches > 0:
            skip_frac = num_skipped / float(num_minibatches)
        else:
            skip_frac = 0.0

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
        avg_progress = progress_sum / float(horizon)
        avg_progress_sq = progress_sq_sum / float(horizon)
        progress_var = max(avg_progress_sq - avg_progress ** 2, 0.0)
        avg_safety_penalty = safety_penalty_sum / float(horizon)
        avg_safety_penalty_sq = safety_penalty_sq_sum / float(horizon)
        safety_var = max(avg_safety_penalty_sq - avg_safety_penalty ** 2, 0.0)
        avg_smooth_penalty = smooth_penalty_sum / float(horizon)
        avg_smooth_penalty_sq = smooth_penalty_sq_sum / float(horizon)
        smooth_var = max(avg_smooth_penalty_sq - avg_smooth_penalty ** 2, 0.0)
        avg_target_speed_reward = target_speed_reward_sum / float(horizon)
        avg_target_speed_reward_sq = target_speed_reward_sq_sum / float(horizon)
        target_speed_var = max(avg_target_speed_reward_sq - avg_target_speed_reward ** 2, 0.0)
        avg_reward_clip = reward_clip_sum / float(horizon)
        if cmd_corr_count > 1:
            mean_x = cmd_corr_sum_x / cmd_corr_count
            mean_y = cmd_corr_sum_y / cmd_corr_count
            cov_xy = (cmd_corr_sum_xy / cmd_corr_count) - mean_x * mean_y
            var_x = (cmd_corr_sum_x2 / cmd_corr_count) - mean_x ** 2
            var_y = (cmd_corr_sum_y2 / cmd_corr_count) - mean_y ** 2
            if var_x > 0.0 and var_y > 0.0:
                cmd_goal_corr = cov_xy / ((var_x * var_y) ** 0.5)
            else:
                cmd_goal_corr = 0.0
        else:
            cmd_goal_corr = 0.0
        if episode_len_count > 0:
            avg_episode_len = episode_len_sum / episode_len_count
        else:
            avg_episode_len = 0.0
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
            log_line = (
                f"iter {iteration + 1:05d} | success {success_rate:.3f} | reach {reach_rate:.3f} | "
                f"collision {collision_rate:.3f} | boundary_collision_rate {boundary_collision_rate:.3f} | "
                f"obstacle_collision_rate {obstacle_collision_rate:.3f} | timeout {timeout_rate:.3f} | "
                f"cost {execution_cost:.1f} | "
                f"avg_reward {avg_reward:.3f} | "
                f"progress {avg_progress:.6f} | safety {avg_safety_penalty:.3f} | smooth {avg_smooth_penalty:.3f} | "
                f"target_speed {avg_target_speed_reward:.3f} | "
                f"progress_var {progress_var:.6f} | safety_var {safety_var:.6f} | "
                f"smooth_var {smooth_var:.6f} | target_speed_var {target_speed_var:.6f} | "
                f"goal_dist {avg_goal_dist:.3f} | min_hazard {avg_min_hazard:.3f} | "
                f"cmd_goal_corr {cmd_goal_corr:.4f} | "
                f"reward_clip {avg_reward_clip:.3f} | action_std {action_std:.3f} | "
                f"policy_loss {policy_loss:.5f} | value_loss {value_loss:.5f} | "
                f"approx_kl {approx_kl:.5f} | clip_frac {clip_fraction:.3f} | "
                f"adv_mean {adv_mean:.6f} | adv_std {adv_std:.6f} | "
                f"ratio_mean {ratio_mean:.6f} | ratio_std {ratio_std:.6f} | "
                f"ratio_abs_mean {ratio_abs_mean:.6f} | ratio_min {ratio_min:.6f} | ratio_max {ratio_max:.6f} | "
                f"lr {lr:.6f} | "
                f"ppo_updates {num_updates:d} | ppo_skipped {num_skipped:d} | "
                f"ppo_skip_frac {skip_frac:.3f} | ppo_skip_kl {num_skipped_kl:d} | "
                f"ppo_skip_nonfinite {num_skipped_nonfinite:d} | "
                f"ep_len_mean {avg_episode_len:.1f} | "
                f"\n"
            )
            print(log_line)
            log_fp.write(log_line + "\n")
            log_fp.flush()

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
    args.compute_device_id = 1
    args.sim_device_id = 1
    args.rl_device = "cuda:1"
    args.sim_device = "cuda:1"
    train_reward_shaping(args)
