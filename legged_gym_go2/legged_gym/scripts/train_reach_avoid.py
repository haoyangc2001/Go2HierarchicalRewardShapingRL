#!/usr/bin/env python3
import os
import time
from datetime import datetime
from typing import Tuple
import json
import csv

import isaacgym
import torch

from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args

from rsl_rl.algorithms.reach_avoid_ppo import ReachAvoidPPO
from rsl_rl.modules import ActorCritic


class HierarchicalVecEnv:
    def __init__(self, env: HierarchicalGO2Env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.device = env.device

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, g_vals, h_vals = self.env.reset()
        return obs, g_vals, h_vals

    def step(self, actions: torch.Tensor):
        obs, g_vals, h_vals, dones, infos = self.env.step(actions)
        return obs, g_vals, h_vals, dones, infos

    def close(self) -> None:
        self.env.close()


def create_env(env_cfg, train_cfg, args, device) -> HierarchicalVecEnv:
    base_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )
    return HierarchicalVecEnv(base_env)


def compute_reach_avoid_success_rate(g_sequence: torch.Tensor, h_sequence: torch.Tensor):
    """
    计算Reach-Avoid任务的成功率和执行成本
    
    Reach-Avoid任务要求智能体：
    1. 最终到达目标区域（reach条件）
    2. 在到达目标之前不违反安全约束（avoid条件）
    
    参数:
        g_sequence: 目标函数序列，形状为 [time_steps, num_envs]
                    g < 0 表示到达目标区域
        h_sequence: 安全约束函数序列，形状为 [time_steps, num_envs]
                    h >= 0 表示违反安全约束
    
    返回:
        success_rate: 成功率，满足reach和avoid条件的环境比例
        execution_cost: 执行成本，所有成功环境到达目标所需的平均时间步数
    """
    with torch.no_grad():
        time_steps, num_envs = g_sequence.shape
        
        # 标记目标函数小于0的时间步（即到达目标的时间步）
        g_negative = g_sequence < 0
        
        # 判断每个环境是否至少有一次到达目标
        has_success = g_negative.any(dim=0)
        
        # 找到每个环境第一次到达目标的时间步索引
        first_success = torch.argmax(g_negative.long(), dim=0)
        
        # 对于从未到达目标的环境，将其首次成功时间设为time_steps（超出序列范围）
        first_indices = torch.where(
            has_success,
            first_success,
            torch.full((num_envs,), time_steps, device=g_sequence.device, dtype=torch.long),
        )

        # 创建时间索引 [0, 1, 2, ..., time_steps-1]，形状为 [time_steps, 1]
        time_index = torch.arange(time_steps, device=g_sequence.device).unsqueeze(1)
        
        # 标记在首次到达目标之前的所有时间步
        # before_success形状为 [time_steps, num_envs]
        before_success = time_index < first_indices.unsqueeze(0)

        # 检查在到达目标之前是否违反了安全约束
        # h >= 0 表示违反约束，且必须在到达目标之前
        h_violation = (h_sequence >= 0) & before_success
        
        # 判断每个环境在到达目标之前是否始终安全（没有违反约束）
        safe_before = ~h_violation.any(dim=0)

        # 最终成功条件：既要到达目标，又要在此之前保持安全
        success = has_success & safe_before
        
        # 计算成功率：成功环境的数量占总环境数量的比例
        success_rate = success.float().mean().item()
        
        # 计算执行成本：所有成功环境到达目标所需的平均时间步数
        # 首先筛选出成功环境的首次到达时间索引
        success_first_indices = first_indices[success]
        
        # 计算成功环境的平均时间步数
        # 如果没有成功环境，则执行成本为0
        if success.sum().item() > 0:
            execution_cost = success_first_indices.float().mean().item()
        else:
            execution_cost = 0.0
        
        return success_rate, execution_cost


def train_reach_avoid(args) -> None:
    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()

    # 配置网络大小
    train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]
    train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]

    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

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
    ).to(device)

    alg = ReachAvoidPPO(
        actor_critic=actor_critic,
        device=device,
        **train_cfg.algorithm.__dict__,
    )
    alg.init_storage(
        num_envs=env.num_envs,
        horizon=train_cfg.algorithm.num_steps_per_env,
        obs_shape=(env.num_obs,),
        action_shape=(env.num_actions,),
    )

    start_iteration = 0
    log_dir = None

    if getattr(train_cfg.runner, "resume", False):
        resume_path = getattr(train_cfg.runner, "resume_path", "")
        if resume_path and os.path.isfile(resume_path):
            # Resume时使用检查点所在的目录作为日志目录
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
            raise FileNotFoundError(f"Resume enabled but checkpoint not found: {resume_path}. Please provide a valid checkpoint path.")

    # 如果没有log_dir（非Resume），创建新目录
    if log_dir is None:
        log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", train_cfg.runner.experiment_name, log_timestamp)
        os.makedirs(log_dir, exist_ok=True)
        print(f"created new log directory: {log_dir}")
    else:
        # Resume时确保目录存在
        os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(log_dir, "training.log")
    log_fp = open(log_file, 'a', encoding='utf-8')
    print(f"training log file: {log_file}")

    gh_dump_interval = getattr(train_cfg.runner, "gh_dump_interval", 0)
    gh_dump_dir = None
    if gh_dump_interval and gh_dump_interval > 0:
        gh_dump_dir = os.path.join(log_dir, "gh_snapshots")
        os.makedirs(gh_dump_dir, exist_ok=True)

    print("Reach-Avoid training")
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
    horizon = train_cfg.algorithm.num_steps_per_env

    max_iterations = train_cfg.runner.max_iterations
    save_interval = train_cfg.runner.save_interval
    success_rate = 0.0
    interval_start = time.time()


    for iteration in range(start_iteration, max_iterations):
        rollout_obs = obs.new_empty(horizon + 1, env.num_envs, env.num_obs)
        rollout_actions = obs.new_empty(horizon, env.num_envs, env.num_actions)
        rollout_log_probs = obs.new_empty(horizon, env.num_envs)
        rollout_values = obs.new_empty(horizon, env.num_envs)
        rollout_g = g_vals.new_empty(horizon + 1, env.num_envs)
        rollout_h = h_vals.new_empty(horizon + 1, env.num_envs)
        rollout_dones = torch.empty(horizon, env.num_envs, device=device, dtype=torch.bool)

        rollout_obs[0].copy_(obs)
        rollout_g[0].copy_(g_vals)
        rollout_h[0].copy_(h_vals)

        for step in range(horizon):
            # 在当前时间步用策略选择动作并返回相关信息：
            # actions: 形状 [num_envs, num_actions]，策略输出的动作（通常是连续动作向量）
            # log_probs: 形状 [num_envs]，所选动作的对数概率（用于 PPO 的目标函数与重要性采样）
            # values: 形状 [num_envs] 或 [num_envs, 1]，状态价值估计（由 critic 产生，用于计算优势和返回）
            actions, log_probs, values = alg.act(rollout_obs[step])

            # 将动作应用到环境，env.step 返回下一步信息：
            # next_obs: 下一个观测，形状 [num_envs, obs_dim]
            # next_g: 目标函数 g 的值序列（reach 指标），形状 [num_envs] 或 [num_envs, ...]
            # next_h: 安全函数 h 的值序列（avoid 指标），形状 [num_envs] 或 [num_envs, ...]
            # dones: 布尔张量，指示每个环境是否已结束（episode 终止或被截断）
            # _: 额外信息（infos），此处不使用
            next_obs, next_g, next_h, dones, _ = env.step(actions)

            # 将从环境返回的张量移动到训练设备（例如 GPU）上，便于后续计算
            next_obs = next_obs.to(device)
            next_g = next_g.to(device)
            next_h = next_h.to(device)
            dones = dones.to(device)

            # 将当前时间步的数据保存到 rollout 缓冲区：
            # rollout_actions: 存储每个时间步每个环境采取的动作
            # rollout_log_probs: 存储每个动作的对数概率（用于策略更新）
            # rollout_values: 存储 critic 在该状态下的价值估计（用于计算优势）
            # rollout_dones: 存储每个环境在该时间步是否结束（布尔值）
            rollout_actions[step].copy_(actions)
            rollout_log_probs[step].copy_(log_probs)
            rollout_values[step].copy_(values)
            rollout_dones[step].copy_(dones.bool())

            # 将下一步的观测与 g/h 值保存为下一时间步的输入（rollout_obs[0] 在循环外初始化）
            # rollout_obs[t+1] 对应时间步 t+1 的观测（供策略在下一步作为输入）
            # rollout_g / rollout_h 保存与每个时间步对应的 reach/avoid 度量，用于后续成功率评估和训练信号
            rollout_obs[step + 1].copy_(next_obs)
            rollout_g[step + 1].copy_(next_g)
            rollout_h[step + 1].copy_(next_h)

            # 更新当前观测与指标以便下个循环使用，也用于在循环外计算最后时刻的价值估计
            obs = next_obs
            g_vals = next_g
            h_vals = next_h

        # force horizon truncation to behave like episode termination
        # TODO : 这里为什么要强制将最后一步的 dones 设为 True？ 这里的 Dones 是指在每个时间步环境是否结束的标志，作用是什么？
        rollout_dones[-1].fill_(True)

        alg.buffer.store_rollout(
            observations=rollout_obs,
            actions=rollout_actions,
            log_probs=rollout_log_probs,
            values=rollout_values,
            g_values=rollout_g,
            h_values=rollout_h,
            dones=rollout_dones,
        )

        with torch.no_grad():
            last_values = alg.actor_critic.evaluate(obs).squeeze(-1)

        alg.buffer.compute_advantages(last_values, alg.gamma, alg.lam)
        success_rate, execution_cost = compute_reach_avoid_success_rate(
            alg.buffer.g_values[1:], alg.buffer.h_values[1:]
        )
        policy_loss, value_loss = alg.update()
        value_stats = getattr(alg, "last_value_stats", {})
        v_mean = value_stats.get("value_mean", float("nan"))
        r_mean = value_stats.get("return_mean", float("nan"))
        v_rmse = value_stats.get("value_rmse", float("nan"))
        v_expvar = value_stats.get("explained_variance", float("nan"))
        adv_std = value_stats.get("adv_std", float("nan"))

        if (iteration + 1) % 1 == 0:
            elapsed = time.time() - interval_start
            
            # 迭代轮数 | 成功率 | 执行成本 | 策略损失 | 价值损失 | 价值均值 | 回报均值 | 价值RMSE | 解释方差(值函数预测质量) | 优势标准差(优势函数稳定性) | 迭代用时
            log_line = f"iter {iteration + 1:05d} | success {success_rate:.3f} | cost {execution_cost:.1f} | policy_loss {policy_loss:.5f} | value_loss {value_loss:.5f} | Vmean {v_mean:.3f} | Rmean {r_mean:.3f} | Vrmse {v_rmse:.3f} | VexpVar {v_expvar:.3f} | adv_std {adv_std:.3f} | elapsed {elapsed:.2f}s"
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


        # start the next rollout from a freshly reset environment
        if iteration + 1 < max_iterations:
            obs, g_vals, h_vals = env.reset()
            obs = obs.to(device)
            g_vals = g_vals.to(device)
            h_vals = h_vals.to(device)

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
    train_reach_avoid(args)



