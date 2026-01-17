# Repository Guidelines

## Project Overview
MCRA_RL is a hierarchical reinforcement learning system for Unitree Go2 navigation. The low-level locomotion policy is fixed and pre-trained, while the high-level policy is trained with reward-shaping PPO to reach a target and avoid obstacles/boundaries.

## Key Paths
- Environments: `legged_gym_go2/legged_gym/envs/go2/`
- Training scripts: `legged_gym_go2/legged_gym/scripts/`
- RL algorithms: `rsl_rl/rsl_rl/algorithms/`
- Deployment: `legged_gym_go2/deploy/`
- Logs/checkpoints: `/home/caohy/repositories/MCRA_RL/logs/`

## Hierarchical RL Structure
- Low-level (locomotion): `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- High-level (navigation wrapper): `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- Hierarchical wrapper: `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- High-level actions are repeated at low level via `GO2HighLevelCfg.env.high_level_action_repeat`.

## Environment Overview
### Low-Level Environment (Locomotion)
- Implements `GO2Robot` by extending `LeggedRobot`, with safety metrics injected into `step()` outputs.
- `reset()` calls `reset_idx` then performs a zero-action `step` to prime observations.
- `_compute_safety_metrics()` computes:
  - `avoid_metric`: positive inside unsafe regions.
  - `reach_metric`: XY distance to target center.
  - `min_hazard_distance`: nearest hazard surface distance (min of obstacle surface distance and boundary distance).
- `check_termination()` augments termination with reach/avoid checks when `terminate_on_reach_avoid` is enabled; collision prefers `min_hazard_distance < collision_dist`.

### High-Level Navigation Wrapper
- Builds high-level observations from low-level state:
  - Base 8 dims: `cos(heading)`, `sin(heading)`, `body_vx`, `body_vy`, `yaw_rate`, normalized `dist_to_target`, `target_dir_body_x`, `target_dir_body_y`.
  - Optional target lidar bins (smooth angular binning with distance decay).
  - Optional obstacle/boundary lidar bins (max intensity per bin, boundary handled via ray intersections).
- Maps high-level actions to low-level velocity commands in `update_velocity_commands` with fixed scaling and ranges.

### Hierarchical Wrapper
- Loads a fixed low-level policy via `OnPolicyRunner` and exposes a high-level interface.
- Each high-level action is repeated for `high_level_action_repeat` low-level steps; dones are aggregated.
- High-level observations and g/h values are computed after the low-level rollouts.

### Vectorized Adapter
- `HierarchicalVecEnv` provides a vectorized API for PPO training while delegating to the hierarchical environment.

### Environment Configuration
- Low-level base config: `GO2RoughCfg` (terrain, domain randomization, rewards, obstacle/target layout).
- High-level config: `GO2HighLevelCfg` (lidar, action repeat, observation dimension).
- Observation dimension is computed as `8 + target_lidar_num_bins + lidar_num_bins`.

### High-Level Action Mapping (Do Not Change)
In `update_velocity_commands`:
- Clip high-level actions to `[-1, 1]`.
- Multiply by `HighLevelNavigationConfig.action_scale`.
- Map to base commands:
  - `vx = action[0] * 0.6`
  - `vy = action[1] * 0.2`
  - `vyaw = action[2] * 0.8`
With default `action_scale = [1, 1, 1]`, the effective command ranges are:
`vx ∈ [-0.6, 0.6]`, `vy ∈ [-0.2, 0.2]`, `vyaw ∈ [-0.8, 0.8]`.

## High-Level Observations
- Base features (8):
  1) `cos(heading)`
  2) `sin(heading)`
  3) `body_vx` (scaled, clipped)
  4) `body_vy` (scaled, clipped)
  5) `yaw_rate` (scaled, clipped)
  6) `dist_to_target` (normalized)
  7) `target_dir_body_x`
  8) `target_dir_body_y`
- Optional target lidar bins: `target_lidar_num_bins`
- Optional obstacle/boundary lidar bins: `lidar_num_bins`
- Total dim: `8 + target_lidar_num_bins + lidar_num_bins` when manual lidar is enabled.

## Reward Shaping PPO (High Level)
- Training script: `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
- PPO implementation: `rsl_rl/rsl_rl/algorithms/ppo.py`
- Reward design:
  - Success: `+success_reward` when `reach_metric <= goal_reached_dist`
  - Collision: `-collision_penalty` when `min_hazard_distance < collision_dist`
  - Dense step reward:
    - `forward_reward_scale * dot(v_cmd_xy, target_dir_body)`
    - `- yaw_penalty_scale * angle_error` (masked when `|v_cmd_xy|` is near zero)
    - `- obstacle_penalty_scale * r3(min_hazard_distance)`
  - Goal progress: `goal_progress_scale * (prev_goal_dist - reach_metric)` (not applied on terminated/truncated steps)
  - Timeout: optional `timeout_penalty`
  - `r3(d) = max(1 - d / obstacle_avoid_dist, 0)`
- Termination:
  - `collision = min_hazard_distance < collision_dist`
  - `success = reach_metric <= goal_reached_dist`
  - `terminated = collision OR success`
  - `truncated = time_out AND NOT terminated`
  - PPO bootstraps only on `truncated` via `time_outs`.

## Safety Metrics
- `min_hazard_distance` is computed in `legged_gym_go2/legged_gym/envs/go2/go2_env.py` as the nearest hazard surface distance (min of obstacle surface distance and boundary distance).
- `avoid_metric` is positive inside unsafe regions; `reach_metric` is XY distance to the target.

## Logging and Outputs
- Training logs/checkpoints are saved to:
  `/home/caohy/repositories/MCRA_RL/logs/<experiment_name>/<timestamp>/`
- The training log file is `training.log`.
- Logged metrics include: `success`, `reach`, `collision`, `timeout`, `cost`, `avg_reward`, `proj`,
  `angle`, `progress`, `obstacle`, `goal_dist`, `min_hazard`, `cmd_speed`, `body_speed`,
  `speed_ratio`, `speed_ratio_active`, `cmd_delta`, `cmd_zero`, `cmd_std_vx`, `cmd_std_vy`, `cmd_std_vyaw`,
  `cmd_speed_std`, `action_sat`, `done_frac`,
  `action_std`, `policy_loss`, `value_loss`, `approx_kl`, `clip_frac`, `elapsed`,
  plus PPO/diagnostic metrics such as `entropy`, `lr`, `ratio_mean`, `ratio_max`,
  `logp_diff_max`, `approx_kl_raw`, `grad_norm`, `value_clip_frac`, `Vmean`, `Vstd`,
  `Rmean`, `Rstd`, `adv_mean`, `adv_std`, `reward_clip`, `hazard_p10`, `hazard_p50`,
  `hazard_p90`, `boundary_violation`, `boundary_collision`, `obstacle_collision`,
  `boundary_collision_rate`, `obstacle_collision_rate`,
  `ep_len_mean`, `ep_len_std`, `init_goal_dist`.

### Training Log Field Meanings
- `iter`: 迭代编号（一次 rollout + 一次 PPO 更新为一轮），从 1 开始计数。
- `success`: 成功率，按本迭代内结束的 episode 统计；成功定义为到达目标且该 episode 内未发生碰撞。
- `reach`: 到达率，按结束的 episode 统计，满足 `reach_metric <= goal_reached_dist`。
- `collision`: 碰撞率，按结束的 episode 统计，满足 `min_hazard_distance < collision_dist`。
- `timeout`: 超时率，按结束的 episode 统计，仅 `time_outs` 且非到达/碰撞。
- `cost`: 成功 episode 的平均高层步数成本（不是秒），越低代表越快到达。
- `avg_reward`: 该迭代每步平均 shaped reward（含终止奖励与缩放/裁剪）。
- `proj`: 目标方向投影均值 `dot(v_cmd_xy, target_dir_body)`，越大代表朝目标方向运动越强。
- `angle`: 命令方向与目标方向夹角均值（弧度），速度接近 0 时会被屏蔽。
- `progress`: 平均目标距离进度 `prev_goal_dist - reach_metric`，终止/截断步置 0。
- `obstacle`: 避障惩罚项 `r3(min_hazard_distance)` 均值（0~1，越大越接近障碍）。
- `goal_dist`: 平均目标距离 `reach_metric`（XY 平面距离，单位米）。
- `min_hazard`: 平均最近危险距离（障碍表面距离与边界距离的最小值，可能为负表示进入危险/越界）。
- `cmd_speed`: 平均平面速度命令 `||v_cmd_xy||`（单位同底层命令，通常 m/s）。
- `body_speed`: 平均机体平面速度 `||v_body_xy||`（单位 m/s）。
- `speed_ratio`: `body_speed / cmd_speed` 的均值（命令能否转化为实际速度的粗略指标）。
- `speed_ratio_active`: 仅在 `cmd_speed > 0.1` 时统计的 `speed_ratio` 均值（过滤低指令/停滞步）。
- `cmd_delta`: 高层速度命令相邻步差值的均值（`||cmd_t - cmd_{t-1}||`）。
- `cmd_zero`: `cmd_speed < 0.1` 的比例（高层几乎不下发平面速度的步比例）。
- `cmd_std_vx`: 高层命令 `vx` 的均值标准差（跨 env 统计，按步平均）。
- `cmd_std_vy`: 高层命令 `vy` 的均值标准差（跨 env 统计，按步平均）。
- `cmd_std_vyaw`: 高层命令 `vyaw` 的均值标准差（跨 env 统计，按步平均）。
- `cmd_speed_std`: 高层命令平面速度模长 `||v_cmd_xy||` 的标准差（跨 env 统计，按步平均）。
- `action_sat`: 动作饱和比例，统计 `|a| > 0.95` 的比例（tanh 输出接近边界）。
- `action_std`: 策略分布 std 的均值（探索强度指标）。
- `policy_loss`: PPO 策略损失的迭代均值。
- `value_loss`: PPO 价值损失的迭代均值。
- `approx_kl`: 近似 KL（由 log-prob 差分计算的均值）。
- `clip_frac`: PPO ratio 被裁剪的样本占比。
- `entropy`: 策略分布熵均值（探索强度的直接指标）。
- `lr`: 当前 PPO 学习率（自适应调度时会动态变化）。
- `ratio_mean`: `ratio=exp(logp_diff)` 的均值（更新强度指示）。
- `ratio_max`: `ratio` 的最大值（是否存在极端更新）。
- `logp_diff_max`: 未裁剪 `logp_diff` 的绝对值最大值（爆炸预警）。
- `approx_kl_raw`: 基于未裁剪 `logp_diff` 的 KL 估计。
- `grad_norm`: 梯度范数均值（爆炸或过小都可见）。
- `value_clip_frac`: 价值函数更新中被 clip 的样本占比。
- `Vmean`: 价值函数输出均值。
- `Vstd`: 价值函数输出标准差。
- `Rmean`: 回报（return）均值。
- `Rstd`: 回报（return）标准差。
- `adv_mean`: 未归一化优势均值（return - value）。
- `adv_std`: 未归一化优势标准差。
- `reward_clip`: reward 被裁剪的平均比例。
- `hazard_p10`: `min_hazard_distance` 的 10% 分位数。
- `hazard_p50`: `min_hazard_distance` 的 50% 分位数（中位数）。
- `hazard_p90`: `min_hazard_distance` 的 90% 分位数。
- `boundary_violation`: 越界比例（boundary_distance < 0）。
- `boundary_collision`: 每步发生边界碰撞的平均比例（基于最近危险源判定）。
- `obstacle_collision`: 每步发生障碍物碰撞的平均比例（基于最近危险源判定）。
- `boundary_collision_rate`: 按 episode 结束统计的边界碰撞率（与 `collision` 同口径）。
- `obstacle_collision_rate`: 按 episode 结束统计的障碍碰撞率（与 `collision` 同口径）。
- `done_frac`: 每步 done 标记的平均比例（高层终止频率）。
- `ep_len_mean`: 本迭代内完成的 episode 平均步数。
- `ep_len_std`: episode 步数标准差。
- `init_goal_dist`: 每次迭代开始时的目标初始距离均值。
- `elapsed`: 本次日志与上一次日志的间隔时间（秒）。

## Configuration Entry Points
- Reward shaping parameters: `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2HighLevelCfg.reward_shaping`)
- Termination distances and hazards: `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2RoughCfg.rewards_ext`)
- PPO hyperparameters: `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2HighLevelCfgPPO`)
- Observation dimension: computed at end of `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
- Low-level checkpoint path: `GO2HighLevelCfgPPO.runner.low_level_model_path`

## Common Commands
Before running any script:
```bash
conda activate unitree-rl
```

Train reward shaping:
```bash
python legged_gym_go2/legged_gym/scripts/train_reward_shaping.py --headless=true --num_envs=32
```

Plot arena layout:
```bash
python legged_gym_go2/legged_gym/scripts/plot_env_layout.py
```

Plot training logs:
```bash
python legged_gym_go2/legged_gym/scripts/plot_training_results.py /home/caohy/repositories/MCRA_RL/logs/<experiment>/<timestamp>/training.log
```

Deploy in Mujoco (example):
```bash
python legged_gym_go2/deploy/deploy_mujoco/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml
```

## Development Notes
- `train_reward_shaping.py` overrides some CLI args in `__main__` (headless + device IDs). Edit there if you need different devices.
- `HierarchicalGO2Env` sets `terminate_on_reach_avoid` based on reward shaping flags.
- The low-level policy is fixed; high-level training should not modify it.

## Non-Negotiable Constraint
- Do not change the high-level speed limit mapping in `update_velocity_commands` or its effective ranges.
