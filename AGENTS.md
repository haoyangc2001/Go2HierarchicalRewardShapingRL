# Repository Guidelines

## Project Overview

MCRA_RL is a hierarchical reinforcement learning system for Unitree Go2 quadruped navigation. The current training focus is **reward shaping PPO** for high-level obstacle-avoidance navigation, with a fixed, pre-trained low-level locomotion controller.

## Architecture

### Hierarchical RL Structure
- **Low-level**: Pre-trained locomotion policy (velocity → joint actions) in `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- **High-level**: Trainable navigation policy (observations → velocity commands) in `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- **Hierarchical wrapper**: `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- **Action repeat**: High-level actions repeated at low-level via `high_level_action_repeat`
- **Action scaling**: High-level actions are clipped to [-1, 1], scaled by `action_scale`, then mapped to
  base commands (vx × 0.6, vy × 0.2, vyaw × 0.8) in `update_velocity_commands`.
- **High-level speed limits**: In `update_velocity_commands` (`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`),
  high-level actions are clipped to [-1, 1], multiplied by `HighLevelNavigationConfig.action_scale` (default [1.0, 1.0, 1.0]),
  then mapped to base commands with fixed multipliers (vx × 0.6, vy × 0.2, vyaw × 0.8). With default scaling, the effective
  command ranges are vx ∈ [-0.6, 0.6], vy ∈ [-0.2, 0.2], vyaw ∈ [-0.8, 0.8].
  注意：后续所有改动都不能更改这个高层速度限制相关的内容

### Reward Shaping PPO
- **Training script**: `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
- **RL algorithm**: Standard PPO (`rsl_rl/algorithms/ppo.py`)
- **Reward design**: Reference-style shaping aligned with Gazebo baseline:
  - **Success**: +`success_reward` when within `goal_reached_dist`
  - **Collision**: -`collision_penalty` when within `collision_dist`
  - **Dense step reward**: `forward_reward_scale * dot(v_cmd_xy, target_dir_body) - yaw_penalty_scale * angle_error - obstacle_penalty_scale * r3(min_laser)`
    (uses executed velocity commands from `base_env.commands`; `angle_error` is the heading difference between command direction and target direction, masked when command speed is near zero)
  - **Goal progress**: `goal_progress_scale * (prev_goal_dist - reach_metric)` (not applied on terminated/truncated steps)
  - **Obstacle penalty**: `r3(min_laser) = max(1 - min_laser / obstacle_avoid_dist, 0)`
  - **Timeout**: optional `timeout_penalty`
- **Reward parameters**: `GO2HighLevelCfg.reward_shaping` in `legged_gym_go2/legged_gym/envs/go2/go2_config.py`

### Algorithm Details
- **PPO policy loss (clipped)**:
  - `ratio = exp(logp_new - logp_old)`
  - `L_clip = mean(max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps)))`
- **Value loss (clipped)**:
  - `V_clip = V_old + clip(V - V_old, -eps, +eps)`
  - `L_v = mean(max((V - R)^2, (V_clip - R)^2))`
- **Entropy regularization**: `-entropy_coef * entropy`
- **GAE advantages** (time_out bootstrap supported):
  - `delta_t = r_t + gamma * (1 - done_t) * V_{t+1} - V_t`
  - `A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}`
  - `R_t = A_t + V_t`
  - Advantages are normalized per rollout.
- **Termination logic**:
  - `collision = min_hazard_distance < collision_dist`
  - `success = reach_metric <= goal_reached_dist`
  - `terminated = collision OR success`
  - `truncated = time_out AND NOT terminated`
  - PPO bootstraps only on `truncated` via `time_outs`.

### Hazard Distance for Reward
- **Nearest hazard distance** (obstacle surface or boundary) is computed in
  `legged_gym_go2/legged_gym/envs/go2/go2_env.py` and stored as
  `self.min_hazard_distance` for reward shaping.

### High-Level Observations
- **Base features (8)**:
  1. `cos(heading)`
  2. `sin(heading)`
  3. `body_vx` (scaled)
  4. `body_vy` (scaled)
  5. `yaw_rate` (scaled)
  6. `dist_to_target` (normalized)
  7. `target_dir_body_x`
  8. `target_dir_body_y`
- **Target lidar bins**: soft target direction encoding (optional)
- **Obstacle lidar bins**: manual lidar sector encoding (optional)
- **Total dim**: `8 + target_lidar_num_bins + lidar_num_bins` when manual lidar is enabled.

## Key Components

1. **Environments**: `legged_gym_go2/legged_gym/envs/go2/`
2. **RL Algorithms**: `rsl_rl/algorithms/` (PPO)
3. **Training Scripts**: `legged_gym_go2/legged_gym/scripts/`
4. **Deployment**: `legged_gym_go2/deploy/`
5. **Configuration**: Python config classes in `legged_gym_go2/legged_gym/envs/go2/go2_config.py`

## Training Flow (Reward Shaping)

1. Reset env → get obs and metrics
2. Rollout horizon steps:
   - Sample action, step env
   - Compute reward (success/collision/dense shaping)
   - Track dones and timeouts
3. Compute GAE returns and PPO update
4. Log `success`, `cost`, `avg_reward`, `policy_loss`, `value_loss`

### Logging Metrics
- `success`: fraction of episodes ended in the rollout that reached target before collision
- `reach`: fraction of ended episodes that reached the target (regardless of collisions)
- `collision`: fraction of ended episodes that collided
- `timeout`: fraction of ended episodes that ended by timeout
- `cost`: average high-level steps to reach target (successful episodes only)
- `avg_reward`: mean step reward over the rollout horizon
- `proj`: mean target-direction projection of executed velocity commands
- `angle`: mean angle error between commanded direction and target direction
- `progress`: mean goal progress term (distance reduction)
- `obstacle`: mean obstacle penalty term
- `goal_dist`: mean distance to target over the rollout
- `min_hazard`: mean nearest hazard distance over the rollout
- `cmd_speed`: mean commanded speed magnitude over the rollout
- `action_sat`: fraction of actions near saturation (|a| > 0.95)
- `action_std`: mean policy action std (exploration scale)
- `policy_loss`: PPO surrogate loss (clipped objective)
- `value_loss`: critic loss (clipped value regression)
- `approx_kl`: approximate KL divergence between old/new policy
- `clip_frac`: fraction of samples clipped by PPO ratio
- `elapsed`: wall-clock time since previous log line

## Common Commands

### Training
```bash
python legged_gym_go2/legged_gym/scripts/train_reward_shaping.py --headless=true --num_envs=32
```

### Evaluation and Visualization
```bash
python legged_gym_go2/legged_gym/scripts/play_reach_avoid.py --model_path=logs/high_level_go2_reward_shaping/.../model_1000.pt
```

### Deployment
```bash
python legged_gym_go2/deploy/deploy_mujoco/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml
```

## Key Parameters

Reward shaping parameters (in `GO2HighLevelCfg.reward_shaping`):
- `goal_reached_dist`
- `collision_dist`
- `obstacle_avoid_dist`
- `success_reward`
- `collision_penalty`
- `timeout_penalty`
- `reward_scale`

Termination distances (in `GO2RoughCfg.rewards_ext`):
- `goal_reached_dist`
- `collision_dist`
- `unsafe_radius_h_eval_scale`

## Development Notes

- **PPO advantages**: Computed by standard GAE in `rsl_rl/rsl_rl/storage/rollout_storage.py`
- **Timeout handling**: `time_outs` bootstrap is applied in `rsl_rl/algorithms/ppo.py`
- **Success metric**: In `train_reward_shaping.py`, success = reached before collision for episodes that end within the rollout

## Troubleshooting

- If success stays near zero, adjust `success_reward`, `collision_penalty`, or `obstacle_avoid_dist`
- If value_loss spikes, lower `learning_rate` or increase `reward_clip`

### notice
执行所有脚本之前，必须先使用 conda activate unitree-rl 激活 unitree-rl 环境，之后使用 python 执行脚本。
