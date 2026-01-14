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
  `angle`, `progress`, `obstacle`, `goal_dist`, `min_hazard`, `cmd_speed`, `action_sat`,
  `action_std`, `policy_loss`, `value_loss`, `approx_kl`, `clip_frac`, `elapsed`.

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
