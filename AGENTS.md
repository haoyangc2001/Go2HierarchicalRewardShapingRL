# Repository Guidelines

## Project Overview
MCRA_RL is a hierarchical reinforcement learning system for Unitree Go2 navigation. The low-level locomotion policy is fixed and pre-trained, while the high-level policy is trained with reward shaping PPO to reach a target and avoid obstacles/boundaries.

## Key Paths
- Environments: `legged_gym_go2/legged_gym/envs/go2/`
- Training scripts: `legged_gym_go2/legged_gym/scripts/`
- RL algorithms: `rsl_rl/rsl_rl/algorithms/`
- Deployment: `legged_gym_go2/deploy/`
- Logs/checkpoints: `/home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/`

## Hierarchical RL Structure
- Low-level (locomotion): `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- High-level (navigation wrapper): `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- Hierarchical wrapper: `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- High-level actions are repeated at low level via `GO2HighLevelCfg.env.high_level_action_repeat`.

## Environment Overview
### Low-Level Environment (Locomotion)
- Implements `GO2Robot` by extending `LeggedRobot`.
- `step()` returns `(obs, privileged_obs, reward, done, info)`.
- `reset()` calls `reset_idx` then performs a zero-action `step` to prime observations.
- `_compute_safety_metrics()` computes:
  - `avoid_metric`: positive inside unsafe regions.
  - `reach_metric`: XY distance to target center.
  - `min_hazard_distance`: nearest hazard surface distance (min of obstacle surface distance and boundary distance).
- `check_termination()` augments termination with reach/avoid checks when `terminate_on_reach_avoid` is enabled; collision prefers `min_hazard_distance < collision_dist`.

### High-Level Navigation Wrapper
- Builds high-level observations from low-level state:
  - Base 8 dims: `cos(heading)`, `sin(heading)`, `body_vx`, `body_vy`, `yaw_rate`, `reach_metric` (scaled), `target_dir_body_x`, `target_dir_body_y`.
  - Optional target lidar bins (smooth angular binning with distance decay).
  - Optional obstacle/boundary lidar bins (max intensity per bin, boundary handled via ray intersections).
- Exposes helpers to derive distances from observations:
  - Target distance from target lidar intensity (or normalized distance if target lidar disabled).
  - Hazard distance from obstacle/boundary lidar intensity.
- Maps high-level actions to low-level velocity commands in `update_velocity_commands`.

### Hierarchical Wrapper
- Loads a fixed low-level policy via `OnPolicyRunner` and exposes a high-level interface.
- Each high-level action is repeated for `high_level_action_repeat` low-level steps; dones are aggregated.
- Reward and termination are computed inside the hierarchical env using true `reach_metric` and min hazard distance (minimum across repeated low-level steps).
- `step()` returns `(obs, reward, done, info)` only.
- Info fields (used for logging/diagnostics):
  - `time_outs`, `reached`, `success`, `collision`, `terminated`, `truncated`
  - `target_distance`, `target_distance_est`, `reach_metric`
  - `min_hazard_distance`, `min_hazard_distance_est`, `min_hazard_distance_true`
  - `boundary_distance`, `obstacle_surface_distance`
  - `base_lin_vel`, `desired_commands`
  - `progress`, `safety_penalty`, `smooth_penalty`, `command_speed`, `body_speed`, `command_delta`, `reward_clip_frac`

### Vectorized Adapter
- `HierarchicalVecEnv` provides a vectorized API for PPO training while delegating to the hierarchical environment.
- Returns `(obs, reward, done, info)` and sets `num_privileged_obs = None`.

### Environment Configuration
- Low-level base config: `GO2RoughCfg` (terrain, domain randomization, rewards, obstacle/target layout).
- High-level config: `GO2HighLevelCfg` (lidar, action repeat, reward shaping).
- Observation dimension is computed as `8 + target_lidar_num_bins + lidar_num_bins`.
- `GO2HighLevelCfg.reach_metric_scale` controls the scaling of obs[5].

## High-Level Action Mapping (Current)
In `update_velocity_commands`:
- Clip high-level actions to `[-1, 1]`.
- Multiply by `HighLevelNavigationConfig.action_scale`.
- Map to base commands:
  - `vx = action[0] * 0.6`
  - `vy = action[1] * 0.2`
  - `vyaw = action[2] * 0.8`
With default `action_scale = [1.3, 1.0, 1.0]`, the effective command ranges are:
`vx in [-0.78, 0.78]`, `vy in [-0.2, 0.2]`, `vyaw in [-0.8, 0.8]`.

## High-Level Observations
- Base features (8):
  1) `cos(heading)`
  2) `sin(heading)`
  3) `body_vx` (scaled, clipped)
  4) `body_vy` (scaled, clipped)
  5) `yaw_rate` (scaled, clipped)
  6) `reach_metric` (true XY distance to target center, scaled by `GO2HighLevelCfg.reach_metric_scale`)
  7) `target_dir_body_x`
  8) `target_dir_body_y`
- Optional target lidar bins: `target_lidar_num_bins`
- Optional obstacle/boundary lidar bins: `lidar_num_bins`
- Total dim: `8 + target_lidar_num_bins + lidar_num_bins` when manual lidar is enabled.

## Reward Design (High Level)
- Implemented in `HierarchicalGO2Env._compute_reward`.
- Target distance uses true `reach_metric`; hazard distance uses true min hazard distance.
- Dense terms (per high-level step):
  - Progress: `progress_scale * (prev_target_distance - target_distance)`.
  - Safety shaping:
    - If `d_min >= d_safe`, `safety = 0`.
    - Else `safety = -((d_safe - d_min)/d_safe)^2`.
  - Smoothness: `- smooth_scale * ||cmd_t - cmd_{t-1}||^2`.
- Terminal terms:
  - Success bonus when `target_distance <= goal_reached_dist` on a done step.
  - Collision penalty when `hazard_distance <= collision_dist` (or base failure) on a done step.
  - Timeout penalty on truncation.
- Reward scaling and clipping: `reward_scale`, `reward_clip`.
- Done flags follow the base environment resets to avoid desyncs; success/collision are derived for logging.

## PPO Training (High Level)
- Training script: `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`.
- PPO implementation: `rsl_rl/rsl_rl/algorithms/ppo.py`.
- The training loop consumes environment rewards directly (no external shaping).
- PPO bootstraps only on `time_outs` in `info`.

## Safety Metrics
- Computed in `legged_gym_go2/legged_gym/envs/go2/go2_env.py`:
  - `avoid_metric`, `reach_metric`, `min_hazard_distance`, `obstacle_surface_distance`, `boundary_distance`.
- Used for termination and diagnostics; not returned in low-level `step()`.
- `boundary_distance < 0` indicates out-of-bounds; base env resets immediately.

## Logging and Outputs
- Training logs/checkpoints are saved to:
  `/home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/<timestamp>/`
- The training log file is `training.log`.
- Logged metrics include: `success`, `reach`, `collision`, `boundary_collision_rate`, `obstacle_collision_rate`,
  `timeout`, `cost`, `avg_reward`, `progress`, `safety`, `smooth`, `goal_dist`, `min_hazard`,
  `reward_clip`, `action_std`, `policy_loss`, `value_loss`, `approx_kl`, `clip_frac`, `lr`, `ep_len_mean`.

### Training Log Field Meanings
- `iter`: Iteration index (one rollout + one PPO update), starting from 1.
- `success`: Success rate over finished episodes; success means reached target without collision.
- `reach`: Reach rate over finished episodes; `target_distance <= goal_reached_dist` on a done step.
- `collision`: Collision rate over finished episodes; from `info.collision`.
- `boundary_collision_rate`: Episode-level collision rate attributed to boundary hazards.
- `obstacle_collision_rate`: Episode-level collision rate attributed to obstacles.
- `timeout`: Timeout rate over finished episodes; `time_outs` and not reached/collision.
- `cost`: Average high-level steps for successful episodes (lower is faster).
- `avg_reward`: Mean reward per step (includes terminal rewards and clipping).
- `progress`: Mean distance progress `prev_target_distance - target_distance`.
- `safety`: Mean safety shaping term (non-positive near hazards).
- `smooth`: Mean smoothness penalty term.
- `goal_dist`: Mean target distance (meters), derived from `reach_metric`.
- `min_hazard`: Mean nearest hazard distance (meters, true value).
- `reward_clip`: Fraction of rewards clipped.
- `action_std`: Mean policy std (exploration strength).
- `policy_loss`, `value_loss`: PPO losses.
- `approx_kl`, `clip_frac`: PPO diagnostics.
- `lr`: Current PPO learning rate.
- `ep_len_mean`: Episode length mean.

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
python legged_gym_go2/legged_gym/scripts/plot_training_results.py /home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/<timestamp>/training.log
```

Deploy in Mujoco (example):
```bash
python legged_gym_go2/deploy/deploy_mujoco/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml
```

## Development Notes
- `train_reward_shaping.py` overrides CLI args in `__main__` (headless + device IDs). Edit there if you need different devices.
- `HierarchicalGO2Env` sets `terminate_on_reach_avoid` based on reward shaping flags.
- The low-level policy is fixed; high-level training should not modify it.
- Reward computation lives inside the hierarchical environment and uses true distances.
- If you change lidar bin counts or ranges, update `GO2HighLevelCfg` and the computed observation dimension.
