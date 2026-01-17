# Repository Guidelines

## Project Overview
MCRA_RL is a hierarchical reinforcement learning system for Unitree Go2 navigation. The low-level locomotion policy is fixed and pre-trained, while the high-level policy is trained with reward shaping PPO to reach a target and avoid obstacles/boundaries.

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
- Implements `GO2Robot` by extending `LeggedRobot`.
- `step()` returns the standard `(obs, privileged_obs, reward, done, info)` tuple (no safety metrics in the return).
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
- Maps high-level actions to low-level velocity commands in `update_velocity_commands` with fixed scaling and ranges (see mapping below).

### Hierarchical Wrapper
- Loads a fixed low-level policy via `OnPolicyRunner` and exposes a high-level interface.
- Each high-level action is repeated for `high_level_action_repeat` low-level steps; dones are aggregated.
- Reward and termination signals are computed inside the hierarchical env using lidar-derived distances.
- `step()` returns `(obs, reward, done, info)` only.
- Info fields (used for logging/diagnostics):
  - `time_outs`, `reached`, `success`, `collision`, `terminated`, `truncated`
  - `target_distance`, `min_hazard_distance`
  - `boundary_distance`, `obstacle_surface_distance`
  - `base_lin_vel`, `desired_commands`
  - `progress`, `alignment`, `obstacle_penalty`, `command_speed`, `body_speed`, `command_delta`, `reward_clip_frac`

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
With default `action_scale = [1, 1, 1]`, the effective command ranges are:
`vx in [-0.6, 0.6]`, `vy in [-0.2, 0.2]`, `vyaw in [-0.8, 0.8]`.

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
- Target distance uses true `reach_metric`; hazard distance uses obstacle/boundary lidar intensity.
- Dense terms:
  - Progress: `progress_scale * (prev_target_distance - reach_metric)` (masked on terminated/truncated steps).
  - Alignment: `alignment_scale * dot(v_body_xy, target_dir_body)`.
  - Obstacle penalty: `- obstacle_penalty_scale * r(d)` where `r(d) = max((obstacle_avoid_dist - hazard_distance)/obstacle_avoid_dist, 0)`.
  - Yaw penalty: `- yaw_rate_scale * abs(yaw_rate)`.
  - Action smoothness: `- action_smooth_scale * ||cmd_t - cmd_{t-1}||`.
  - Optional body speed bonus and idle penalty.
- Terminal terms:
  - Success bonus when `target_distance <= goal_reached_dist` on a done step.
  - Collision penalty when `hazard_distance <= collision_dist` (or base failure) on a done step.
  - Timeout penalty on truncation.
- Done flags follow the base environment resets to avoid desyncs; success/collision are derived for logging.

## PPO Training (High Level)
- Training script: `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`.
- PPO implementation: `rsl_rl/rsl_rl/algorithms/ppo.py`.
- The training loop consumes environment rewards directly (no external shaping).
- PPO bootstraps only on `time_outs` in `info`.

## Safety Metrics
- Computed in `legged_gym_go2/legged_gym/envs/go2/go2_env.py`:
  - `avoid_metric`, `reach_metric`, `min_hazard_distance`, `obstacle_surface_distance`, `boundary_distance`.
- Used for termination and diagnostics; not returned in `step()`.
- `boundary_distance < 0` indicates out-of-bounds; base env resets immediately.

## Logging and Outputs
- Training logs/checkpoints are saved to:
  `/home/caohy/repositories/MCRA_RL/logs/<experiment_name>/<timestamp>/`
- The training log file is `training.log`.
- Logged metrics include: `success`, `reach`, `collision`, `timeout`, `cost`, `avg_reward`, `proj`,
  `progress`, `obstacle`, `goal_dist`, `min_hazard`, `cmd_speed`, `body_speed`,
  `speed_ratio`, `speed_ratio_active`, `cmd_delta`, `cmd_zero`, `action_sat`, `done_frac`,
  `action_std`, `policy_loss`, `value_loss`, `approx_kl`, `clip_frac`, `elapsed`,
  plus PPO/diagnostic metrics such as `entropy`, `lr`, `grad_norm`, `value_clip_frac`, `Vmean`, `Vstd`,
  `Rmean`, `Rstd`, `adv_mean`, `adv_std`, `reward_clip`, `hazard_p10`, `hazard_p50`,
  `hazard_p90`, `boundary_violation`,
  `boundary_collision_rate`, `obstacle_collision_rate`,
  `ep_len_mean`, `ep_len_std`, `init_goal_dist`.

### Training Log Field Meanings
- `iter`: Iteration index (one rollout + one PPO update), starting from 1.
- `success`: Success rate over finished episodes; success means reached target without collision in that episode.
- `reach`: Reach rate over finished episodes; `target_distance <= goal_reached_dist` on a done step.
- `collision`: Collision rate over finished episodes; `hazard_distance <= collision_dist` on a done step.
- `timeout`: Timeout rate over finished episodes; `time_outs` and not reached/collision.
- `cost`: Average high-level steps for successful episodes (lower is faster).
- `avg_reward`: Mean reward per step (includes terminal rewards and clipping).
- `proj`: Mean alignment `dot(v_body_xy, target_dir_body)`.
- `progress`: Mean distance progress `prev_target_distance - target_distance` (masked on terminated/truncated steps).
- `obstacle`: Mean obstacle penalty term `r(hazard_distance)` in `[0, 1]`.
- `goal_dist`: Mean target distance (meters), derived from `reach_metric`.
- `min_hazard`: Mean nearest hazard distance (meters, true value).
- `cmd_speed`: Mean planar command speed `||v_cmd_xy||`.
- `body_speed`: Mean planar body speed `||v_body_xy||`.
- `speed_ratio`: Mean `body_speed / cmd_speed`.
- `speed_ratio_active`: Mean `speed_ratio` where `cmd_speed > 0.1`.
- `cmd_delta`: Mean `||cmd_t - cmd_{t-1}||`.
- `cmd_zero`: Fraction where `cmd_speed < 0.1`.
- `action_sat`: Fraction where `|a| > 0.95`.
- `action_std`: Mean policy std (exploration strength).
- `policy_loss`, `value_loss`: PPO losses.
- `approx_kl`, `clip_frac`: PPO diagnostics.
- `entropy`, `lr`, `grad_norm`, `value_clip_frac`: PPO diagnostics.
- `Vmean`, `Vstd`: Value function output mean/std.
- `Rmean`, `Rstd`: Return mean/std.
- `adv_mean`, `adv_std`: Advantage mean/std.
- `reward_clip`: Fraction of rewards clipped.
- `hazard_p10/p50/p90`: Quantiles of `min_hazard_distance`.
- `boundary_violation`: Fraction where `boundary_distance < 0`.
- `boundary_collision_rate`, `obstacle_collision_rate`: Episode-level collision rates by source.
- `done_frac`: Mean done ratio per step.
- `ep_len_mean`, `ep_len_std`: Episode length mean/std.
- `init_goal_dist`: Mean initial target distance at the start of an iteration.
- `elapsed`: Wall-clock seconds between logs.

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
- `train_reward_shaping.py` overrides CLI args in `__main__` (headless + device IDs). Edit there if you need different devices.
- `HierarchicalGO2Env` sets `terminate_on_reach_avoid` based on reward shaping flags.
- The low-level policy is fixed; high-level training should not modify it.
- Reward computation lives inside the hierarchical environment and uses lidar-derived distances.
- If you change lidar bin counts or ranges, update `GO2HighLevelCfg` and the computed observation dimension.
