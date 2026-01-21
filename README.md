# Go2 Hierarchical Reward-Shaping RL

## Project Overview
This repository implements a hierarchical reinforcement learning system for Unitree Go2 navigation in IsaacGym. The low-level locomotion controller is fixed and pre-trained, while the high-level navigation policy is trained with reward-shaping PPO to reach a target while avoiding obstacles and boundaries.

## Architecture
- **Low-level (locomotion)**: velocity commands -> joint actions
  `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- **High-level (navigation)**: observations -> velocity commands
  `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- **Hierarchical wrapper**: bridges high/low levels and handles macro steps
  `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

## High-Level Reward Shaping
Reward is computed inside `HierarchicalGO2Env._compute_reward` using true target distance and min hazard distance.

Per high-level step:
```
progress = d_prev - d_t
safety = 0                                 if d_min >= d_safe
         -((d_safe - d_min) / d_safe)^2    otherwise
smooth = ||cmd_t - cmd_{t-1}||^2

reward = progress_scale * progress
       + safe_scale * safety
       - smooth_scale * smooth
```
Terminal terms (on done steps):
- Success: `+goal_reward` when `d_t <= goal_reached_dist`
- Collision: `-collision_penalty` when `d_min <= collision_dist` (or base failure)
- Timeout: `-timeout_penalty` on truncation

Optional global scaling and clipping: `reward_scale`, `reward_clip`.

## Training
Before running any script:
```bash
conda activate unitree-rl
```

Train the high-level policy:
```bash
python legged_gym_go2/legged_gym/scripts/train_reward_shaping.py --headless=true --num_envs=32
```

## Logs
Training logs and checkpoints are stored under:
```
/home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/<timestamp>/
```
The training log file is `training.log`. Logged metrics:
- `success`, `reach`, `collision`, `boundary_collision_rate`, `obstacle_collision_rate`, `timeout`, `cost`
- `avg_reward`, `progress`, `safety`, `smooth`, `goal_dist`, `min_hazard`, `reward_clip`
- `action_std`, `policy_loss`, `value_loss`, `approx_kl`, `clip_frac`, `lr`, `ep_len_mean`

## Configuration Entry Points
- Reward shaping parameters:
  `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2HighLevelCfg.reward_shaping`)
- PPO hyperparameters:
  `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2HighLevelCfgPPO`)
- Low-level checkpoint path:
  `GO2HighLevelCfgPPO.runner.low_level_model_path`

## Notes
- High-level actions are repeated via `GO2HighLevelCfg.env.high_level_action_repeat`.
- Velocity commands are scaled in `update_velocity_commands` (see `high_level_navigation_env.py`).
- `train_reward_shaping.py` overrides device args in `__main__` for headless runs.
