# Go2 Hierarchical Reward-Shaping RL

## Project Overview
This repository implements a hierarchical reinforcement learning system for Unitree Go2 navigation. The low-level locomotion controller is fixed and pre-trained, while the high-level navigation policy is trained with reward-shaping PPO to reach a target while avoiding obstacles and boundaries.

## Key Focus
- High-level navigation trained with dense reward shaping.
- Low-level locomotion policy kept fixed for stable gait execution.
- Safety-aware termination via collision and boundary checks.

## Architecture
- **Low-level (locomotion)**: velocity commands -> joint actions  
  `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- **High-level (navigation)**: observations -> velocity commands  
  `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- **Hierarchical wrapper**: bridges high/low levels and handles action repeat  
  `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

## Reward-Shaping PPO (High-Level)
The high-level policy is optimized with standard PPO using a dense, goal-directed reward:
- Target-direction projection of executed velocity commands
- Angle error penalty to align commanded direction with target
- Obstacle proximity penalty based on nearest hazard distance
- Goal progress reward (distance reduction)
- Terminal rewards for success/collision/timeout

Full algorithm details and formulas are in: `RewardShapingPPODesign.md`

## Training
Before running any script, activate the environment:
```bash
conda activate unitree-rl
```

Train the high-level policy:
```bash
python legged_gym_go2/legged_gym/scripts/train_reward_shaping.py --headless=true --num_envs=32
```

## Evaluation
```bash
python legged_gym_go2/legged_gym/scripts/play_reach_avoid.py --model_path=logs/logs/high_level_go2_reward_shaping/.../model_1000.pt
```

## Configuration Entry Points
- Reward shaping parameters:  
  `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2HighLevelCfg.reward_shaping`)
- PPO hyperparameters:  
  `legged_gym_go2/legged_gym/envs/go2/go2_config.py` (`GO2HighLevelCfgPPO`)

## Logs
Training logs and checkpoints are stored under:
```
logs/logs/<experiment_name>/<timestamp>/
```

## Notes
- High-level velocity limits are enforced in `update_velocity_commands` and are part of the project constraints.
- Large `.pt` checkpoints and logs are ignored by default via `.gitignore`.
