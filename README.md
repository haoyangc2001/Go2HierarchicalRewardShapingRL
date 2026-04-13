# Go2 Hierarchical Reward-Shaping RL

## 项目概述

Go2 Hierarchical Reward-Shaping RL 是一个面向 Unitree Go2 的分层导航强化学习项目。项目在 Isaac Gym 仿真环境中固定低层运动控制器，仅训练高层导航策略，使机器人在多障碍与边界约束环境中完成“朝目标前进、避免碰撞、保持动作稳定”的联合任务。

本项目采用标准 PPO 作为优化器主体，在高层决策时间尺度上引入 Reward Shaping 奖励设计，以提升训练早期的可学习性、成功率和避障稳定性。

## 核心功能

### 1. 分层控制架构

#### 低层策略（Locomotion）
- 功能：将速度命令转换为关节级动作，负责稳定步态与速度跟踪。
- 特点：低层策略已预训练完成，在高层训练阶段固定参数，不参与更新。
- 相关位置：
  - 环境主体：`legged_gym_go2/legged_gym/envs/go2/go2_env.py`
  - 低层策略加载：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

#### 高层策略（Navigation）
- 功能：根据目标信息、速度状态、障碍物与边界感知结果输出高层导航动作。
- 输出动作：三维速度指令 `[v_x, v_y, \omega_z]`。
- 相关位置：
  - 高层观测与动作封装：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
  - 高层训练脚本：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`

#### 分层桥接逻辑
- `HierarchicalGO2Env` 负责把高层动作转换为底层速度命令，并在一个高层 step 内重复执行多个低层控制周期。
- 当前默认 `high_level_action_repeat = 5`，即高层每决策一次，底层连续执行 5 个仿真步。
- 相关位置：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

### 2. Reward Shaping PPO 训练机制

高层策略仍使用标准 PPO 更新，但奖励不再是单一稀疏成败信号，而是由以下项共同组成：

- 目标推进奖励：鼓励机器人持续缩短到目标的距离。
- 近障惩罚：在进入危险区域前就给出稠密负反馈。
- 动作平滑惩罚：抑制速度指令跳变，减轻底层跟踪压力。
- 终止奖励/惩罚：对成功到达、碰撞、超时进行显式塑形。
- 可选目标速度奖励：在安全前提下鼓励沿目标方向产生前进速度。

详细算法原理、公式、配置项和实现映射见 `ALGORITHM_DESIGN.md`。

### 3. 训练过程监控

训练脚本会在每个 iteration 输出完整统计信息，便于定位“能否到达”“为何碰撞”“PPO 是否数值稳定”等问题。当前日志重点指标包括：

- 任务结果指标：`success`、`reach`、`collision`、`timeout`
- 碰撞拆分指标：`boundary_collision_rate`、`obstacle_collision_rate`
- 奖励分解指标：`avg_reward`、`progress`、`safety`、`smooth`、`target_speed`
- 环境状态指标：`goal_dist`、`min_hazard`、`ep_len_mean`
- PPO 训练指标：`policy_loss`、`value_loss`、`approx_kl`、`clip_frac`、`lr`
- 稳定性指标：`adv_mean`、`adv_std`、`ratio_mean`、`ratio_std`、`ppo_skipped`

## 技术架构

### 主要模块

| 模块 | 作用 | 位置 |
|------|------|------|
| `HighLevelNavigationEnv` | 组织高层观测、动作映射、手工激光雷达特征 | `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` |
| `HierarchicalGO2Env` | 封装高低层交互、奖励计算、终止条件与统计信息 | `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py` |
| `PPO` | 高层策略优化器，包含自适应 KL 学习率与 value clipping | `rsl_rl/rsl_rl/algorithms/ppo.py` |
| `RolloutStorage` | 存储高层 rollout，并计算 GAE/returns | `rsl_rl/rsl_rl/storage/rollout_storage.py` |
| `ActorCritic` | 高层 actor-critic 网络，actor 输出 tanh-squash 高斯动作 | `rsl_rl/rsl_rl/modules/actor_critic.py` |
| `train_reward_shaping.py` | 训练入口、日志记录、checkpoint 保存 | `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py` |

### 高层观测空间

当前高层观测由以下部分组成：

1. 基础状态 8 维
   - `cos(heading)`
   - `sin(heading)`
   - 机体坐标系线速度 `v_x, v_y`
   - 偏航角速度 `yaw_rate`
   - 缩放后的目标距离度量
   - 目标方向单位向量 2 维
2. 目标方向强度编码 16 维
3. 障碍物与边界的手工 lidar 编码 16 维

因此默认总观测维度为 40。

### 高层动作空间

- 动作维度：3
- 动作含义：`[v_x, v_y, \omega_z]`
- 策略输出先经 tanh 压缩到 `[-1, 1]`
- 再乘以 `action_scale = [1.3, 1.0, 1.0]`
- 最终映射到底层命令时，额外使用：
  - `x` 方向系数 `0.6`
  - `y` 方向系数 `0.2`
  - `yaw` 系数 `0.8`

## 环境与任务设置

### 导航目标

- 机器人在平面环境中从起点出发，前往目标区域。
- 成功判定阈值由 `goal_reached_dist` 控制，当前默认值为 `0.3` 米。

### 危险区域

- 环境中包含多个圆柱障碍物，同时训练场地边界也作为危险源参与最小风险距离计算。
- 碰撞判定阈值由 `collision_dist` 控制，当前默认值为 `0.35` 米。
- 安全 shaping 距离阈值由 `safe_distance` 控制，当前默认值为 `1.5` 米。

### 时间尺度

- 高层 episode 长度：`episode_length_s = 40`
- 高层 rollout 长度：`num_steps_per_env = 400`
- 高层动作重复次数：`high_level_action_repeat = 5`

## 安装与环境准备

### 基础依赖

- Ubuntu
- NVIDIA GPU
- CUDA 环境
- Conda
- PyTorch
- Isaac Gym

### 推荐安装流程

1. 克隆仓库
   ```bash
   git clone https://github.com/haoyangc2001/Go2HierarchicalRewardShapingRL.git
   cd Go2HierarchicalRewardShapingRL
   ```

2. 创建并激活 Conda 环境
   ```bash
   conda create -n unitree-rl python=3.8
   conda activate unitree-rl
   ```

3. 安装 PyTorch
   ```bash
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. 安装 Isaac Gym
   ```bash
   cd isaacgym/python
   pip install -e .
   cd ../..
   ```

5. 安装项目内 Python 包
   ```bash
   pip install -e legged_gym_go2
   pip install -e rsl_rl
   ```

## 训练方法

### 训练入口

高层 Reward-Shaping PPO 训练脚本：

```bash
python legged_gym_go2/legged_gym/scripts/train_reward_shaping.py --headless=true --num_envs=32
```

### 启动前建议检查

- `legged_gym_go2/legged_gym/envs/go2/go2_config.py` 中的高层训练配置是否符合当前机器资源。
- `GO2HighLevelCfgPPO.runner.low_level_model_path` 是否指向可用的低层策略 checkpoint。
- `train_reward_shaping.py` 中设备参数是否适合本机，例如当前 `__main__` 会直接覆盖为 `cuda:1`。

### 训练输出

训练日志和模型默认保存在相对目录：

```text
logs/high_level_go2_Reward_Shaping/<timestamp>/
```

目录内通常包含：

- `training.log`
- `model_<iteration>.pt`
- `model_final.pt`

## 关键配置说明

### Reward Shaping 参数

位于 `legged_gym_go2/legged_gym/envs/go2/go2_config.py` 的 `GO2HighLevelCfg.reward_shaping`：

| 参数 | 含义 | 当前默认值 |
|------|------|------------|
| `goal_reached_dist` | 到达判定距离 | `0.3` |
| `collision_dist` | 碰撞判定距离 | `0.35` |
| `safe_distance` | 安全惩罚开始生效的距离 | `1.5` |
| `progress_scale` | 距离推进项权重 | `4.5` |
| `target_speed_scale` | 目标方向速度奖励权重 | `0.1` |
| `goal_reward` | 成功终止奖励 | `80.0` |
| `safe_scale` | 近障惩罚权重 | `4.0` |
| `smooth_scale` | 动作平滑惩罚权重 | `0.05` |
| `collision_penalty` | 碰撞终止惩罚 | `120.0` |
| `timeout_penalty` | 超时终止惩罚 | `10.0` |
| `reward_scale` | 全局奖励缩放 | `1.0` |
| `reward_clip` | 奖励裁剪阈值 | `200.0` |

### PPO 参数

位于 `legged_gym_go2/legged_gym/envs/go2/go2_config.py` 的 `GO2HighLevelCfgPPO.algorithm`：

| 参数 | 当前默认值 |
|------|------------|
| `learning_rate` | `3e-5` |
| `clip_param` | `0.07` |
| `value_clip_param` | `0.2` |
| `value_loss_coef` | `0.5` |
| `entropy_coef` | `0.003` |
| `desired_kl` | `0.03` |
| `schedule` | `adaptive` |
| `min_lr` | `1e-5` |
| `max_lr` | `1e-3` |
| `num_learning_epochs` | `2` |
| `num_mini_batches` | `4` |
| `num_steps_per_env` | `400` |
| `max_grad_norm` | `1.0` |

### 网络结构

- Actor 隐层：`[512, 512, 512, 512]`
- Critic 隐层：`[512, 512, 512, 512]`
- 激活函数：继承 `LeggedRobotCfgPPO.policy.activation`
- 动作分布：高斯分布采样后做 `tanh` 压缩
- 初始动作噪声标准差：`0.3`

## 目录说明

```text
.
├── README.md
├── ALGORITHM_DESIGN.md
├── legged_gym_go2/
│   └── legged_gym/
│       ├── envs/go2/
│       └── scripts/
├── rsl_rl/
└── logs/
```

## 文档索引

- 算法原理与公式：`ALGORITHM_DESIGN.md`
- 高层配置入口：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
- 训练脚本入口：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`

## 说明

- `AGENTS.md` 与 `DEBUG_SUMMARY.md` 为本地开发辅助文件，已按本仓库忽略规则设置为不再上传。
- 文档中展示的目录均使用仓库相对路径，不包含本机绝对路径信息。
