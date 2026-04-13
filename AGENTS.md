# 仓库指南

## 项目概览
Go2HierarchicalRewardShapingRL 是一个基于 IsaacGym 的 Unitree Go2 分层强化学习导航系统：
- **底层（locomotion）策略固定且已训练**，负责将速度指令稳定跟踪为关节动作。
- **上层（navigation）策略需要训练**，通过 Reward Shaping 的 PPO 学会“到达目标 + 避障/避边界”。
- 上层以低频宏步决策，每个上层动作会在底层重复执行 `high_level_action_repeat` 次。

## 关键路径
- 环境与分层封装：`legged_gym_go2/legged_gym/envs/go2/`
- 训练脚本：`legged_gym_go2/legged_gym/scripts/`
- PPO/算法：`rsl_rl/rsl_rl/algorithms/`
- 分层向量化适配：`legged_gym_go2/legged_gym/utils/hierarchical_env_utils.py`
- 部署：`legged_gym_go2/deploy/`
- 训练日志/模型：`/home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/`

## 系统架构与数据流
### 低层环境（Locomotion）
- 入口类：`GO2Robot`（`legged_gym_go2/legged_gym/envs/go2/go2_env.py`），继承 `LeggedRobot`。
- `reset()`：调用 `reset_idx` 后执行一次零动作 `step()` 以刷新观测。
- `step()`：沿用基类 `LeggedRobot.step()`，返回 `(obs, privileged_obs, reward, done, info)`。
- `_compute_safety_metrics()` 计算安全/目标指标：
  - `avoid_metric`：进入不安全区域时为正。
  - `reach_metric`：到目标中心的 XY 距离。
  - `min_hazard_distance`：到最近危险体表面的距离（障碍物表面与边界距离的最小值）。
  - 同时维护 `obstacle_surface_distance` 与 `boundary_distance` 供诊断。
- `check_termination()`：在 `terminate_on_reach_avoid` 打开时，结合 `collision_dist` 与 `goal_reached_dist` 扩展终止逻辑。
- 边界与出生点：
  - 边界使用 `terrain_length/terrain_width` 定义（配置为 12m × 12m，半径 6m）。
  - `reset` 时在边界内随机采样，避免离障碍/边界过近。

### 高层导航包装（HighLevelNavigationEnv）
文件：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- **观测构建**（基础 8 维 + 可选雷达）：
  1) `cos(heading)`  2) `sin(heading)`
  3) `body_vx`（缩放并裁剪）  4) `body_vy`（缩放并裁剪）
  5) `yaw_rate`（缩放并裁剪）
  6) `reach_metric`（真值距离 × `reach_metric_scale`）
  7) `target_dir_body_x`  8) `target_dir_body_y`
- **目标雷达**：按角度平滑分配到邻近 bin，强度=1-归一化表面距离。
- **障碍/边界雷达**：每 bin 取最大强度；边界通过射线与矩形边界求交得到强度。
- 提供从观测恢复距离的工具函数：
  - `extract_target_distance()`
  - `extract_hazard_distance()`
- **高层动作到速度指令映射**：`update_velocity_commands()`
  - clip 到 `[-1,1]`，乘以 `action_scale`。
  - 速度命令映射：
    - `vx = action[0] * 0.6`
    - `vy = action[1] * 0.2`
    - `vyaw = action[2] * 0.8`
  - 同步航向控制：`commands[3] = heading + 2.0 * vyaw`。

### 分层封装（HierarchicalGO2Env）
文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- 通过 `task_registry.make_env("go2")` 创建底层环境。
- 使用 `OnPolicyRunner` 加载固定的低层策略（checkpoint 由 `GO2HighLevelCfgPPO.runner.low_level_model_path` 提供）。
- 高层动作重复执行 `high_level_action_repeat` 次：
  - 聚合 `done`，并取 **最小 hazard 距离** 作为该高层步的安全指标。
  - `reach_metric` 取尚未 done 的最新值。
  - `time_outs` 来自底层 `info`。
- Reward 在分层环境中计算（使用真值距离），并与底层 done 对齐避免 desync。

### 向量化适配（HierarchicalVecEnv）
文件：`legged_gym_go2/legged_gym/utils/hierarchical_env_utils.py`
- 为 PPO 暴露统一接口：`(obs, reward, done, info)`。
- `num_privileged_obs = None`。

## 高层观测与维度
- 总维度：`8 + target_lidar_num_bins + lidar_num_bins`（当 `enable_manual_lidar=True`）。
- 默认配置（`GO2HighLevelCfg`）：
  - `reach_metric_scale = 0.2`
  - `lidar_num_bins = 16`
  - `target_lidar_num_bins = 16`
  - `lidar_max_range = 8.0`，`target_lidar_max_range = 8.0`
  - 因此默认高层观测维度 = `8 + 16 + 16 = 40`

## 高层动作范围（当前代码行为）
`HierarchicalGO2Env._update_high_level_config()` 已同步 `GO2HighLevelCfg.action_scale`，因此运行时采用配置值：
- `action_scale = [1.3, 1.0, 1.0]`
- 结合映射后范围：`vx ∈ [-0.78, 0.78]`、`vy ∈ [-0.2, 0.2]`、`vyaw ∈ [-0.8, 0.8]`

## 安全指标与终止逻辑
- 低层 `_compute_safety_metrics()` 计算：
  - `avoid_metric`：进入不安全区域时 > 0（包含边界与障碍）。
  - `min_hazard_distance`：障碍表面与边界距离的最小值。
  - `boundary_distance < 0` 视为越界，立即触发 reset。
- `terminate_on_reach_avoid` 打开时：
  - 若配置了 `collision_dist`，使用 `min_hazard_distance < collision_dist` 判断碰撞。
  - 否则使用 `avoid_metric > 0`。
  - `reach_metric <= goal_reached_dist` 触发到达终止。
- 分层环境将 `terminate_on_reach_avoid` 设置为：
  - `reward_shaping.terminate_on_safety_violation` 或 `reward_shaping.terminate_on_success` 为 True 时开启。

## 高层奖励塑形（HierarchicalGO2Env._compute_reward）
核心项：
- **进展奖励**：`progress = prev_target_distance - target_distance`
- **安全惩罚**：若 `hazard_distance < safe_distance`，惩罚为 `-((safe_distance - d)/safe_distance)^2`
- **平滑惩罚**：`- smooth_scale * ||cmd_t - cmd_{t-1}||^2`
- **可选目标速度奖励**：基于 body 速度在目标方向上的投影（并受安全权重调制）
- **终止奖励/惩罚**：成功奖励、碰撞惩罚、超时惩罚

默认参数（`GO2HighLevelCfg.reward_shaping`）：
- `goal_reached_dist=0.3`，`collision_dist=0.35`，`safe_distance=1.5`
- `progress_scale=4.5`，`safe_scale=4.0`，`smooth_scale=0.05`
- `target_speed_scale=0.1`
- `goal_reward=80.0`，`collision_penalty=120.0`，`timeout_penalty=10.0`
- `reward_scale=1.0`，`reward_clip=200.0`

## PPO 训练流程
脚本：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
- 使用 `HierarchicalVecEnv` 训练上层策略。
- Actor/Critic 观测相同，动作通过 `tanh` squash。
- **PPO bootstrapping** 仅在 `info.time_outs` 为 True 时进行（见 `rsl_rl/rsl_rl/algorithms/ppo.py`）。
- 日志保存到：
  `/home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/<timestamp>/training.log`
- `__main__` 中强制覆盖设备与 headless 参数，如需改设备需在脚本内改。

## 日志字段与含义（training.log）
- `iter`：迭代编号（一次 rollout + 一次 PPO update）。
- `success`：成功率（到达目标且未碰撞）。
- `reach`：到达率（`target_distance <= goal_reached_dist`）。
- `collision`：碰撞率（来自 info/collision）。
- `boundary_collision_rate`：边界引起的碰撞比例。
- `obstacle_collision_rate`：障碍引起的碰撞比例。
- `timeout`：超时率。
- `cost`：成功 episode 的平均步数（越小越好）。
- `avg_reward`：每步平均 reward（包含终止项与裁剪）。
- `progress`：平均进展（`prev_target_distance - target_distance`）。
- `safety`：平均安全惩罚（<=0）。
- `smooth`：平均平滑惩罚。
- `goal_dist`：平均目标距离（米）。
- `min_hazard`：平均最近危险距离（米）。
- `reward_clip`：reward 被裁剪的比例。
- `action_std`：策略输出 std（探索强度）。
- `policy_loss`/`value_loss`：PPO 损失。
- `approx_kl`/`clip_frac`：PPO 诊断指标。
- `adv_mean`/`adv_std`：优势函数均值/标准差（raw，归一化前）。
- `ratio_mean`/`ratio_std`/`ratio_abs_mean`/`ratio_min`/`ratio_max`：PPO ratio 分布统计（`exp(log_ratio)`）。
- `lr`：当前学习率。
- `ppo_updates`/`ppo_skipped`/`ppo_skip_frac`/`ppo_skip_kl`/`ppo_skip_nonfinite`：PPO 更新/跳过统计。
- `ep_len_mean`：平均 episode 长度。
- `target_speed`：目标速度奖励分量均值。
- `progress_var`/`safety_var`/`smooth_var`/`target_speed_var`：对应 reward 分量的方差估计（per-step 均值上的二阶统计）。
- `cmd_goal_corr`：`command_speed` 与 `progress` 的相关性（用作“动作幅度 ↔ 目标距离变化”的近似关联）。


## 调试流程（训练日志）
目标：对给定训练日志做可复现的异常诊断 + 收敛判断，并将结论与历史 debug 记录及代码变更对齐，最终输出可执行的下一步建议。

### 1) 日志分析与异常检测（必须先做）
从日志中抽取并分段统计（前/中/后段或滑动均值）：
- 任务结果：`success`, `reach`, `collision`, `boundary_collision_rate`, `obstacle_collision_rate`, `timeout`, `ep_len_mean`, `cost`
- 学习稳定性：`approx_kl`, `clip_frac`, `policy_loss`, `value_loss`, `lr`, `action_std`, `reward_clip`
- 行为/进展/安全：`avg_reward`, `progress`, `safety`, `smooth`, `goal_dist`, `min_hazard`
> 若你在训练脚本中额外记录了 `command_speed/body_speed/command_delta` 等字段，可作为补充信号参考。

必须判断并记录：
- 成功率趋势：是否提升、是否达到平台、是否后期回落。
- 收敛判断：success 的滑动均值是否在连续区间稳定波动（振幅小且无系统性下降）。
- 异常模式（示例）：  
  - 碰撞主导：`collision` 高且 `timeout` 低，`ep_len_mean` 下降，`min_hazard` 持续偏小。  
  - 进展停滞：`goal_dist` 不降、`progress` 低，`success` 长期停滞。  
  - 奖励饱和：`reward_clip` 长期偏高，`avg_reward` 与 `success` 脱钩。  
  - 学习不稳：`approx_kl/clip_frac` 持续偏高或 `value_loss` 激增。  
  - 探索塌缩：`action_std` 快速接近 0 且 `success` 无提升。  

输出要求：形成结构化分析报告，至少包含：
- 关键区间统计（起始/峰值/末尾，或分段均值）
- 结论（是否收敛、主要异常、伴随证据）
- 可能的机制解释（例如“碰撞惩罚过强导致保守停滞”“reward_clip 过多导致信号失真”等）

### 2) 回顾历史 Debug 记录（DEBUG_SUMMARY.md）并关联代码
针对发现的异常：
- 先检索 `/home/caohy/repositories/Go2HierarchicalRewardShapingRL/DEBUG_SUMMARY.md` 中是否已有同类或相关记录。
- 若有：核对当时的改动项与目标指标，判断是否对当前异常生效。
- 若无：标记为新异常模式，后续建议需明确验证策略。

同时回查相关代码与配置，确认当前实验是否包含这些改动或被覆盖，并检查是否存在逻辑偏差：
- 奖励与终止：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- 安全指标/边界：`legged_gym_go2/legged_gym/envs/go2/go2_env.py`
- 动作/观测缩放与 lidar：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 训练脚本与超参覆盖：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`

### 3) 评估“之前 debug 是否起作用”
必须回答以下问题：
- 相关改动是否已生效？（对比日志中 `reward_clip/action_std/approx_kl/...` 与配置）
- 目标指标是否改善？（例如碰撞下降、success 上升、goal_dist 明显下降）
- 若无效：分析导致无效的原因（如配置未生效、脚本覆盖、日志口径变化等）。

### 4) 下一步建议（必须给出且可执行）
- 基于异常类型给出优先级明确的调整建议（参数/代码/训练策略）。
- 每条建议需说明：目标指标 + 预期变化方向 + 验证窗口（如“重新训练 200–400 iter 后观察 success、collision、goal_dist”）。
- 若怀疑数值问题，建议加入最小化的日志/保护以定位来源（例如 NaN 检测）。

### 推荐报告结构（输出模板）
1) 日志结论：是否收敛 + 主要异常 + 证据。  
2) 历史关联：DEBUG_SUMMARY 中的相关条目 + 是否生效。  
3) 原因分析：为何异常持续/复现（机制解释）。  
4) 下一步建议：按优先级列出 2–4 条可执行动作。

## 配置入口与关键参数
- 高层环境/奖励：`legged_gym_go2/legged_gym/envs/go2/go2_config.py` → `GO2HighLevelCfg`。
- 低层环境与安全配置：`GO2RoughCfg.rewards_ext`（障碍物、目标、边界与终止距离）。
- PPO 超参数：`GO2HighLevelCfgPPO`。
- 观测维度在 `go2_config.py` 末尾自动计算，修改雷达配置后需同步。
- 低层策略路径：`GO2HighLevelCfgPPO.runner.low_level_model_path`。
- PPO 学习率上限：`GO2HighLevelCfgPPO.algorithm.max_lr` 当前为 `1e-3`。

## 常用命令
运行前激活环境：
```bash
conda activate unitree-rl
```

训练高层策略：
```bash
python legged_gym_go2/legged_gym/scripts/train_reward_shaping.py --headless=true --num_envs=32
```

绘制环境布局：
```bash
python legged_gym_go2/legged_gym/scripts/plot_env_layout.py
```

绘制训练曲线：
```bash
python legged_gym_go2/legged_gym/scripts/plot_training_results.py /home/caohy/repositories/Go2HierarchicalRewardShapingRL/logs/high_level_go2_Reward_Shaping/<timestamp>/training.log
```

Mujoco 部署示例：
```bash
python legged_gym_go2/deploy/deploy_mujoco/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml
```

## 开发注意事项
- 上层策略训练**不能改动**低层 policy 权重，仅作推理。
- 修改雷达 bin 数/范围时，需同时检查 `GO2HighLevelCfg` 与观测维度计算。
- 终止逻辑由 `reward_shaping.terminate_on_*` 控制，会影响底层 `terminate_on_reach_avoid`。
- `train_reward_shaping.py` 会在 `__main__` 覆盖设备与 headless 参数，改设备需改脚本。
