# Debug Summary (Reward Shaping PPO Success Rate)

## 20260116-121100 现象与日志结论
- success 维持在 ~1%，collision 仍高（约 0.68）；avg_reward 改善但 progress 很小。
- cmd_speed 上升、body_speed 维持 ~0.042，speed_ratio 下降；目标距离与 min_hazard 基本不变。
- 新增指令统计显示：cmd_zero 降、cmd_delta 仍高、cmd_std_* 上升，指令更“激进”但未转化为到达率。
- PPO 更新仍有极端尖峰（ratio_max、logp_diff_max、grad_norm）。
- 初步判断：指令抖动/策略不稳定导致“乱冲 + 碰撞/重置”，而非低层跟踪问题。

### 修改代码，验证“指令抖动”主因
- 调整 `high_level_action_repeat`：5 → 10，用于测试是否能降低指令抖动。

## 20260116-131156 现象与日志结论
- cmd_delta 未下降，指令分布仍更分散；抖动并未缓解。
- success 进一步下降（~0.34%），collision 下降但成功率未提升，策略趋于保守。
- 结论：抖动不是由动作频率导致，核心仍在高层策略/奖励结构。

### 修改代码与后续调整方向
- 恢复 `high_level_action_repeat` 至 5。
- 继续微调奖励/超参以增强真实进度、降低 PPO 极端更新：
  - `goal_progress_scale` 20.0 → 25.0
  - `clip_param` 0.08 → 0.06
  - `desired_kl` 0.02 → 0.015
  - `max_lr` 1e-4 → 5e-5
  - `max_grad_norm` 1.0 → 0.8
- 解释并新增 `speed_ratio_active`：仅统计 `cmd_speed > 0.1`，用于诊断“有指令时的跟踪效率”，不直接提升成功率。

## 20260116-151036 现象与日志结论
- 训练日志中成功率长期在 1% 左右，且碰撞率约 0.65，超时接近 0。
- `progress` 接近 0、`goal_dist` 基本不变，说明对目标推进不足。
- `cmd_speed` 与 `body_speed` 存在明显差距，但用户已通过 `check_low_level_tracking.py` 确认底层跟踪能力正常。

### 排查与分析内容
- 统计口径问题：
  - `collision` 按 episode 结束统计；
  - `boundary_collision`/`obstacle_collision` 按 step 统计，数值会比 episode 口径小很多。
  - 因此出现 `collision` 高、`boundary_collision` 近 0 的“统计差异”现象。
- 观测构造检查：
  - 目标方向与激光雷达观测未发现明显常量/错误归一化导致的退化。
  - 重点问题转移到控制与奖励/难度配置。
- 控制通道问题：
  - 底层 `heading_command=True` 时，`commands[2]` 会由 `commands[3]` 计算；
  - 高层仅设置 `commands[:3]`，导致 `commands[3]` 随机重采样，造成 yaw 指令被随机覆盖。
  - 结论：高层 yaw 控制通道存在失真。
- 奖励/难度问题：
  - 正向推进奖励偏弱（`command_proj_weight=0`，几乎只靠实际速度投影）。
  - 碰撞惩罚偏大、终止严格，探索被强惩罚限制。
  - 障碍密度高、初始位置随机，任务难度较大。

### 已做代码修改
1) 同步高层 yaw 目标至 `commands[3]`，修复 heading 模式下的 yaw 失真  
   - 文件：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`  
   - 修改：根据当前朝向和 `vyaw` 计算目标航向，写入 `commands[3]`

2) 奖励权重微调（增强推进信号、降低惩罚强度）  
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`  
   - 修改：
     - `forward_reward_scale` 1.2 → 1.5  
     - `command_proj_weight` 0.0 → 0.3  
     - `obstacle_penalty_scale` 0.4 → 0.3  
     - `goal_progress_scale` 25.0 → 30.0  
     - `collision_penalty` 150.0 → 120.0

3) 记录同口径碰撞统计（episode 口径）  
   - 文件：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`  
   - 新增日志字段：`boundary_collision_rate` / `obstacle_collision_rate`
   - 同步更新说明：`AGENTS.md`


### 下一步建议
1) 先短训 50–100 iter 验证修复效果，观察 `proj / progress / success / collision` 是否改善。  
2) 若仍偏低，可进一步：
   - 提升 `command_proj_weight` 至 0.4；
   - 或再提升 `goal_progress_scale`（例如 35）。
3) 若仍难提升，考虑引入 curriculum（缩小初始采样范围或减少障碍数量）。

## 20260116-165512 现象与日志结论
- success 仍维持在 ~1%（前10轮 0.0117，后10轮 0.0135），collision 约 0.66，timeout 为 0。
- avg_reward 明显改善（-0.3085 → -0.0573），但 progress 仍接近 0，goal_dist 基本不变（~4.35）。
- proj 上升、angle 下降（1.47 → 0.43），说明“指令方向对齐”变好，但真实接近目标没有提升。
- cmd_speed 大幅上升（0.153 → 0.383），body_speed 几乎不变（~0.042），speed_ratio 与 speed_ratio_active 显著下降（0.47 → 0.13）。
- 指令分布更激进：cmd_zero 近 0，cmd_std_vx/vy/vyaw 与 cmd_speed_std 上升，cmd_delta 仍偏大。
- PPO 更新过激仍存在：approx_kl 从 ~0.001 升到 ~0.12，ratio_max、logp_diff_max 很高，entropy 略降。
- 结论：奖励主要驱动了“对齐与指令强度”，但没有驱动真实推进；PPO 更新偏激进影响收敛。

### 修改代码与方向调整
- 让方向误差与真实速度对齐，并引入“速度不匹配惩罚”：
  - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
    - `angle_error` 改为基于 `body_dir`，并用 `body_speed` 做门控。
    - 新增 `speed_mismatch_scale`（仅在 `cmd_speed > 0.1` 时惩罚 `command_speed - body_speed`）。
    - 新增日志组件 `speed_mismatch`。
- 奖励与 PPO 超参调整：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `command_proj_weight` 0.3 → 0.0
    - `goal_progress_scale` 30.0 → 35.0
    - 新增 `speed_mismatch_scale = 0.2`
    - `learning_rate` 2e-5 → 1e-5
    - `clip_param` 0.06 → 0.05
    - `desired_kl` 0.015 → 0.01
    - `min_lr` 5e-6 → 2e-6
    - `max_lr` 5e-5 → 2e-5
    - `num_learning_epochs` 3 → 2
    - `num_mini_batches` 4 → 2
    - `max_grad_norm` 0.8 → 0.6

### 下一步建议
1) 用当前修改重新训练 50–100 iter，观察 `progress/goal_dist/body_speed/speed_ratio_active` 是否抬升、`approx_kl` 是否显著下降。  
2) 若 `speed_ratio_active` 仍很低，可将 `speed_mismatch_scale` 逐步提高到 0.3–0.4。  
3) 若 KL 仍偏高，考虑进一步降低更新强度（如 `num_learning_epochs=1` 或引入 KL 早停）。  

## 20260116-184939 现象与日志结论
- success 仍在 ~1%，collision ~0.66，timeout 为 0。
- avg_reward 未改善（-0.326 → -0.328），proj 仍接近 0，angle 维持 ~1.56，说明“朝向对齐”未学到。
- progress/goal_dist 基本不变，真实接近目标没有提升。
- cmd_speed 维持 ~0.15，cmd_zero ~0.34，指令整体偏小且大量为零。
- speed_ratio_active ~0.243，反映并非低层跟踪问题，而是高层指令本身偏弱。
- PPO 更新过于保守：approx_kl ~1e-4、ratio_max ~1.2、clip_frac ≈ 0，策略几乎不更新。
- 结论：策略“学不动”，落在低指令/低推进的局部最优。

### 修改代码与方向调整
- 恢复 PPO 更新强度、抑制过度保守更新：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `learning_rate` 1e-5 → 2e-5
    - `clip_param` 0.05 → 0.08
    - `desired_kl` 0.01 → 0.02
    - `min_lr` 2e-6 → 5e-6
    - `max_lr` 2e-5 → 5e-5
    - `num_learning_epochs` 2 → 3
    - `num_mini_batches` 2 → 4
- 降低“速度不匹配惩罚”以避免策略靠小指令逃惩罚：
  - `speed_mismatch_scale` 0.2 → 0.1
- 恢复部分指令引导：
  - `command_proj_weight` 0.0 → 0.15
- 角度惩罚门控更严格，避免低速噪声惩罚：
  - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
    - `angle_mask`：`body_speed > 1e-3` → `body_speed > 0.1`

### 下一步建议
1) 重新训练 50–100 iter，观察 `approx_kl` 是否回升到 0.005–0.02、`proj/progress` 是否同步抬升。  
2) 若仍“学不动”，可进一步上调 `learning_rate` 或 `clip_param`（小幅）。  
3) 若 `cmd_speed` 仍过小，可将 `command_proj_weight` 提至 0.2。  

## 20260116-194430 现象与日志结论
- success 仍在 ~1%（前10轮 0.0117，后10轮 0.0137），collision ~0.66，timeout 为 0。
- proj 仍接近 0，angle 约 1.56（接近 90°），说明“朝目标对齐”仍未学到。
- progress/goal_dist 基本不变，真实接近目标没有提升。
- cmd_speed 略升（0.151 → 0.165），但 cmd_zero 仍高（~0.30）；指令小且抖动不减。
- speed_ratio_active 仍在 ~0.23，说明并非低层跟踪问题，而是高层指令偏弱。
- PPO 更新仍偏保守：approx_kl 很低、clip_frac 极小，策略更新不足。
- 结论：策略仍停留在低指令/低推进的局部最优，学习信号不足。

### 修改代码与方向调整
- 提高 PPO 更新强度 + 强化方向指令奖励：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `command_proj_weight` 0.15 → 0.25
    - `learning_rate` 2e-5 → 3e-5
    - `clip_param` 0.08 → 0.1
    - `desired_kl` 0.02 → 0.03
    - `max_lr` 5e-5 → 8e-5

### 下一步建议
1) 训练 50–100 iter，观察 `approx_kl` 是否回到 0.005–0.02，`proj` 是否显著上升。  
2) 若 `proj` 仍接近 0，可进一步提高 `command_proj_weight`（0.3）。  
3) 若更新仍过于保守，可继续上调 `learning_rate` 或 `clip_param`（小幅）。  

## 20260116-204129 现象与日志结论
- success 仍在 ~1%，collision ~0.66，timeout 为 0。
- proj 上升（0.002 → 0.035），但 angle 仍接近 90°；progress/goal_dist 仍基本不动。
- cmd_speed 上升（0.152 → 0.218），cmd_zero 明显下降，但 body_speed 仍 ~0.042。
- speed_ratio_active 下降（~0.24 → ~0.20），指令更激进但推进效率更低。
- PPO 更新更强（approx_kl 上升、ratio_max/logp_diff_max 上升），但仍未转化为成功率。
- 结论：指令更大但真实推进无提升，成功率仍不上升。

### 修改代码与方向调整
- 强化真实推进奖励、抑制“只下指令”：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `goal_progress_scale` 35.0 → 45.0
    - 新增 `body_speed_scale = 0.1`
  - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
    - 增加 `body_speed` 正奖励项（`body_speed_scale * body_speed`）

### 下一步建议
1) 训练 50–100 iter，观察 `body_speed` 是否上升，`progress/goal_dist` 是否改善。  
2) 若仍无提升，可继续提高 `body_speed_scale`（小幅递增）。  
3) 若碰撞明显增加，可适度回调 `goal_progress_scale`。  

## 20260116-214034 现象与日志结论
- success 仍在 ~1%，collision ~0.66，timeout 为 0。
- proj 上升（0.002 → 0.027），但 angle 仍接近 90°；progress/goal_dist 仍基本不动。
- cmd_speed 上升（0.152 → 0.198），cmd_zero 明显下降，但 body_speed 仍 ~0.042。
- speed_ratio_active 下降（~0.24 → ~0.21），指令更激进但推进效率更低。
- PPO 更新更强（approx_kl 上升、ratio_max/logp_diff_max 上升），但未转化为成功率。
- 结论：指令更大但真实推进无提升，成功率仍不上升。

### 修改代码与方向调整
- 强化真实推进奖励并加入轻微抖动惩罚：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
    - `goal_progress_scale` 45.0 → 60.0
    - `body_speed_scale` 0.1 → 0.2
    - 新增 `cmd_delta_scale = 0.05`
  - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
    - `_compute_reward_shaping` 增加 `prev_command_xy` 输入
    - 新增 `cmd_delta` 惩罚项（`cmd_delta_scale * ||cmd_t - cmd_{t-1}||`）
    - 组件记录新增 `cmd_delta`

### 下一步建议
1) 训练 50–100 iter，观察 `body_speed` 是否上升、`progress/goal_dist` 是否改善。  
2) 若 `cmd_delta` 未下降，可继续上调 `cmd_delta_scale`（小幅）。  
3) 若推进仍弱，可进一步提高 `body_speed_scale` 或 `goal_progress_scale`。  

## 20260116-223756 现象与日志结论
- success/reach 均值约 1.1%（最大 3.3%），collision ~0.66，timeout 为 0；成功率仍无上升。
- avg_reward 改善（-0.195 → -0.076）、proj 上升（0.002 → 0.097），但 angle ~1.56、progress 近 0、goal_dist 基本不变，真实接近目标没有提升。
- cmd_speed 大幅上升（0.15 → 0.44）、cmd_zero 降至 0、cmd_std_* 与 action_sat 上升，指令更激进。
- body_speed 始终 ~0.042、speed_ratio_active 下降（0.24 → 0.11），cmd_speed 与 body_speed 相关性极低，指令难以转化为真实速度。
- PPO 更新不稳定：approx_kl/ratio_max/logp_diff_max/grad_norm 多次尖峰，Vstd/Rstd 膨胀。
- 结论：策略主要在“放大指令/奖励投影”上优化，真实推进不足；命令-运动脱钩 + 更新不稳定导致成功率不上升。

### 下一步建议
1) 记录 per-step `cmd_speed` vs `body_speed`（含执行通道），确认命令-运动脱钩的具体链路是否存在控制/裁剪问题。  
2) 奖励进一步绑定真实推进（如 `body_speed` 投影或 `goal_progress` 权重），并对过激指令/饱和动作加入抑制项以避免 reward hacking。  
3) 引入 KL 早停或更严格的梯度裁剪，降低 `ratio_max/logp_diff_max` 尖峰，提升更新稳定性。  
