# Debug Summary
# 20160117-121702
## Analysis Conclusions
- Using target direction as vx/vy caused mixed progress; switching to "forward + turn toward target" makes progress consistently positive.
- Low-level tracking is healthy: body_speed tracks cmd_speed at ~0.9 ratio in diagnostics.
- Reward now uses real target distance (`reach_metric`) so progress/success are reliable.
- Lidar-estimated target distance (`target_distance_est`) is systematically larger than real `reach_metric` by ~1.2-1.3 m, indicating a bias in the perception-derived distance. This affects observation scale but not reward now.

## Code Changes
- Reward uses real distance for progress and success:
  - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
    - Reward and progress now use `reach_metric` instead of lidar-estimated distance.
    - Info now includes `target_distance_est` and `reach_metric` for comparison.
- Diagnostics updated for "forward + turn" control and distance comparison:
  - `legged_gym_go2/legged_gym/scripts/diagnose_command_tracking.py`
    - Command: `vx` fixed, `vy=0`, `vyaw` from `atan2(target_dir)`.
    - Logs both `goal_dist` (lidar estimate) and `reach_metric` (real).

## Next Steps
- Decide whether to correct observation distance:
  - Option A: replace obs[5] with real `reach_metric` while keeping lidar bins for perception.
  - Option B: calibrate `target_distance_est` via a linear scale/offset to match `reach_metric`.
- If training stability is still an issue after observation fix, retune reward weights (primarily `progress_scale` and `obstacle_penalty_scale`).


# 20160117-123005
## 分析结论
- 目标方向直接作为 vx/vy 会导致进度正负混合；改成“前进 + 转向到目标”后进度稳定为正。
- 低层跟踪正常，`body_speed / cmd_speed` 约 0.9。
- 目标距离估计存在系统性偏差：雷达反推的 `goal_dist` 比真实 `reach_metric` 大约 1.2~1.3 m。
- 奖励已改为使用真实 `reach_metric`，因此训练不会被距离估计偏差误导，但观测尺度仍可能影响策略理解。

## 代码变更
- 高层观测第 6 维改为真实 `reach_metric`（米），雷达 bins 保留用于感知：
  - `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 奖励/进度使用真实距离，info 中输出真实与估计距离对比：
  - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- 诊断脚本改为“前进 + 转向到目标”，并同时打印 `goal_dist` 与 `reach_metric`：
  - `legged_gym_go2/legged_gym/scripts/diagnose_command_tracking.py`
- 训练日志字段清理与口径修正：
  - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
    - `goal_dist` 与 `init_goal_dist` 采用真实 `reach_metric`。
    - `min_hazard` 使用真实危险距离。
    - 删除冗余/低参考价值指标（命令 std、ratio_max 等）。
- 文档更新：
  - `AGENTS.md` 观测说明与日志字段列表同步最新实现。

## 下一步建议
- 重新跑一次训练，确认 `goal_dist`/`init_goal_dist` 与真实距离一致，`progress` 持续为正。
- 若训练仍不稳定，优先微调 `progress_scale` 与 `obstacle_penalty_scale`。
- 若需要区分“摔倒失败”与“真实碰撞”，可新增独立 `failure_rate` 指标（当前 `collision` 已对齐危险距离）。

---

## 追加记录（按优先顺序修改）

### 分析结论
- 成功率不上升的主要原因是策略几乎不动（速度小、进度小）、碰撞率高、PPO 更新幅度很小。
- 奖励尺度与观测尺度不匹配会放大价值误差并抑制策略更新。
- 初始位置过近障碍/边界会导致早期碰撞占比过高。

### 代码变更
1) 奖励尺度调整（更重视进度、降低避障惩罚、提高速度奖励、降低动作平滑惩罚）
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `progress_scale: 15 -> 45`
   - `obstacle_penalty_scale: 1.5 -> 0.6`
   - `body_speed_scale: 0.2 -> 0.6`
   - `action_smooth_scale: 0.1 -> 0.03`
   - `yaw_rate_scale: 0.05 -> 0.02`
2) 观测尺度归一化（缩放真实距离）
   - `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
   - `GO2HighLevelCfg.reach_metric_scale = 0.2`
   - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py` 传递配置
3) 初始采样安全过滤（远离障碍/边界）
   - `legged_gym_go2/legged_gym/envs/go2/go2_env.py`
4) PPO 超参数调整（增强探索与更新幅度）
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `learning_rate: 3e-5 -> 1e-4`
   - `entropy_coef: 0.001 -> 0.003`
5) 文档同步
   - `AGENTS.md`

### 下一步建议（对比诊断指标）
- 奖励尺度调整后：观察 `avg_reward`、`progress`、`cmd_speed`、`body_speed` 上升，`collision` 下降。
- 观测缩放后：观察 `value_loss`、`value_clip_frac` 下降，`approx_kl`/`clip_frac` 恢复到合理区间。
- 采样过滤后：观察 `collision`、`boundary_collision_rate`、`obstacle_collision_rate` 下降，`ep_len_mean` 上升。
- PPO 调整后：观察 `approx_kl`、`policy_loss` 不再长期接近 0，`action_std` 有适度上升。

---

## 20260117-150000
### 分析结论
- 本轮日志显示成功率长期停留在 ~1% 且碰撞率 ~60%，策略几乎不产生有效前进：`proj≈0`、`cmd_speed≈0.16`、`body_speed≈0.04`。
- PPO 更新幅度过小（`approx_kl`、`clip_frac`、`policy_loss` 长期接近 0），策略学习几乎停滞。
- 价值误差偏大（`value_loss`、`value_clip_frac` 较高），回报尺度对价值学习不友好。
- 终止主要来自碰撞，说明避障策略未成型，奖励与终止口径不一致会进一步削弱学习信号。

### 代码变更（按优先级）
1) 奖励与终止危险距离口径统一
   - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 奖励使用真实 `min_hazard_distance`，雷达估计仅用于观测/对比日志。
   - 新增 `min_hazard_distance_est` 日志字段。
   - 增加 `command_speed_scale` 进入奖励项。
2) 强化移动激励
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `alignment_scale: 1.0 -> 2.0`
   - `body_speed_scale: 0.6 -> 1.0`
   - `command_speed_scale: 0.3`（新增）
   - `idle_penalty_scale: 0.1 -> 0.3`
3) 放大 PPO 更新幅度
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `learning_rate: 1e-4 -> 2e-4`
   - `max_lr: 8e-5 -> 3e-4`
   - `clip_param: 0.1 -> 0.2`
   - `value_clip_param: 0.1 -> 0.2`
   - `max_grad_norm: 0.6 -> 1.0`
4) 降低碰撞惩罚尺度
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `collision_penalty: 100 -> 50`

### 下一步建议（对比诊断指标）
- 统一危险口径后：观察 `collision` 下降、`obstacle` 与 `min_hazard` 更一致。
- 强化移动激励后：观察 `cmd_speed`、`body_speed`、`proj` 上升，`cmd_zero` 下降。
- PPO 调整后：观察 `approx_kl` 回到 0.01–0.03，`clip_frac` 0.05–0.15，`policy_loss` 不再接近 0。
- 降低碰撞惩罚后：观察 `value_loss`、`value_clip_frac` 下降，`Vmean/Rmean` 绝对值减小。

---

## 20260117-153500
### 分析结论
- 新日志显示 `cmd_speed` 上升但 `body_speed` 仍约 0.04，`speed_ratio` 下降，说明“指令变大但实际不动”，策略仍难有效前进。
- `proj≈0`、`progress≈0.005` 与 `goal_dist` 基本不降，成功率仍停留在 ~1%。
- PPO 更新仍偏小：`approx_kl≈0.0026`、`clip_frac≈0.02`，策略更新不足。
- 奖励惩罚主导，`avg_reward` 仍为负。

### 代码变更（按优先级）
1) 用“指令方向对齐”替代“指令速度奖励”
   - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 新增 `command_alignment`，奖励项使用 `command_alignment_scale`。
   - 日志新增 `cmd_align` 以验证指令是否朝目标。
2) 增加动作保持与平滑
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `high_level_action_repeat: 5 -> 10`
   - `action_smooth_scale: 0.03 -> 0.08`
3) 提升 PPO 更新幅度
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `learning_rate: 2e-4 -> 3e-4`
   - `max_lr: 3e-4 -> 5e-4`
   - `desired_kl: 0.02 -> 0.01`
4) 扩大高层动作范围（不改映射）
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `action_scale: [1.0, 1.0, 1.0] -> [1.3, 1.0, 1.0]`
5) 日志增加 `cmd_align`
   - `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`

### 下一步建议（对比诊断指标）
- 观察 `cmd_align` 与 `proj`：若 `cmd_align` 升而 `proj` 仍低，说明低层跟踪或动作保持仍不足。
- 观察 `body_speed` 与 `speed_ratio`：若依旧低，优先检查低层速度跟踪与高层动作频率。
- 观察 `approx_kl`、`clip_frac`：应接近目标区间（`approx_kl≈0.01`）。

---

## 20260117-162500
### 分析结论
- 新日志显示超时占比高（约 70%），成功率仍在 0.3%~0.7% 区间。
- `cmd_align` 仍为 0（记录异常），导致无法验证“指令方向奖励”是否生效。
- `cmd_speed` 上升但 `body_speed` 仍约 0.036，实际前进几乎没有变化。
- 自适应 PPO 出现学习率下降（`lr` 到 5e-6），更新可能被抑制。

### 代码变更（按优先级）
1) 修复 `cmd_align` 记录
   - `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 将 `command_alignment` 写入 `infos`，日志可直接观测指令方向是否对齐目标。
2) 延长 episode 时长以降低超时
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `episode_length_s: 20 -> 40`
3) 增加超时/怠速惩罚
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `idle_speed_threshold: 0.05 -> 0.1`
   - `idle_penalty_scale: 0.3 -> 0.6`
   - `timeout_penalty: 0 -> 10`
4) PPO 自适应保持但放宽 KL 目标
   - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - `desired_kl: 0.01 -> 0.03`

### 下一步建议（对比诊断指标）
- 观察 `cmd_align` 是否明显上升；若仍为 0，需排查指令与目标方向计算/日志写入链路。
- 观察 `timeout` 是否明显下降；若仍高，考虑进一步延长 `episode_length_s` 或降低 `high_level_action_repeat`。
- 观察 `body_speed` 与 `speed_ratio` 是否上升；若仍低，需检查低层速度跟踪或高层动作幅度。
- 观察 `lr` 是否稳定在合理区间；若仍频繁掉到 `min_lr`，考虑提高 `desired_kl` 或减小更新步长。
