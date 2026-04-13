# 2026-01-21 训练日志分析与改动记录（20260120-191554）

## 一、日志结论（是否收敛/异常）
- 训练已进入平台期：最近 200 次迭代成功率斜率很小，success 均值约 0.32，波动很小，说明已基本收敛但停在中等水平。
- 失败结构：collision ~0.20、timeout ~0.48，主要失败来自超时，碰撞主要来自障碍物而非边界。
- 进展不足：progress 均值约 0.005/step，说明向目标推进较慢。
- 学习率过早降到最小：lr 在约第 81 次迭代降到 min_lr，之后几乎不变，可能导致后期学习停滞。
- PPO 关键诊断缺失：approx_kl 和 clip_frac 全程为 nan，训练过程缺少更新强度监控。
- value_loss 后期明显上升，需关注价值函数拟合质量。

## 二、本次代码变更（为修复诊断与学习率问题）
1) 记录 PPO 的 approx_kl 与 clip_frac（用于日志输出）
   - 文件：`rsl_rl/rsl_rl/algorithms/ppo.py`
   - 修改：在 update 中计算并累计 approx_kl/clip_frac，保存到 `self.last_approx_kl`、`self.last_clip_frac`

2) 训练脚本读取 PPO 最新的 approx_kl/clip_frac
   - 文件：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
   - 修改：当 update() 仅返回 (value_loss, policy_loss) 时，从 `alg.last_approx_kl/last_clip_frac` 取值

3) 提高最小学习率，避免过早降到极小值
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`GO2HighLevelCfgPPO.algorithm.min_lr` 从 `5e-6` 提升到 `5e-5`

## 三、下一步建议（提升成功率）
1) 先验证诊断与学习率：重新训练后确认 approx_kl 与 clip_frac 不再是 nan，观察 lr 是否仍迅速降到最小值。
2) 若仍早衰，考虑将 PPO 的 schedule 改为 fixed，或提高 min_lr/learning_rate。
3) 增强“朝目标前进”的稠密奖励占比（如提高 progress_scale 或加入朝向目标速度奖励），避免仅依赖终止奖励。
4) 减少超时占比：可适当延长 episode_length_s 或提高高层控制频率（降低 high_level_action_repeat）。
5) 若碰撞仍高：提高 lidar 分辨率/范围，或略降低 safe_scale 使策略更敢向目标推进。

# 2026-01-21 训练日志分析与改动记录（20260121-130230）

## 一、日志结论（是否收敛/异常）
- 训练稳定但处于平台期：success ~0.32，collision ~0.20，timeout ~0.48，整体性能提升有限。
- PPO 诊断已恢复：approx_kl 与 clip_frac 不再是 nan。
- 但更新过猛：approx_kl 均值约 0.26、clip_frac 均值约 0.41，远高于期望（desired_kl=0.03），说明策略更新强度过大。
- 学习率仍在约第 80 次迭代降到 min_lr 并长期停留，和之前结论一致。
- goal_dist 均值仍上升，progress 偏小，说明对目标推进效率不足。

## 二、本次代码变更（降低 PPO 更新强度）
1) 降低 clip 强度
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`clip_param` 从 `0.2` 调整为 `0.1`

2) 降低学习率并改为固定调度
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`learning_rate` 从 `3e-4` 调整为 `1e-4`，`schedule` 从 `adaptive` 改为 `fixed`

3) 减少每轮 PPO 更新次数
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`num_learning_epochs` 从 `3` 降为 `2`，`num_mini_batches` 从 `4` 降为 `2`

## 三、下一步建议（按顺序）
1) 重新训练验证 KL/clip 是否回落到合理区间（approx_kl 接近 0.01~0.05，clip_frac < 0.2）。
2) 若仍偏大，继续下调 learning_rate 或 clip_param；必要时考虑减小 batch 更新频次。
3) 若 KL/clip 恢复正常但 success 仍停滞，再调整奖励结构（提高 progress_scale 或加入朝向目标速度奖励）。

# 2026-01-22 训练日志分析与改动记录（20260121-221510）

## 一、日志结论（是否收敛/异常）
- 成功率已进入平台期：最近 100 次迭代 success 均值约 0.281、std≈0.007，线性斜率接近 0（约 -1e-5/iter），可判定已收敛在 ~0.28。
- 失败结构仍以超时为主：timeout ≈ 0.495，collision ≈ 0.224，说明主要问题是“到不了目标”而非频繁碰撞。
- 进展过慢：progress ≈ 0.005/step，goal_dist 均值约 4.31m，表明推进效率不足。
- PPO 更新强度偏大：approx_kl ≈ 0.107、clip_frac ≈ 0.426，显著高于期望（desired_kl=0.03），更新仍过猛。
- value_loss 偏高（≈ 278）且较早期上升明显，价值函数拟合质量不佳。

## 二、本次代码变更（仅调整奖励与 PPO 强度）
1) 强化“推进”与“时间成本”信号，降低终止奖惩极端性
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `progress_scale` 5.0 → 8.0
     - `goal_reward` 120.0 → 80.0
     - `collision_penalty` 120.0 → 80.0
     - `timeout_penalty` 0.0 → 20.0

2) 降低 PPO 更新强度（步长与更新次数）
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `learning_rate` 1e-4 → 5e-5
     - `clip_param` 0.1 → 0.07
     - `num_learning_epochs` 2 → 1

## 三、下一步建议（按顺序）
1) 重新训练验证指标是否改善：success 是否上升，timeout 是否下降，approx_kl 回落到 0.03~0.06、clip_frac < 0.2。
2) 若 KL/clip 仍偏高，继续下调 `learning_rate` 或 `clip_param`；必要时把 `num_mini_batches` 降到 1。
3) 若 success 仍停滞但 KL/clip 正常，再追加“每步时间惩罚”（如 -0.01/step）或加入朝目标速度奖励。

# 2026-01-22 训练日志分析与改动记录（20260122-095334）

## 一、日志结论（是否收敛/异常）
- 成功率已收敛在低水平：最近 100 次迭代 success 均值≈0.167、std≈0.012，线性斜率≈ -2.4e-06/iter，平台期明显。
- 失败结构变为“碰撞为主”：collision≈0.462、timeout≈0.372，且障碍物碰撞≈0.361、边界碰撞≈0.101。
- 进展仍偏慢：progress≈0.0078/step，goal_dist≈3.96m，仅较早期略有改善。
- 奖励与成功对齐不足：avg_reward≈-0.41（较早期更负），reward_clip≈0。
- PPO 更新强度已回归正常：approx_kl≈0.020、clip_frac≈0.204，但 value_loss≈211 持续偏高。

## 二、本次代码变更（提升安全与推进效率）
1) 提高高层控制频率，并保持低层总时长一致
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `GO2HighLevelCfg.env.high_level_action_repeat` 10 → 5
     - `GO2HighLevelCfgPPO.algorithm.num_steps_per_env` 200 → 400

2) 重新平衡奖励，强化安全与目标推进
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `safe_distance` 1.0 → 1.5
     - `safe_scale` 2.0 → 4.0
     - `collision_penalty` 80.0 → 120.0
     - `progress_scale` 8.0 → 4.0
     - `timeout_penalty` 20.0 → 10.0
     - 新增 `target_speed_scale = 0.2`

3) 适度加大 PPO 更新强度
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`learning_rate` 5e-5 → 1e-4

4) 新增“朝目标速度”奖励项
   - 文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 修改：在 `_compute_reward` 中加入 `dot(base_lin_vel_xy, target_dir_body)`，按 `target_speed_scale` 加到总奖励里。

## 三、下一步建议（按顺序）
1) 重新训练后重点观察：success 是否提升、collision 是否下降、timeout 是否继续下降。
2) 若碰撞仍偏高，继续上调 `safe_scale` 或 `safe_distance`；若推进不足，适当上调 `target_speed_scale` 或小幅回调 `progress_scale`。
3) 若 success 仍停滞但 KL/clip 正常，可再微调动作频率（action_repeat）或增大感知分辨率（lidar bins）。

# 2026-01-22 训练日志分析与改动记录（20260122-165028）

## 一、日志结论（是否收敛/异常）
- 成功率未收敛且出现回落：最近 100 次 success 均值≈0.229、std≈0.057，线性斜率≈-3.8e-4/iter；最近 50 次 success≈0.216。
- 失败结构转为“高碰撞”：collision≈0.620、timeout≈0.152；最近 50 次 collision≈0.738、timeout≈0.047，说明策略更倾向高速碰撞终止。
- PPO 更新强度过大：approx_kl≈0.162、clip_frac≈0.384（最近 50 次约 0.293/0.545），显著高于期望（desired_kl=0.03）。
- 奖励分布恶化：avg_reward≈-1.28（最近 50 次≈-1.75），safety≈-0.153，value_loss≈335，价值函数拟合质量较差。
- 推进效率仍不足：progress≈0.0048/step，goal_dist≈3.59m。

## 二、本次代码变更（降低更新强度，缓和目标速度奖励）
1) 降低目标速度奖励强度
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`target_speed_scale` 0.2 → 0.1

2) 降低学习率
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`learning_rate` 1e-4 → 3e-5

## 三、下一步建议（按顺序）
1) 重新训练验证 KL/clip 是否回落到合理区间（approx_kl≈0.02~0.05，clip_frac<0.2）。
2) 若碰撞仍高，考虑对 target_speed 奖励加入“安全门控”（仅在 hazard_distance>safe_distance 时生效），或进一步下调 `target_speed_scale`。
3) 若 success 仍停滞但 KL/clip 正常，优先提高感知分辨率（lidar bins）或微调动作上限（vx 上限/转向增益）。

# 2026-01-23 训练日志分析与改动记录（20260122-233443）

## 一、日志结论（是否收敛/异常）
- 训练未收敛且后期崩塌：success 中期提升后显著下降，最近 100 次 success≈0.046、std≈0.015，线性斜率为负（≈-5.1e-4/iter）。
- 失败结构极端碰撞化：最近 100 次 collision≈0.893、timeout≈0.061；最近 50 次 collision≈0.897。
- PPO 更新强度严重异常：最近 100 次 approx_kl≈3.00、clip_frac≈0.664，远高于期望（desired_kl=0.03），说明更新失稳。
- 奖励分布恶化：avg_reward 最近 100 次≈-2.46，value_loss≈284 持续偏高。
- 最佳表现窗口出现在 iter 425–524：success≈0.244，但此时 approx_kl≈0.101、clip_frac≈0.316 已偏大；约 iter 551 后 KL/clip 持续升高并导致性能崩塌。

## 二、本次代码变更（抑制更新强度 & 安全门控）
1) 恢复自适应学习率并修正最小学习率
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：
     - `schedule` fixed → adaptive
     - `min_lr` 5e-5 → 1e-5（保证 `min_lr <= learning_rate`）

2) 增加 KL early stop 降学习率
   - 文件：`rsl_rl/rsl_rl/algorithms/ppo.py`
   - 修改：当 `approx_kl > 1.5 * desired_kl` 时，`learning_rate = max(min_lr, learning_rate / 1.5)` 并同步更新 optimizer

3) 目标速度奖励加安全门控
   - 文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 修改：`target_speed_scale * target_speed` 乘以 `clamp((d_min - safe)/safe, 0, 1)`，仅在安全距离外鼓励速度

## 三、下一步建议（按顺序）
1) 重新训练验证 KL/clip 是否回落至合理区间（approx_kl≈0.03~0.06，clip_frac<0.2）；若仍偏高，继续下调 `learning_rate` 或降低 `desired_kl`。
2) 若碰撞仍高，继续下调 `target_speed_scale` 或收紧动作上限（vx/vyaw），确认安全门控是否有效。
3) 若 KL/clip 正常但 success 仍低，考虑提高 lidar 分辨率或强化 progress 相关奖励。

# 2026-01-24 训练日志分析与改动记录（20260123-200522）

## 一、日志结论（是否收敛/异常）
- 训练**未收敛且出现后期回落**：success 先升后降，最佳窗口（iter 583–683）success≈0.190，最近100次 success≈0.067。
- 失败结构：最近100次 collision≈0.401、timeout≈0.532，超时为主且碰撞占比上升；障碍物碰撞≈0.311、边界碰撞≈0.091。
- 进展不足：progress≈0.0037/step、goal_dist≈4.06m，目标距离基本没有下降；cost≈115 表明成功更慢。
- PPO 更新强度**偏弱但稳定**：approx_kl≈0.0085、clip_frac≈0.118、reward_clip=0；学习率长期停在 max_lr=5e-4，说明自适应已放大步长但 KL 仍偏小。
- value_loss 仍偏高（≈188），但无数值爆炸迹象。

## 二、本次代码变更（动作尺度同步 + 微调进展奖励）
1) 同步高层动作尺度（提升 vx 上限）
   - 文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 修改：在 `_update_high_level_config()` 中同步 `GO2HighLevelCfg.action_scale`
   - 预期：将 vx 上限恢复为 0.78，降低超时风险

2) 微调进展奖励权重
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`progress_scale` 4.0 → 4.5
   - 预期：增强推进信号，缓解“慢而不成”的局部最优

## 三、下一步建议（按顺序）
1) 进行小范围对照训练（200–400 iter），重点观察 success、timeout、goal_dist 是否改善，同时监控 collision 与 boundary_collision_rate 是否上升。
2) 若超时显著下降但碰撞升高，可微调 `safe_scale/safe_distance` 或轻微收紧 `vx` 上限（例如 action_scale 从 1.3 回调至 1.2）。
3) 若 success 仍停滞但 KL/clip 正常，考虑提升 lidar 分辨率（如 24/32 bins）或适度上调 `target_speed_scale`。

# 2026-01-26 训练日志分析与改动记录（20260125-104722）

## 一、日志结论（是否收敛/异常）
- 训练**未收敛且出现数值崩溃**：success 早期有小幅提升（中段约 0.16），后期急剧恶化，最后迭代直接报错退出。
- 失败结构后期极端碰撞化：最近 100 次 collision≈0.977、timeout≈0.001，几乎全是碰撞终止。
- PPO 更新强度失控：approx_kl 在约 iter 713 起持续升高（>0.5），clip_frac 持续偏高（>0.6），并在 iter 784–787 出现 policy_loss 超大值/inf。
- 数值异常触发崩溃：iter 787 policy_loss=inf，iter 788 action_std=nan，导致后续 `Normal(mean, std)` 抛错。

## 二、原因分析（机制解释）
- PPO 更新过猛导致策略参数爆炸：高 KL + 高 clip_frac 表明更新幅度过大；在无数值防护的情况下 `log_prob/ratio` 溢出，最终把 actor 的均值或 std 写成 NaN。
- 分布参数无约束：`std` 作为可学习参数未做 `softplus/exp` 或 clamp，NaN/负值可直接传入 Normal，引发崩溃。
- 缺少训练数值守护：loss/log_prob/value 未做 `isfinite` 检查，出现 inf/NaN 时仍执行反向传播。

## 三、本次代码变更（数值稳定性修复 + 断点恢复）
1) PPO 更新数值守护与溢出保护  
   - 文件：`rsl_rl/rsl_rl/algorithms/ppo.py`  
   - 修改：
     - 对 `actions_log_prob/value/mu/sigma/entropy/log_ratio/loss` 做 `isfinite` 检查，异常即跳过该 minibatch 更新。  
     - `ratio = exp(clamp(log_ratio, -20, 20))` 避免指数溢出。  
     - 当 `approx_kl > 2 * desired_kl` 时跳过该更新，避免继续发散。  
     - 仅统计有效更新数，避免全跳过导致除 0。

2) 行为分布 std 安全裁剪  
   - 文件：`rsl_rl/rsl_rl/modules/actor_critic.py`  
   - 修改：`std` 通过 `nan_to_num + clamp(1e-6, 10.0)`，避免 NaN/负数直接喂给 Normal。


## 四、下一步建议（按优先级）
1) 继续训练 200–400 iter，重点观察 `approx_kl/clip_frac/policy_loss/action_std` 是否保持有限且稳定（approx_kl 目标 0.02–0.06，clip_frac < 0.2）。  
2) 若 KL/clip 仍快速升高，继续降低更新强度：下调 `learning_rate/max_lr` 或 `clip_param`（如 0.05）。  
3) 若数值稳定但 success 仍低，按碰撞主导策略调整：适度提高 `safe_scale/safe_distance` 或收紧 `vx` 上限。  
4) 如仍出现 NaN，可在训练脚本加观测/奖励 `isfinite` 统计，进一步定位环境数值异常。

# 2026-01-27 训练日志分析与改动记录（20260126-104932）

## 一、日志结论（是否收敛/异常）
- 训练**几乎无学习**：success 全程维持在 ~0.0037，最佳 100-iter 窗口仅 0.00397（iter 752–851），无明显上升趋势。
- 失败结构以**超时为主**：collision≈0.308、timeout≈0.688；边界/障碍碰撞比例稳定，无显著变化。
- 进展停滞：progress≈0.0026/step，goal_dist≈4.34m 基本不降，说明“慢而不成”。
- PPO 更新强度**过弱**：approx_kl≈4.7e-05、clip_frac≈0.003、policy_loss≈1e-4；学习率长期停在 5e-4 但更新仍很弱。
- action_std 持续升高（0.316→0.469）但收益不变，探索未转化为有效推进。

## 二、原因分析（机制解释）
- PPO 更新幅度过小或有效更新过少，导致策略几乎不变（KL/clip 极低、policy_loss 接近 0）。
- 可能存在“更新被跳过但日志未显式记录”的盲区，需要增加最小化日志来确认是否有大量 minibatch 被跳过（非有限值或 KL early-stop）。
- `resume=False` 可能导致从零开始训练，若目标是承接旧 checkpoint，会加剧“长期低性能”现象。

## 三、本次代码变更（增强更新强度 + 记录更新跳过）
1) 增强 PPO 更新强度  
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`  
   - 修改：`num_learning_epochs` 1 → 2，`num_mini_batches` 2 → 4

2) 增加 PPO 更新/跳过统计并落日志  
   - 文件：`rsl_rl/rsl_rl/algorithms/ppo.py`  
   - 修改：统计 `last_num_minibatches/last_num_updates/last_num_skipped/last_num_skipped_kl/last_num_skipped_nonfinite`  
   - 文件：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`  
   - 修改：日志新增 `ppo_updates/ppo_skipped/ppo_skip_frac/ppo_skip_kl/ppo_skip_nonfinite`

## 四、下一步建议（按优先级）
1) 重新训练 200–400 iter，重点观察 `ppo_updates/ppo_skipped/ppo_skip_frac` 是否正常（跳过比例应低于 0.1），并验证 `approx_kl≈0.01–0.05`、`clip_frac≈0.05–0.2`。  
2) 若跳过比例高：优先定位非有限值来源（观测/奖励/优势）或 KL early-stop 触发频率；必要时降低 `learning_rate` 或 `clip_param`。  
3) 若 KL/clip 仍过低且跳过比例正常，再上调更新强度（例如 `num_learning_epochs=3` 或小幅提高 `learning_rate`）。  
4) 若需要承接旧模型，请确认 `GO2HighLevelCfgPPO.runner.resume=True` 且路径有效，避免“从零开始”导致的长期低性能。

# 2026-01-28 训练日志分析与改动记录（20260127-142720）

## 一、日志结论（是否收敛/异常）
- 训练**几乎无学习且稳定停滞**：success 全程维持在 ~0.0035 左右（末 100 次均值≈0.0035），无上升趋势，已形成低水平平台。
- 失败结构以**超时为主**：timeout≈0.688、collision≈0.308，边界/障碍碰撞比例稳定且无改善。
- 进展停滞：progress≈0.0026/step，goal_dist≈4.34m 基本不降，min_hazard≈0.97m 稳定。
- PPO 更新强度**过弱且进一步变弱**：末 100 次 approx_kl≈1.45e-4、clip_frac≈0.0052，policy_loss 近 0；与初期相比 KL/clip 进一步下降。
- action_std 持续上升（≈0.33 → 0.88），但 success/goal_dist 无改进，说明探索未转化为有效推进；smooth 均值上升（≈0.28 → 0.52）。

## 二、原因分析（机制解释）
- PPO **更新幅度过小**：KL/clip 极低且不升，策略分布几乎不变，说明学习率/更新强度对当前任务过保守。
- **优势信号有效性不足**：progress/goal_dist 长期不变，奖励差异过小导致 policy_loss 接近 0。
- 行为噪声增大但缺乏有效反馈：action_std 上升却不改善目标距离，可能是“探索不转化”或奖励主导项过弱/过平坦。

## 三、本次代码变更（增加诊断日志 + 提升更新上限）
1) 增加优势统计（adv_mean/adv_std）
   - 文件：`rsl_rl/rsl_rl/storage/rollout_storage.py`
   - 修改：记录 raw advantages 的 mean/std，并保留归一化后统计
   - 文件：`rsl_rl/rsl_rl/algorithms/ppo.py`
   - 修改：`compute_returns()` 暴露 adv_mean/adv_std 到 PPO 侧

2) 增加 ratio 分布统计（判断 ratio 是否集中在 1 附近）
   - 文件：`rsl_rl/rsl_rl/algorithms/ppo.py`
   - 修改：记录 `ratio_mean/ratio_std/ratio_abs_mean/ratio_min/ratio_max`

3) 增加 reward 分量与相关性日志
   - 文件：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
   - 修改：新增 `target_speed_reward` 组件并写入 infos
   - 文件：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
   - 修改：记录 progress/safety/smooth/target_speed 均值+方差；记录 `cmd_goal_corr`（command_speed 与 progress 的相关性）

4) 提升自适应学习率上限以拉升 KL
   - 文件：`legged_gym_go2/legged_gym/envs/go2/go2_config.py`
   - 修改：`GO2HighLevelCfgPPO.algorithm.max_lr` 5e-4 → 1e-3

## 四、下一步建议（按优先级）
1) 重新训练 200–400 iter，重点观察 `approx_kl` 是否回升到 0.01–0.05、`clip_frac` 回到 0.05–0.2；同时查看新增的 `adv_mean/adv_std` 与 `ratio_*` 是否正常。
2) 若 KL/clip 仍偏低：优先提高更新强度（如 `num_learning_epochs=3` 或 `clip_param` 回调到 0.1），窗口 200–400 iter 验证。
3) 若 KL/clip 恢复但 success 仍不升：检查 `cmd_goal_corr`、`progress_var` 是否接近 0；必要时调整奖励分布（略降 safe_scale/safe_distance 或增强 progress/target_speed）。
