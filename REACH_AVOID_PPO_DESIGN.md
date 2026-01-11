# Reach-Avoid PPO 强化学习设计说明

## 摘要
MCRA_RL 将导航任务表述为“到达目标 + 始终安全”的双目标优化。训练过程中不直接使用传统奖励，而是由目标函数 g 与安全约束 h 构造 Reach-Avoid 值函数，并以此驱动 PPO 的策略与价值更新。

## 问题设定
- 环境为马尔可夫决策过程 \((s_t, a_t)\)，每个时间步返回两类信号：
  - 目标代价 \(g(s_t)\)：进入目标区域时为负值，否则与距离正相关。
  - 安全约束 \(h(s_t)\)：安全区域为负值，违规时为非负。
- 终止条件融合安全硬约束：
  \[
  d_t = d_t^{env} \lor [h(s_t) \ge 0]
  \]

## Reach-Avoid 值函数定义
先构造“安全优先”的候选值：
\[
Q_t^{(k)}=\max\Big(h_t,\ \min\big(g_t,\ \gamma V_{t+1}^{(k)}\big)\Big)
\]
其中 \(V_{t+1}^{(k)}\) 来自实现中的 `value_table`（包含 \(V_{t+1}\) 以及更深回溯的 Reach-Avoid 目标），因此每一步会得到一组不同回溯深度的 \(Q_t^{(k)}\)。

最终的价值目标（critic target）采用 GAE 风格加权平均：
\[
\hat{Q}_t=\sum_{k=0}^{K} w_{t,k}\,Q_t^{(k)},\qquad \sum_k w_{t,k}=1
\]
优势定义为：
\[
A_t=\hat{Q}_t-V_t
\]

## Reach-Avoid GAE 与回报目标
GAE 权重 \(w_{t,k}\) 由系数递推得到，并在终止状态时重置或清零。实现要点：
1. 系数向右滚动一格（时间推进）。
2. 若上一时刻未终止，用 \(\lambda\) 衰减。
3. 若上一时刻终止，用 \(\lambda/(1-\lambda)\) 重新初始化。
4. 当前时刻终止直接清零。
5. 归一化使 \(\sum_k w_{t,k}=1\)。

对应实现位置：`rsl_rl/rsl_rl/algorithms/reach_avoid_ppo.py`.

## Reach-Avoid PPO 总损失
整体优化以“最小化损失”的形式实现：
\[
L(\theta) = -L^{CLIP}(\theta) + c_v L^V(\theta) - c_e \mathbb{E}_t[H(\pi_\theta)]
\]

## Reach-Avoid PPO 策略损失
保留 PPO 的 clipped surrogate 形式，只是优势来自 Reach-Avoid 目标：
\[
L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\Big]
\]
其中 \(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\)。

## 价值损失（clipped value loss）
\[
L^V(\theta) = \frac{1}{2}\,\mathbb{E}_t\Big[\max\big((V_\theta-\hat{Q}_t)^2,\ (V_\theta^{clip}-\hat{Q}_t)^2\big)\Big]
\]

## 熵正则项
熵正则用于鼓励探索，防止策略过早变得确定性：
\[
- c_e \mathbb{E}_t[H(\pi_\theta)]
\]
其中 \(c_e\) 控制探索强度，\(H(\pi_\theta)\) 为策略分布熵。

## 实现要点与对应代码
- g/h 的具体定义：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
  - 目标区域内 \(g < 0\)，安全区域内 \(h < 0\)。
  - 默认参数：`g_target_value=-300`，`g_distance_scale=100`，`h_safe_value=-300`，`h_unsafe_value=300`。
- 终止序列：
  - `done_seq = env_dones OR (h_values >= 0)`。
  - rollout 末尾强制截断：`rollout_dones[-1]=True`。
- PPO 更新：
  - 优势标准化后使用（稳定训练）。
  - 策略损失以“最小化”实现，内部会使用 `gae_batch = -advantage`。
