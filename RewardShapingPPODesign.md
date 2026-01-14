# Reward-Shaping PPO 算法说明（高层导航）

## 摘要
MCRA_RL 采用分层强化学习：低层为固定的行走控制器，高层为可训练的导航策略。高层策略输出速度指令，由 reward-shaping PPO 进行优化，奖励由目标进度、目标方向投影、角度误差和避障惩罚组成，并叠加到达/碰撞终止奖励。

## 系统结构与交互
- **低层控制**：预训练策略接收速度命令并输出关节动作（`legged_gym_go2/legged_gym/envs/go2/go2_env.py`）。
- **高层控制**：可训练策略输出 \([v_x, v_y, v_{yaw}]\)（`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`）。
- **层间交互**：
  - 高层动作先裁剪到 \([-1, 1]\)，按 `action_scale` 缩放；
  - 再映射为底层命令：\(v_x \times 0.6\), \(v_y \times 0.2\), \(v_{yaw} \times 0.8\)；
  - 每个高层动作执行 `high_level_action_repeat` 次低层步进（`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`）。

## 问题设定
高层导航建模为 MDP \((s_t, a_t)\)：
- 状态 \(s_t\)：包含朝向、机体速度、目标方向/距离、以及手动 lidar bins。
- 动作 \(a_t\)：高层速度命令（单位无量纲，经过裁剪与缩放）。
- 目标：在安全约束下尽快接近目标并到达目标区域。

## 安全与距离度量
环境在每步计算：
- **目标距离**：\(d_t = \|p_{robot}^{xy} - p_{target}^{xy}\|\)
- **最近危险距离**：`min_hazard_distance`（障碍物表面或边界的最近距离）
对应实现：`legged_gym_go2/legged_gym/envs/go2/go2_env.py`

## Reward-Shaping 奖励定义
设执行的速度命令为 \(v_{cmd}\)（来自 `base_env.commands`），目标方向在机体坐标系下为单位向量 \(\hat{u}_{target}\)。

**(1) 目标方向投影与角度误差**
\[
R_{proj} = w_f \cdot (v_{cmd}^{xy} \cdot \hat{u}_{target})
\]
设 \( \hat{u}_{cmd} = \frac{v_{cmd}^{xy}}{\|v_{cmd}^{xy}\|} \)，角度误差
\[
\theta = \arccos(\text{clip}(\hat{u}_{cmd} \cdot \hat{u}_{target}, -1, 1))
\]
角度惩罚仅在 \(\|v_{cmd}^{xy}\| > 1e^{-3}\) 时启用：
\[
R_{ang} = - w_\theta \cdot \theta
\]

**(2) 避障惩罚**
设 \(d_{haz}\) 为最近危险距离，避障距离为 \(d_{avoid}\)：
\[
r_3(d_{haz}) = \max\Big(1 - \frac{\max(d_{haz}, 0)}{d_{avoid}}, 0\Big)
\]
障碍惩罚项：
\[
R_{obs} = - w_o \cdot r_3(d_{haz})
\]

**(3) 目标进度奖励**
设 \(d_{t-1}\) 为上一步目标距离：
\[
R_{prog} = w_p \cdot (d_{t-1} - d_t)
\]

**(4) 终止奖励**
到达/碰撞/超时：
\[
R_{term} =
\begin{cases}
 +r_{success}, & d_t \le d_{goal} \\
 -r_{collision}, & d_{haz} < d_{coll} \\
 -r_{timeout}, & time\_out
\end{cases}
\]

**(5) 合成奖励与裁剪**
\[
R_t = R_{proj} + R_{ang} + R_{obs} + R_{prog} + R_{term}
\]
最终奖励会乘以 `reward_scale` 并根据 `reward_clip` 进行裁剪。

对应实现：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`

## 终止逻辑与 time_outs
- **到达**：\(d_t \le d_{goal}\)
- **碰撞**：\(d_{haz} < d_{coll}\)
- **终止**：\(terminated = reached \lor collision\)
- **截断**：\(truncated = time\_out \land \lnot terminated\)
PPO 仅在 `truncated` 时进行 bootstrap（通过 `time_outs` 字段）。

## PPO 目标与损失
采用标准 PPO：
\[
L(\theta) = -L^{CLIP}(\theta) + c_v L^V(\theta) - c_e \mathbb{E}_t[H(\pi_\theta)]
\]

**策略损失**
\[
L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\Big]
\]
其中 \(r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\)。

**价值损失（clipped）**
\[
L^V(\theta) = \frac{1}{2}\mathbb{E}_t\Big[\max\big((V_\theta-R_t)^2,\ (V_\theta^{clip}-R_t)^2\big)\Big]
\]

**优势估计（GAE）**
\[
\delta_t = r_t + \gamma (1-done_t) V_{t+1} - V_t,\quad
A_t = \delta_t + \gamma \lambda (1-done_t) A_{t+1}
\]

## 动作分布与数值稳定性
- 策略分布为 `Normal(mean, std)`，动作通过 `tanh` 压缩到 \([-1, 1]\)。
- 训练时对 log-prob 做变换修正（change-of-variables），保持 PPO 一致性。
- 为避免溢出，训练中对 `logp_diff` 做 \([-20, 20]\) 裁剪，并在 `approx_kl` 超出阈值时提前停止更新。

## 训练流程
1. Reset 环境，获得观测与初始距离。
2. Rollout `num_steps_per_env` 步：
   - 采样动作（tanh-squashed）
   - 高层动作映射到底层速度命令并执行
   - 计算奖励并记录终止、超时
3. 计算 GAE 回报与优势
4. PPO 更新（含 KL 早停）
5. 记录日志与定期保存模型

## 关键实现位置
- 高层环境与速度映射：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 层级封装与动作重复：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- 奖励整形训练脚本：`legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
- PPO 算法实现：`rsl_rl/rsl_rl/algorithms/ppo.py`
- PPO 模型定义：`rsl_rl/rsl_rl/modules/actor_critic.py`
