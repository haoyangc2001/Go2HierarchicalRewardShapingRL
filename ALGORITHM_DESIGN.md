# Go2 Hierarchical Reward-Shaping RL 算法设计

## 1. 文档目的

本文档说明本项目当前实际采用的算法设计，而不是泛化的方案草稿。内容覆盖：

- 分层控制结构
- 高层观测与动作定义
- Reward Shaping 数学形式
- 终止条件与成功判定
- PPO 优化流程与数值稳定性设置
- 关键配置项与代码映射

所有说明均以仓库当前实现为准，主要对应：

- `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
- `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
- `rsl_rl/rsl_rl/algorithms/ppo.py`
- `rsl_rl/rsl_rl/storage/rollout_storage.py`
- `rsl_rl/rsl_rl/modules/actor_critic.py`

## 2. 任务定义

项目目标是在 Isaac Gym 中训练 Go2 的高层导航策略，使机器人在二维平面障碍环境中完成：

1. 朝目标位置持续推进
2. 避免与障碍物或边界发生碰撞
3. 输出平滑、可被低层控制器稳定跟踪的速度命令

这里的任务不是直接输出关节动作，而是输出高层速度指令。

## 3. 分层控制结构

### 3.1 高层与低层职责划分

- 高层策略负责导航决策，输出动作
  $$
  a_t = [v_x, v_y, \omega_z]
  $$
- 低层策略负责运动控制，将速度命令映射为关节级动作

因此，整个系统可视为：

$$
\text{observation}_t \xrightarrow{\pi_\theta} a_t
\xrightarrow{\text{command mapping}} c_t
\xrightarrow{\pi_{\text{low}}} u_t
\xrightarrow{\text{Isaac Gym}} s_{t+1}
$$

其中：

- $\pi_\theta$ 为待训练的高层策略
- $\pi_{\text{low}}$ 为固定的低层预训练控制器
- $c_t$ 为底层速度命令
- $u_t$ 为底层关节控制动作

### 3.2 宏步长交互

高层不是每个仿真步都决策，而是每个高层 step 固定一个命令，并让底层重复执行若干次。当前默认：

$$
K = \texttt{high\_level\_action\_repeat} = 5
$$

也就是说，高层每输出一次动作，底层连续执行 5 个 sim steps。

这样做的意义是：

- 降低高层时间尺度的抖动
- 让高层更关注导航决策而不是底层稳定性
- 让 PPO rollout 更符合“决策层”而不是“执行层”

## 4. 高层状态与动作空间

### 4.1 高层动作空间

高层策略输出三维连续动作：

$$
a_t = [a_t^{x}, a_t^{y}, a_t^{\omega}] \in [-1,1]^3
$$

策略网络输出高斯采样动作后，通过 `tanh` 压缩到 `[-1, 1]` 区间。

随后动作经过两级缩放：

第一层是配置中的动作尺度：

$$
\tilde{a}_t = a_t \odot [1.3, 1.0, 1.0]
$$

第二层是写入底层命令时的物理映射：

$$
c_t^x = 0.6 \tilde{a}_t^x,\quad
c_t^y = 0.2 \tilde{a}_t^y,\quad
c_t^\omega = 0.8 \tilde{a}_t^\omega
$$

此外，heading 命令被写为：

$$
c_t^{\text{heading}} = \psi_t + 2 c_t^\omega
$$

其中 $\psi_t$ 为当前机体朝向角。

### 4.2 高层观测空间

当前默认高层观测总维度为 40，由三部分组成：

#### 基础特征 8 维

$$
o_t^{\text{base}} =
[\cos\psi_t,\ \sin\psi_t,\ v_{x,t}^{body},\ v_{y,t}^{body},\ \dot{\psi}_t,\ \hat{d}_t,\ \hat{e}_{x,t},\ \hat{e}_{y,t}]
$$

其中：

- $\psi_t$ 为机体朝向
- $v_{x,t}^{body}, v_{y,t}^{body}$ 为机体系线速度
- $\dot{\psi}_t$ 为机体系偏航角速度
- $\hat{d}_t$ 为缩放后的目标距离度量
- $[\hat{e}_{x,t}, \hat{e}_{y,t}]$ 为机体系目标方向单位向量

#### 目标方向编码 16 维

目标方向不是只保留一个角度，而是投影到 16 个 body-frame 方向 bins 中，形成连续强度编码。这使高层网络更容易学习“目标大致在哪个方向”。

#### 障碍物/边界编码 16 维

项目没有直接使用真实激光雷达，而是使用手工构造的极坐标距离编码。每个 bin 存储该方向上的最大危险强度。

如果某方向最近危险源的表面距离为 $d_{\text{surf}}$，最大感知距离为 $d_{\max}$，则强度定义为：

$$
I = 1 - \mathrm{clip}\left(\frac{d_{\text{surf}}}{d_{\max}}, 0, 1\right)
$$

距离越近，强度越大。

## 5. 几何量定义

### 5.1 目标距离

设机器人平面位置为 $p_t \in \mathbb{R}^2$，目标位置为 $g \in \mathbb{R}^2$，则：

$$
d_t = \|p_t - g\|_2
$$

项目中用于奖励的目标距离采用 `base_env.reach_metric`，并在观测中乘以 `reach_metric_scale`。

### 5.2 危险距离

项目中的危险源包括：

- 圆柱障碍物
- 环境边界

设机器人到最近危险表面的距离为 $h_t$，则：

$$
h_t = \min(\text{obstacle surface distance}, \text{boundary distance})
$$

奖励计算优先使用真实的最小危险距离，而不是从观测反推的估计值。

## 6. Reward Shaping 设计

高层 step 的奖励由以下几部分组成：

$$
r_t =
w_p r_t^{\text{progress}}
+
w_s r_t^{\text{safe}}
-
w_{sm} r_t^{\text{smooth}}
+
r_t^{\text{target-speed}}
+
r_t^{\text{terminal}}
$$

其中各项定义如下。

### 6.1 进展奖励

设上一个高层 step 的目标距离为 $d_{t-1}$，当前为 $d_t$，则：

$$
r_t^{\text{progress}} = d_{t-1} - d_t
$$

若机器人接近目标，则该项为正；若远离目标，则为负。

当前默认权重：

$$
w_p = \texttt{progress\_scale} = 4.5
$$

### 6.2 安全惩罚

设安全距离阈值为 $d_{\text{safe}}$，最近危险距离为 $h_t$，则：

$$
r_t^{\text{safe}} =
\begin{cases}
0, & h_t \ge d_{\text{safe}} \\
-\left(\frac{d_{\text{safe}} - h_t}{d_{\text{safe}}}\right)^2, & h_t < d_{\text{safe}}
\end{cases}
$$

这是一种连续、可微分、近障加速变大的二次惩罚。

当前默认权重：

$$
w_s = \texttt{safe\_scale} = 4.0
$$

### 6.3 动作平滑惩罚

设当前高层命令为 $c_t$，前一高层命令为 $c_{t-1}$，则：

$$
\Delta c_t = c_t - c_{t-1}
$$

平滑惩罚定义为：

$$
r_t^{\text{smooth}} = \|\Delta c_t\|_2^2
$$

总奖励中以负号引入：

$$
- w_{sm} r_t^{\text{smooth}}
$$

当前默认权重：

$$
w_{sm} = \texttt{smooth\_scale} = 0.05
$$

### 6.4 目标方向速度奖励

当机器人在安全区域内时，项目还会给一个沿目标方向的速度奖励。设机体系线速度为 $v_t^{body} \in \mathbb{R}^2$，目标方向单位向量为 $\hat{e}_t \in \mathbb{R}^2$，则沿目标方向的速度投影为：

$$
v_t^{\text{target}} = \langle v_t^{body}, \hat{e}_t \rangle
$$

先将其裁剪到 `[-1, 1]`，再乘以安全权重：

$$
\alpha_t = \mathrm{clip}\left(\frac{h_t - d_{\text{safe}}}{d_{\text{safe}}}, 0, 1\right)
$$

于是：

$$
r_t^{\text{target-speed}} =
w_{ts} \cdot \mathrm{clip}(v_t^{\text{target}}, -1, 1) \cdot \alpha_t
$$

当前默认权重：

$$
w_{ts} = \texttt{target\_speed\_scale} = 0.1
$$

这个项的目的不是替代 progress，而是减少“姿态正确但速度输出过保守”的情况。

### 6.5 终止奖励与惩罚

#### 成功奖励

当满足：

$$
d_t \le d_{\text{goal}}
$$

且该 step 被判为 episode 终止、同时未碰撞时，加入成功奖励：

$$
r_t^{\text{goal}} = +R_{\text{goal}}
$$

当前默认：

$$
d_{\text{goal}} = 0.3,\quad R_{\text{goal}} = 80.0
$$

#### 碰撞惩罚

当最近危险距离满足：

$$
h_t \le d_{\text{coll}}
$$

或底层环境已经报告失败终止时，视为碰撞/失败，加入：

$$
r_t^{\text{collision}} = -R_{\text{coll}}
$$

当前默认：

$$
d_{\text{coll}} = 0.35,\quad R_{\text{coll}} = 120.0
$$

#### 超时惩罚

若该 step 属于时间截断终止，则加入：

$$
r_t^{\text{timeout}} = -R_{\text{timeout}}
$$

当前默认：

$$
R_{\text{timeout}} = 10.0
$$

### 6.6 奖励缩放与裁剪

在所有 shaping 和终止项叠加完成后，项目会进行全局缩放：

$$
r_t \leftarrow \texttt{reward\_scale} \cdot r_t
$$

随后做裁剪：

$$
r_t \leftarrow \mathrm{clip}(r_t, -r_{\max}, r_{\max})
$$

当前默认：

$$
\texttt{reward\_scale} = 1.0,\quad r_{\max} = \texttt{reward\_clip} = 200.0
$$

## 7. 终止条件与成功定义

### 7.1 终止来源

高层 `done` 实际继承自底层环境累计得到的 `base_dones`，主要来源包括：

- 到达目标
- 危险碰撞
- 底层环境失败
- 超时

### 7.2 成功判定

项目中的成功并不是“到达过目标附近”这么简单，而是：

$$
\text{success} = \text{reached} \land \text{done} \land \neg \text{collision}
$$

这意味着成功 episode 必须同时满足：

- 到达目标阈值内
- episode 在该 step 结束
- 在成功 step 前未被判断为碰撞/失败

这个定义比单纯看 reach rate 更严格，因此日志中同时保留：

- `reach`
- `success`
- `collision`

## 8. PPO 优化器设计

### 8.1 基本形式

项目使用的是标准 PPO，而不是约束型 PPO 或 reach-avoid 专用价值结构。也就是说，优化目标仍然是：

$$
\mathcal{L}_{\text{PPO}} =
\mathbb{E}\left[
\max\left(
-A_t \rho_t,
-A_t \cdot \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)
\right)
\right]
+
c_v \mathcal{L}_{\text{value}}
- c_e \mathcal{H}
$$

其中：

$$
\rho_t = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{\text{old}}}(a_t|o_t)}
$$

不同点不在于 PPO 目标本身，而在于高层环境返回的奖励已经是 reward shaping 之后的结果。

### 8.2 GAE 与 return 计算

`RolloutStorage` 使用标准 GAE：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma \lambda (1-d_t) A_{t+1}
$$

其中 $d_t$ 为 done 标志。

随后：

$$
R_t = A_t + V(s_t)
$$

并对 advantage 做标准化处理。

### 8.3 Time-out bootstrapping

若 episode 因 time-out 结束，而非真实终止，则会进行 bootstrap：

$$
r_t \leftarrow r_t + \gamma V(s_{t+1})
$$

这是为了减少截断带来的价值偏差。

### 8.4 自适应学习率

当 `schedule = adaptive` 时，项目会根据 KL 偏差自动调节学习率。

若当前 KL 过大，则降低学习率；若 KL 过小，则适度提高学习率，范围受：

- `min_lr = 1e-5`
- `max_lr = 1e-3`

约束。

若 mini-batch 的近似 KL 明显超过阈值 `2 * desired_kl`，当前 batch 更新会被直接跳过，以避免策略崩坏。

### 8.5 Value clipping

价值函数损失采用 clipped value loss：

$$
V_t^{\text{clip}} =
V_t^{old} + \mathrm{clip}(V_t - V_t^{old}, -\epsilon_v, \epsilon_v)
$$

再对 unclipped/clipped 两种误差取较大值，增强价值更新稳定性。

## 9. 网络结构

### 9.1 Actor 网络

- 输入：高层观测
- 输出：3 维动作均值
- 隐层：`[512, 512, 512, 512]`
- 输出分布：对角高斯
- 采样后经 `tanh` 压缩

### 9.2 Critic 网络

- 输入：当前实现下与 actor 相同的观测
- 输出：标量状态价值
- 隐层：`[512, 512, 512, 512]`

### 9.3 动作噪声

策略维护一个可学习的标准差参数：

$$
\sigma = \text{learnable parameter}
$$

初始值由 `init_noise_std = 0.3` 指定，并在训练中自适应更新。

## 10. 当前默认超参数

### 10.1 Reward Shaping 参数

| 参数 | 默认值 |
|------|--------|
| `goal_reached_dist` | `0.3` |
| `collision_dist` | `0.35` |
| `safe_distance` | `1.5` |
| `progress_scale` | `4.5` |
| `target_speed_scale` | `0.1` |
| `goal_reward` | `80.0` |
| `safe_scale` | `4.0` |
| `smooth_scale` | `0.05` |
| `collision_penalty` | `120.0` |
| `timeout_penalty` | `10.0` |
| `reward_scale` | `1.0` |
| `reward_clip` | `200.0` |

### 10.2 PPO 参数

| 参数 | 默认值 |
|------|--------|
| `learning_rate` | `3e-5` |
| `clip_param` | `0.07` |
| `value_clip_param` | `0.2` |
| `value_loss_coef` | `0.5` |
| `entropy_coef` | `0.003` |
| `desired_kl` | `0.03` |
| `schedule` | `adaptive` |
| `num_learning_epochs` | `2` |
| `num_mini_batches` | `4` |
| `num_steps_per_env` | `400` |
| `max_grad_norm` | `1.0` |

## 11. 日志指标解释

训练脚本每个 iteration 会记录：

- `success`：严格成功率，到达且未碰撞
- `reach`：达到目标阈值的比例
- `collision`：碰撞或失败终止比例
- `timeout`：超时比例
- `boundary_collision_rate`：边界碰撞比例
- `obstacle_collision_rate`：障碍物碰撞比例
- `progress`：平均进展奖励原项
- `safety`：平均安全惩罚原项
- `smooth`：平均动作平滑惩罚原项
- `target_speed`：平均目标方向速度奖励
- `goal_dist`：平均目标距离
- `min_hazard`：平均最近危险距离
- `cmd_goal_corr`：命令速度与进展之间的相关性
- `reward_clip`：奖励被裁剪的比例
- `approx_kl`、`clip_frac`、`ratio_*`：PPO 更新稳定性指标
- `ppo_skipped`：被跳过的更新 batch 数量

## 12. 代码映射

### 奖励与终止逻辑

- `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
  - `step`
  - `_compute_reward`

### 高层观测与动作映射

- `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
  - `update_velocity_commands`
  - `_compute_high_level_observations`
  - `extract_target_distance`
  - `extract_hazard_distance`

### 训练入口与日志

- `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
  - `train_reward_shaping`

### PPO 与存储

- `rsl_rl/rsl_rl/algorithms/ppo.py`
- `rsl_rl/rsl_rl/storage/rollout_storage.py`
- `rsl_rl/rsl_rl/modules/actor_critic.py`

## 13. 实践建议

如果训练中出现以下现象，可优先排查：

- `reach` 高但 `success` 低：通常是到达前后仍发生碰撞，优先检查 `collision_dist`、`safe_distance`、目标区域周边障碍布局。
- `progress` 很低且 `collision` 很低：策略可能学成保守停滞，优先检查 `progress_scale` 与 `goal_reward` 是否偏弱。
- `approx_kl` 持续过高或 `ppo_skipped` 过多：说明更新过猛，优先降低学习率或收紧 `clip_param`。
- `smooth` 项非常大：说明高层命令震荡明显，可适当提高 `smooth_scale`。

## 14. 总结

本项目的核心不是修改 PPO 主体，而是把高层导航问题改写成一个“固定低层控制器 + 稠密 shaping 奖励 + 高层连续动作优化”的分层强化学习问题。

当前实现的关键优点是：

- 高层决策粒度清晰
- 奖励信号密集，易于训练
- 既区分 reach 与 success，也区分障碍物碰撞与边界碰撞
- PPO 侧增加了 value clipping、KL 自适应学习率与异常 batch 跳过逻辑，数值更稳
