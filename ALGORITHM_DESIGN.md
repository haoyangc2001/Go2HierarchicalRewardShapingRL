# Go2 Hierarchical Reward-Shaping RL 算法设计

## 1. 文档目的

本文档说明本项目当前实际采用的算法设计，而不是泛化方案草稿。内容覆盖：

- 分层控制结构
- 高层观测与动作定义
- Reward Shaping 奖励设计
- 终止条件与成功判定
- PPO 优化流程
- 关键配置项与代码映射

所有说明均以当前仓库实现为准，主要对应以下文件：

- `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- `legged_gym_go2/legged_gym/envs/go2/go2_config.py`
- `legged_gym_go2/legged_gym/scripts/train_reward_shaping.py`
- `rsl_rl/rsl_rl/algorithms/ppo.py`
- `rsl_rl/rsl_rl/storage/rollout_storage.py`
- `rsl_rl/rsl_rl/modules/actor_critic.py`

## 2. 任务定义

项目目标是在 Isaac Gym 中训练 Go2 的高层导航策略，使机器人在二维平面障碍环境中同时完成三件事：

1. 持续朝目标推进
2. 避免与障碍物或边界碰撞
3. 输出平滑且可被低层控制器稳定跟踪的速度命令

这里训练的不是关节动作策略，而是高层导航策略。高层策略输出速度命令，低层策略负责执行。

## 3. 分层控制结构

### 3.1 高层与低层分工

高层策略输出三维导航动作：

$$
a_t = [v_x, v_y, \omega_z]
$$

其中：

- $v_x$ 是前向速度命令
- $v_y$ 是侧向速度命令
- $\omega_z$ 是偏航角速度命令

低层策略是固定的预训练 locomotion policy，用于把速度命令转换成关节级动作。

整个系统可写为：

$$
o_t \xrightarrow{\pi_\theta} a_t \xrightarrow{\phi} c_t \xrightarrow{\pi_{\mathrm{low}}} u_t \to s_{t+1}
$$

其中：

- $o_t$ 是高层观测
- $\pi_\theta$ 是待训练的高层策略
- $\phi$ 是命令映射过程
- $c_t$ 是底层速度命令
- $\pi_{\mathrm{low}}$ 是固定低层策略
- $u_t$ 是底层输出的关节动作

### 3.2 宏步长交互

高层策略不是每个仿真步都重新决策。当前实现中，一个高层动作会在底层连续执行 $K$ 个仿真步，其中：

$$
K = 5
$$

这个值对应配置项 `high_level_action_repeat`。

这样做的好处是：

- 降低高层动作抖动
- 让高层更专注于导航而不是步态细节
- 让 rollout 更符合导航层时间尺度

## 4. 高层动作与观测

### 4.1 高层动作空间

策略网络输出三维连续动作，先经过 `tanh` 压缩到区间 $[-1,1]$：

$$
a_t \in [-1,1]^3
$$

然后动作先乘以配置中的尺度向量：

$$
\tilde{a}_t = a_t \odot [1.3, 1.0, 1.0]
$$

再映射到底层命令：

$$
c_t =
\left[
0.6 \tilde{a}_t^{(1)},
0.2 \tilde{a}_t^{(2)},
0.8 \tilde{a}_t^{(3)}
\right]
$$

此外，偏航相关的 heading 命令按照下式设置：

$$
c_t^{(h)} = \psi_t + 2 c_t^{(3)}
$$

其中 $\psi_t$ 是当前机体朝向角。

### 4.2 高层观测空间

当前默认高层观测总维度为 40，由三部分组成：

1. 基础特征 8 维
2. 目标方向编码 16 维
3. 障碍物与边界编码 16 维

基础特征可以写成：

$$
o_t^{(b)} =
\left[
\cos \psi_t,\,
\sin \psi_t,\,
v_{x,t},\,
v_{y,t},\,
\dot{\psi}_t,\,
\bar{d}_t,\,
\hat{e}_{x,t},\,
\hat{e}_{y,t}
\right]
$$

其中：

- $\psi_t$ 是机体朝向
- $v_{x,t}, v_{y,t}$ 是机体系线速度
- $\dot{\psi}_t$ 是机体系偏航角速度
- $\bar{d}_t$ 是缩放后的目标距离度量
- $\hat{e}_t = [\hat{e}_{x,t}, \hat{e}_{y,t}]$ 是机体系目标方向单位向量

目标方向编码和障碍物编码都采用离散方向 bins 的方式构造，这样高层网络更容易学习目标方向和危险方向。

### 4.3 手工 lidar 强度编码

项目没有直接使用真实激光雷达，而是构造了方向离散化的强度特征。

如果某个方向上最近危险源的表面距离为 $d_{\mathrm{surf}}$，最大感知距离为 $d_{\max}$，则该方向的强度定义为：

$$
I = 1 - \operatorname{clip}\left(\frac{d_{\mathrm{surf}}}{d_{\max}}, 0, 1\right)
$$

距离越近，强度越大。

## 5. 几何量定义

### 5.1 目标距离

设机器人平面位置为 $p_t \in \mathbb{R}^2$，目标位置为 $g \in \mathbb{R}^2$，则目标距离定义为：

$$
d_t = \lVert p_t - g \rVert_2
$$

项目中奖励部分优先使用环境内部的真实目标距离度量，而不是仅从观测反推。

### 5.2 危险距离

项目中的危险源包含两类：

- 圆柱障碍物
- 场地边界

设机器人到最近危险表面的距离为 $h_t$，则：

$$
h_t = \min \left(h_t^{(\mathrm{obs})}, h_t^{(\mathrm{bd})}\right)
$$

其中：

- $h_t^{(\mathrm{obs})}$ 是到最近障碍物表面的距离
- $h_t^{(\mathrm{bd})}$ 是到边界的距离

## 6. Reward Shaping 设计

### 6.1 总体形式

高层 step 的总奖励写成：

$$
r_t = w_p r_t^{(p)} + w_s r_t^{(s)} - w_{sm} r_t^{(sm)} + r_t^{(ts)} + r_t^{(\mathrm{term})}
$$

其中：

- $r_t^{(p)}$ 是进展奖励
- $r_t^{(s)}$ 是安全惩罚
- $r_t^{(sm)}$ 是动作平滑惩罚
- $r_t^{(ts)}$ 是目标方向速度奖励
- $r_t^{(\mathrm{term})}$ 是终止奖励或惩罚

下面分别说明。

### 6.2 进展奖励

设上一个高层 step 的目标距离为 $d_{t-1}$，当前目标距离为 $d_t$，则：

$$
r_t^{(p)} = d_{t-1} - d_t
$$

如果机器人朝目标接近，则该项为正；如果远离目标，则该项为负。

当前代码中的默认权重为：

- `progress_scale = 4.5`

### 6.3 安全惩罚

设安全距离阈值为 $d_s$，最近危险距离为 $h_t$，则安全惩罚为：

$$
r_t^{(s)} =
\begin{cases}
0, & h_t \ge d_s \\
-\left(\frac{d_s - h_t}{d_s}\right)^2, & h_t < d_s
\end{cases}
$$

这是一种连续的近障二次惩罚。进入危险区域后，惩罚会随着接近速度加快。

当前代码中的默认配置为：

- `safe_distance = 1.5`
- `safe_scale = 4.0`

### 6.4 动作平滑惩罚

设当前高层命令为 $c_t$，前一高层命令为 $c_{t-1}$，则：

$$
\Delta c_t = c_t - c_{t-1}
$$

平滑惩罚定义为：

$$
r_t^{(sm)} = \lVert \Delta c_t \rVert_2^2
$$

该项在总奖励中以前面的负号出现，用于抑制命令跳变。

当前代码中的默认权重为：

- `smooth_scale = 0.05`

### 6.5 目标方向速度奖励

在安全区域内，项目还会给一个沿目标方向前进的速度奖励。

设机体系线速度为 $v_t \in \mathbb{R}^2$，目标方向单位向量为 $\hat{e}_t \in \mathbb{R}^2$，则沿目标方向的速度分量为：

$$
v_t^{(\mathrm{tar})} = v_t^\top \hat{e}_t
$$

同时定义安全权重：

$$
\alpha_t = \operatorname{clip}\left(\frac{h_t - d_s}{d_s}, 0, 1\right)
$$

于是该奖励项为：

$$
r_t^{(ts)} = w_{ts} \cdot \operatorname{clip}\left(v_t^{(\mathrm{tar})}, -1, 1\right) \cdot \alpha_t
$$

其作用是鼓励机器人在安全条件较好的时候沿目标方向产生真实前进速度，而不是只在姿态上对准目标。

当前代码中的默认权重为：

- `target_speed_scale = 0.1`

### 6.6 终止奖励与惩罚

#### 成功奖励

若机器人到达目标阈值内，即：

$$
d_t \le d_g
$$

则在成功终止时给予正奖励：

$$
r_t^{(\mathrm{goal})} = +R_g
$$

当前默认配置为：

- `goal_reached_dist = 0.3`
- `goal_reward = 80.0`

#### 碰撞惩罚

若最近危险距离小于碰撞阈值，即：

$$
h_t \le d_c
$$

则视为碰撞或失败终止，给予惩罚：

$$
r_t^{(\mathrm{coll})} = -R_c
$$

当前默认配置为：

- `collision_dist = 0.35`
- `collision_penalty = 120.0`

#### 超时惩罚

若 episode 是由时间截断导致终止，则给予超时惩罚：

$$
r_t^{(\mathrm{to})} = -R_{to}
$$

当前默认配置为：

- `timeout_penalty = 10.0`

### 6.7 奖励缩放与裁剪

所有奖励项叠加后，项目还会进行全局缩放与裁剪：

$$
r_t \leftarrow \lambda_r r_t
$$

$$
r_t \leftarrow \operatorname{clip}\left(r_t, -r_{\max}, r_{\max}\right)
$$

当前代码中对应：

- `reward_scale = 1.0`
- `reward_clip = 200.0`

## 7. 终止条件与成功定义

### 7.1 终止来源

高层环境中的 `done` 继承自底层环境累计得到的终止标志，主要来源包括：

- 到达目标
- 碰撞或失败
- 超时

### 7.2 成功定义

当前实现中的成功不是“到达过目标附近”这么简单，而是满足：

$$
\mathrm{success} = \mathrm{reached} \wedge \mathrm{done} \wedge \neg \mathrm{collision}
$$

也就是说，只有“到达目标且在成功 step 上没有被判定为碰撞”的 episode 才算成功。

因此日志里会同时保留：

- `reach`
- `success`
- `collision`

## 8. PPO 优化器设计

### 8.1 基本目标

本项目使用标准 PPO，而不是约束型 PPO 或 reach-avoid 特化价值结构。其基本优化目标可写为：

$$
L_{\mathrm{PPO}} =
\mathbb{E}
\left[
\max
\left(
-A_t \rho_t,\,
-A_t \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)
\right)
\right]
+
c_v L_V
- c_e H
$$

其中概率比为：

$$
\rho_t =
\frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid o_t)}
$$

这里的不同点不在 PPO 主体，而在于环境返回的是 Reward Shaping 之后的高层奖励。

### 8.2 GAE 与 return

`RolloutStorage` 使用标准 GAE。单步 TD 误差为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

优势函数递推为：

$$
A_t = \delta_t + \gamma \lambda (1-d_t) A_{t+1}
$$

其中 $d_t$ 是 done 标志。

最终 return 为：

$$
R_t = A_t + V(s_t)
$$

之后会对 advantage 做标准化。

### 8.3 Time-out bootstrapping

如果 episode 由于超时而不是任务终止结束，项目会做 bootstrap 修正：

$$
r_t \leftarrow r_t + \gamma V(s_{t+1})
$$

这样可以减小时间截断带来的价值偏差。

### 8.4 自适应学习率

当 `schedule = adaptive` 时，项目会根据 KL 大小自动调整学习率。

当前默认学习率边界为：

- `min_lr = 1e-5`
- `max_lr = 1e-3`

如果 mini-batch 的近似 KL 过大，当前 batch 更新会被直接跳过，以避免策略更新过猛。

### 8.5 Value clipping

价值函数更新使用 clipped value loss。被裁剪的价值估计写为：

$$
V_t^{(\mathrm{clip})}
=
V_t^{(\mathrm{old})}
+
\operatorname{clip}\left(
V_t - V_t^{(\mathrm{old})},
-\epsilon_v,
\epsilon_v
\right)
$$

然后在 clipped 与 unclipped 两种误差之间取较大值，提升数值稳定性。

## 9. 网络结构

### 9.1 Actor 网络

- 输入：高层观测
- 输出：3 维动作均值
- 隐层：`[512, 512, 512, 512]`
- 输出分布：对角高斯
- 动作采样后经 `tanh` 压缩

### 9.2 Critic 网络

- 输入：当前实现中与 actor 相同的观测
- 输出：标量状态价值
- 隐层：`[512, 512, 512, 512]`

### 9.3 动作噪声

策略维护一个可学习的动作标准差向量：

$$
\sigma \in \mathbb{R}^3
$$

当前初始值由配置项 `init_noise_std = 0.3` 给出。

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

训练脚本每个 iteration 会记录以下核心指标：

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
- `cmd_goal_corr`：命令速度与进展的相关性
- `reward_clip`：奖励被裁剪的比例
- `approx_kl`、`clip_frac`、`ratio_*`：PPO 更新稳定性指标
- `ppo_skipped`：被跳过的 mini-batch 数量

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

- `reach` 高但 `success` 低：通常表示到达目标附近后仍发生碰撞，应优先检查 `collision_dist`、`safe_distance` 和目标周边障碍布局。
- `progress` 很低且 `collision` 也很低：通常表示策略过于保守，应优先检查 `progress_scale` 和 `goal_reward` 是否偏弱。
- `approx_kl` 持续过高或 `ppo_skipped` 过多：说明更新过猛，应优先降低学习率或收紧 `clip_param`。
- `smooth` 项非常大：说明高层命令震荡明显，可适当提高 `smooth_scale`。

## 14. 总结

本项目的核心不是改写 PPO 主体，而是把高层导航问题改写成一个“固定低层控制器 + 稠密 Reward Shaping + 高层连续动作优化”的分层强化学习问题。

当前实现的关键优点是：

- 高层决策粒度清晰
- 奖励信号密集，训练更容易启动
- 同时区分 `reach` 与 `success`
- 同时区分障碍物碰撞与边界碰撞
- PPO 侧加入了 value clipping、KL 自适应学习率和异常 batch 跳过逻辑，整体数值更稳
