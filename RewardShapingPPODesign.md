# IsaacGym 四足机器人分层导航：上层策略 Reward Shaping + 标准 PPO 训练方案（用于代码实现说明）

> 目标：在 IsaacGym 中训练**上层导航策略**，使四足机器人到达目标点，同时全程不碰撞障碍物。  
> 分层结构：  
> - **上层策略（需训练）**：感知环境，输出三维速度指令动作 `a_t = [v_x, v_y, ω]`（x轴速度、y轴速度、角速度），负责导航避障。  
> - **底层策略（已训练、确定性、固定）**：接收上层指令并输出各关节动作，负责稳定跟踪控制。  
> 方法：使用**标准 PPO**，通过**奖励塑形（reward shaping）**让上层与“障碍物 + 目标点”环境有效交互并学会避障到达。

---

## 1. 上层与底层交互方式：宏步长环境封装（强烈建议）

由于底层控制器固定且确定性，上层采用**低频决策**：上层每一步动作在底层执行 `K` 个仿真步（sim steps）。

### 高层 `step(action)` 推荐实现

- 输入：上层动作 `a_t = [v_x, v_y, ω]`（保持现有上层动作设计不变）
- 内部：底层控制器执行 `K` 个 sim steps 来跟踪该速度指令
- 返回：高层一步的 transition
  - `o_{t+1}`：下一高层观测
  - `r_t`：高层奖励（累计 K 步形成的 reward）
  - `done`：若在 K 步内到达目标/碰撞/超时等则终止
  - `info`：调试信息（dist、d_min、是否碰撞、是否到达等）

### 奖励累计方式（宏步长）

- 高层奖励：
  \[
  r_t = \sum_{i=1}^{K} r^{(i)}
  \]
- done 逻辑：
  - 碰撞：立即 `done=True`
  - 到达：立即 `done=True`
  - 超时：`done=True`

> 推荐 `K需根据控制频率与导航频率调整，这样上层 rollouts 更稳定，也更符合“导航决策频率 < 控制执行频率”的分层结构。

---

## 2. Reward Shaping 总体结构（上层 PPO 的核心）

每个**高层 step** 的总奖励建议由 4 部分组成：

\[
r_t = w_p r_{\text{progress}} + w_g r_{\text{goal}} + w_s r_{\text{safe}} + w_{sm} r_{\text{smooth}} \ +\ r_{\text{collision}}
\]

其中各项定义如下（推荐使用稠密信号以提高学习效率）。

---

## 3. 进展奖励（Progress Reward，最关键）

保证策略会朝目标前进，避免学成“原地不动不撞”。

- 设目标距离：
  \[
  d_t = \|p_t - g\|
  \]
- 进展奖励采用“距离减少量”：
  \[
  r_{\text{progress}} = d_{t-1} - d_t
  \]

> 说明：  
> - 这是一种密集、稳定、尺度鲁棒的 shaping。  
> - 本质上等价于常见的潜在函数（potential）塑形形式之一。

---

## 4. 到达奖励（Goal Bonus）

到达目标区域时给一次较大的正奖励，并终止 episode：

- 若 `d_t < d_goal`：
  - `done=True`
  - \[
    r_{\text{goal}} = +R_{\text{goal}}
    \]
- 否则：
  - \[
    r_{\text{goal}} = 0
    \]

> `R_goal` 通常要显著大于单步 progress 的累计值，以确保策略强烈偏好完成任务。

---

## 5. 安全避障奖励塑形（Safety Shaping，关键：稠密近障惩罚）

仅靠“碰撞大惩罚”信号稀疏，会导致学习慢。必须加入**近障稠密惩罚**。

### 5.1 近障信息（需要环境提供/可计算）

在每个 sim step 或高层 step 内，获得/计算：
- `d_min`：机器人到最近障碍物的距离（可来自传感器观测或几何计算）
- `collision_detected`：是否发生碰撞（接触事件、碰撞 flag 等）

### 5.2 近障惩罚（推荐：两段式 + 平滑）

设安全距离阈值 `d_safe`：

\[
r_{\text{safe}}=
\begin{cases}
0 & d_{\min} \ge d_{\text{safe}}\\
-\left(\frac{d_{\text{safe}}-d_{\min}}{d_{\text{safe}}}\right)^2 & d_{\min}<d_{\text{safe}}
\end{cases}
\]

- 当 `d_min` 大于安全阈值：不惩罚
- 进入危险区：惩罚随接近程度平方增长（更强硬，更易学会保持距离）

> 可选替代（更硬但更敏感）：指数惩罚 `-exp(-k*d_min)`。

---

## 6. 碰撞惩罚（Collision Penalty）+ 终止

碰撞必须强力惩罚，并立即终止：

- 若发生碰撞：
  - `done=True`
  - \[
    r_{\text{collision}} = -R_{\text{coll}}
    \]
- 否则：
  - \[
    r_{\text{collision}} = 0
    \]

> `R_coll` 通常与 `R_goal` 同量级或更大，以形成“宁可绕路也不要撞”的强偏好。

---

## 7. 动作平滑（Smoothness，稳定高层指令，利于底层跟踪）

为了防止高层输出抖动导致贴障/碰撞，建议在高层 step 上加入动作变化惩罚：

\[
r_{\text{smooth}} = -\|a_t - a_{t-1}\|^2
\]

- 也可以只对角速度变化或横向速度变化惩罚（视你任务特性）
- 平滑项通常权重较小，但对稳定性帮助很大

---

## 8. 终止条件（Episode Done）

建议至少包含：

- `collision == True`：碰撞终止
- `dist_to_goal < d_goal`：到达终止
- `time_step >= T_max`：超时终止（避免无限 episode）

---

## 9. PPO 算法层面需要做的改动

标准 PPO，核心不变。需要确保：

1. **Rollout 使用“高层 step”**
2. Reward / done 的计算和存储使用上面 shaping 后的 `r_t`
3. advantage / return 在高层时间尺度上计算（GAE/discount 等照常）
4. 并行环境统计：episode reward、collision rate、success rate、平均最小距离等用于监控

> 其余 PPO 细节使用标准ppo模式：clip、entropy bonus、value loss、GAE 等。

---
