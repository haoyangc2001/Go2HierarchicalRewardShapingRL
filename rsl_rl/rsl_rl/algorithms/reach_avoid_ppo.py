from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def _calculate_reach_gae(
    gamma: float,
    lam: float,
    g_seq: torch.Tensor,
    value_seq: torch.Tensor,
    done_seq: torch.Tensor,
    h_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reach-Avoid 任务的广义优势估计（GAE）计算。

    这是标准 GAE 的扩展版本，同时考虑到达目标（g-values）和避障安全（h-values）的双重目标。
    算法基于 JAX 参考实现 calculate_gae_reach4 移植到 PyTorch。

    数学原理：
    - 对于每个时间步 t，计算值函数目标 Q_t = max(h_t, min(g_t, γ * V_{t+1}))
    - 物理意义：值函数应反映"安全约束"和"到达目标进度"中的较大者
    - 优化目标：在保证安全（h_t < 0）的前提下，最小化到达代价（g_t）   
    - 其中 g_t 是到达目标的代价，h_t 是避障安全约束
    - 使用 GAE 系数平滑多步回报，平衡偏差和方差.


    输入参数：
        gamma: float - 折扣因子，控制未来回报的重要性
        lam: float - GAE 参数，控制优势估计的偏差-方差权衡（λ∈[0,1]）
        g_seq: torch.Tensor - 形状 (horizon+1, num_envs)，到达目标函数值序列
        value_seq: torch.Tensor - 形状 (horizon+1, num_envs)，价值函数预测序列
        done_seq: torch.Tensor - 形状 (horizon, num_envs)，终止标志（环境终止或安全违规）
        h_seq: torch.Tensor - 形状 (horizon+1, num_envs)，避障安全函数值序列

    返回：
        Tuple[torch.Tensor, torch.Tensor] - (advantages, q_targets)
            advantages: 形状 (horizon, num_envs)，优势估计值
            q_targets: 形状 (horizon, num_envs)，Q 值目标用于值函数训练

    注意：
        - g_seq 和 h_seq 比 value_seq 多一个时间步（包含终止状态）
        - 当 h_t >= 0 时表示安全约束被违反，应终止回合
        - 算法从后向前计算，处理终止状态对回报传播的影响
    """
    # 获取输入张量的设备和数据类型，确保后续计算在相同设备上进行
    device = value_seq.device
    dtype = value_seq.dtype

    # 计算时间步长和环境数量
    # g_seq 形状: (horizon+1, num_envs)，value_seq 相同
    horizon = g_seq.shape[0] - 1  # 有效时间步数（排除终止状态）
    num_envs = g_seq.shape[1]     # 并行环境数量

    # 边界检查：如果 horizon <= 0，返回空张量
    if horizon <= 0:
        empty = torch.zeros(0, num_envs, device=device, dtype=dtype)
        return empty, empty

    # GAE 参数计算
    # lam_ratio = λ/(1-λ)，用于处理终止状态后的系数更新
    # 添加微小值防止除零错误
    lam_ratio = lam / max(1.0 - lam, 1e-6)
    # 将标量 gamma 转换为与 value_seq 相同设备和类型的张量
    gamma_tensor = value_seq.new_tensor(gamma)

    # 初始化计算所需的张量
    # GAE 系数：形状 (horizon+1, num_envs)，存储每个未来时间步的折扣权重
    gae_coeffs = torch.zeros(horizon + 1, num_envs, device=device, dtype=dtype)
    # 值表：形状同 gae_coeffs，存储后续状态的值函数估计
    value_table = torch.zeros_like(gae_coeffs)
    value_table[0] = value_seq[-1]  # 初始化为终止状态的价值函数估计
    # 前一步终止标志：记录上一个时间步哪些环境已终止
    prev_done = torch.zeros(num_envs, device=device, dtype=dtype)

    # Q 值目标：形状 (horizon, num_envs)，存储每个时间步的 TD 目标
    q_targets = torch.zeros(horizon, num_envs, device=device, dtype=dtype)
    # 索引掩码：用于限制有效时间步范围，形状 (horizon+1, 1)
    index_mask = torch.arange(horizon + 1, device=device).unsqueeze(1)

    # 主循环：从后向前遍历每个时间步（动态规划），动态规划反向计算
    for idx in range(horizon - 1, -1, -1):
        # 获取当前时间步的终止标志，并转换为计算数据类型
        done_row = done_seq[idx].to(dtype)
        done_row_unsqueezed = done_row.unsqueeze(0)  # 增加时间步维度
        prev_done_unsqueezed = prev_done.unsqueeze(0)  # 增加时间步维度

        # 更新 GAE 系数：考虑终止状态对回报传播的影响
        # 将系数向右滚动一位（时间步向前），为当前步腾出位置
        rolled = torch.roll(gae_coeffs, shifts=1, dims=0)
        # GAE 系数更新公式：
        # - 如果前一步未终止：系数衰减 λ
        # - 如果前一步已终止：使用 λ_ratio = λ/(1-λ) 重新初始化
        # - 如果当前步终止：系数置零（终止后无未来回报）
        # 物理意义：处理终止状态的回报传播

        # 时间线: ... t-1  →  t   →  t+1 (已处理)
        #             当前步   下一步（已计算）
        # - rolled: 将 GAE 系数向右滚动一位，为当前时间步腾出位置
        # - 第一项 rolled * lam * (1.0 - prev_done)：
        #     - 如果前一步未终止：系数衰减 λ（标准 GAE 更新）
        #     - 数学：γλ 衰减，λ∈[0,1] 控制偏差-方差权衡
        # - 第二项 rolled * lam_ratio * prev_done：
        #     - 如果前一步已终止：重新初始化系数
        #     - lam_ratio = λ/(1-λ)：确保系数和为 1
        # - 乘以 (1.0 - done_row)：
        #     - 如果当前步终止：系数置零（终止后无未来回报）
        # - gae_coeffs[0] = 1.0：
        #     - 当前时间步本身的系数为 1（自身回报权重最大）

        # 示例：λ=0.95 时的系数演变

        # 步数:   ...  t-3   t-2   t-1   t   (当前)
        # 系数:   ...  0.86  0.90  0.95  1.0
        # 每个未来时间步的权重按 λ 指数衰减。
        gae_coeffs = (
            rolled * lam * (1.0 - prev_done_unsqueezed)
            + rolled * lam_ratio * prev_done_unsqueezed
        ) * (1.0 - done_row_unsqueezed)
        gae_coeffs[0] = 1.0  # 当前时间步的系数为 1




        # 创建时间步掩码：只考虑 idx+1 之前的时间步（未来步骤）
        mask = (index_mask < (idx + 1)).to(dtype)

        # 计算折扣后的值函数估计：γ * V_{t+1}
        disc_to_gh = gamma_tensor * value_table

        # Reach-Avoid 核心计算：Q_t = max(h_t, min(g_t, γ * V_{t+1}))
        # - g_t: 到达目标的代价（越小越好）
        # - h_t: 避障安全约束（负值表示安全，≥0 表示违规）
        # - γ * V_{t+1}: 折扣后的未来值函数估计
        # 先取 min(g_t, γ*V_{t+1})，再与 h_t 取 max
        # 物理意义：值函数应是安全约束和到达目标中的较大者
        vhs_row = torch.maximum(
            h_seq[idx].unsqueeze(0),  # 安全约束
            torch.minimum(g_seq[idx].unsqueeze(0), disc_to_gh),  # 目标与未来值的较小者
        )
        vhs_row = vhs_row * mask  # 应用时间步掩码，限制有效范围

        # 计算 Q 值目标：加权求和未来时间步的 vhs_row
        # 1. 计算 GAE 系数和，防止除零错误
        coeff_sum = gae_coeffs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        # 2. 归一化系数，使和为 1（概率分布）
        norm_coeffs = gae_coeffs / coeff_sum
        # 3. 计算当前时间步的 Q 值目标：加权平均未来 vhs_row
        q_targets[idx] = (vhs_row * norm_coeffs).sum(dim=0)

        # 更新值表用于下一个时间步（前一步）的计算
        # 1. 将 vhs_row 向右滚动一位：当前步的值变为下一步的"未来值"
        vhs_row = torch.roll(vhs_row, shifts=1, dims=0)
        # 2. 设置索引 0 为下一个状态的价值函数估计 V_{t+1}
        vhs_row[0] = value_seq[idx + 1]
        # 3. 更新值表
        value_table = vhs_row

        # 更新前一步终止标志，用于下一个循环迭代
        prev_done = done_row

    # 计算优势估计：A_t = Q_t - V_t
    # 优势表示当前动作相对于平均水平的优劣程度
    advantages = q_targets - value_seq[:-1]

    # 返回优势估计和 Q 值目标
    return advantages, q_targets


@dataclass
class ReachAvoidBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReachAvoidBuffer:
    def __init__(self, num_envs: int, horizon: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device: torch.device):
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device

        obs_dim = obs_shape[0]
        act_dim = action_shape[0]

        self.observations = torch.zeros(horizon + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.advantages = torch.zeros(horizon, num_envs, device=device)
        self.returns = torch.zeros(horizon, num_envs, device=device)
        self.g_values = torch.zeros(horizon + 1, num_envs, device=device)
        self.h_values = torch.zeros(horizon + 1, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device, dtype=torch.bool)

        self.step = 0

    def clear(self) -> None:
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        g_values: torch.Tensor,
        h_values: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_g: torch.Tensor,
        next_h: torch.Tensor,
    ) -> None:
        idx = self.step
        self.observations[idx] = obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.g_values[idx] = g_values
        self.h_values[idx] = h_values
        self.dones[idx] = dones.bool()
        self.observations[idx + 1] = next_obs
        self.g_values[idx + 1] = next_g
        self.h_values[idx + 1] = next_h
        self.step += 1

    def store_rollout(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        g_values: torch.Tensor,
        h_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        批量存储完整的 rollout 轨迹数据到缓冲区。
        与 add() 方法逐步添加单步数据不同，此方法一次性存入整个 horizon 长度的轨迹。
        常用于外部已经收集好完整轨迹的场景，或从预收集数据加载。
        """
        # 检查输入数据形状与缓冲区预分配形状是否一致
        # 确保数据完整性，防止维度不匹配导致的后续计算错误
        if observations.shape != self.observations.shape:
            raise ValueError("observations shape mismatch")
        if actions.shape != self.actions.shape:
            raise ValueError("actions shape mismatch")
        if log_probs.shape != self.log_probs.shape:
            raise ValueError("log_probs shape mismatch")
        if values.shape != self.values.shape:
            raise ValueError("values shape mismatch")
        if g_values.shape != self.g_values.shape:
            raise ValueError("g_values shape mismatch")
        if h_values.shape != self.h_values.shape:
            raise ValueError("h_values shape mismatch")
        if dones.shape != self.dones.shape:
            raise ValueError("dones shape mismatch")

        # 批量复制数据到缓冲区，使用 copy_() 进行原地更新以提高效率
        self.observations.copy_(observations)
        self.actions.copy_(actions)
        self.log_probs.copy_(log_probs)
        self.values.copy_(values)
        self.g_values.copy_(g_values)
        self.h_values.copy_(h_values)
        self.dones.copy_(dones.bool())
        # 设置 step 为 horizon，标记缓冲区已满，可以执行优势计算和策略更新
        self.step = self.horizon


    def compute_advantages(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        if self.step != self.horizon:
            raise RuntimeError("incomplete rollout stored in buffer")

        value_seq = torch.cat((self.values, last_values.unsqueeze(0)), dim=0)
        env_dones = self.dones
        safety_dones = self.h_values[:-1] >= 0
        done_seq = torch.logical_or(env_dones, safety_dones)
        adv, targets = _calculate_reach_gae(gamma, lam, self.g_values, value_seq, done_seq, self.h_values)
        self.advantages.copy_(adv)
        self.returns.copy_(targets)

    def _flat_view(self) -> ReachAvoidBatch:
        obs = self.observations[:-1].reshape(-1, self.observations.size(-1))
        actions = self.actions.reshape(-1, self.actions.size(-1))
        log_probs = self.log_probs.reshape(-1)
        values = self.values.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        return ReachAvoidBatch(obs, actions, log_probs, values, advantages, returns)

    def iter_batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[ReachAvoidBatch]:
        """
        生成用于PPO多轮小批量训练的数据批次。
        
        PPO算法通常进行多轮（epochs）训练，每轮中将所有数据随机打乱后分割成多个小批量（mini-batches）。
        这种设计提高数据利用率，减少过拟合，并通过随机化改善训练稳定性。
        
        参数:
            num_mini_batches: 每个训练轮次中的小批量数量
            num_epochs: 训练轮次数量（对同一批数据进行多轮训练）
        
        返回:
            生成器，每次迭代返回一个ReachAvoidBatch数据对象
        """
        # 步骤1: 将缓冲区中的多维数据展平为二维张量
        # _flat_view() 将形状为 [horizon, num_envs, ...] 的数据展平为 [horizon*num_envs, ...]
        # 例如：observations从[horizon, num_envs, obs_dim]变为[horizon*num_envs, obs_dim]
        data = self._flat_view()

        # 步骤2: 计算总样本数和小批量大小
        # batch_size = horizon * num_envs，即总的时间步-环境对数量
        batch_size = data.observations.size(0)
        # mini_batch_size = 总样本数 // 小批量数量，确保整除
        mini_batch_size = batch_size // num_mini_batches

        # 步骤3: 外层循环 - 训练轮次（epochs）
        # 对同一批数据训练多轮，提高数据利用率
        for _ in range(num_epochs):
            # 步骤3.1: 随机打乱所有样本的索引
            # torch.randperm生成0到batch_size-1的随机排列，确保每轮数据顺序不同
            # 随机化防止模型过拟合到特定的数据顺序，改善训练稳定性
            indices = torch.randperm(batch_size, device=self.device)

            # 步骤3.2: 内层循环 - 分割成小批量
            # 从0开始，以mini_batch_size为步长，遍历所有样本
            for start in range(0, batch_size, mini_batch_size):
                # 计算当前小批量的结束位置（不包含）
                end = start + mini_batch_size
                # 从打乱的索引中取出当前小批量的索引
                idx = indices[start:end]

                # 步骤3.3: 构建并返回当前小批量的数据
                # 使用生成器（yield）逐个返回小批量，避免一次性加载所有数据到内存
                # 每个ReachAvoidBatch包含PPO更新所需的所有字段：
                #   - observations: 状态观测 [mini_batch_size, obs_dim]
                #   - actions: 执行的动作 [mini_batch_size, action_dim]
                #   - old_log_probs: 采样时的动作对数概率 [mini_batch_size]
                #   - old_values: 采样时的价值函数估计 [mini_batch_size]
                #   - advantages: 计算的优势值 [mini_batch_size]
                #   - returns: 计算的回报（Q值目标）[mini_batch_size]
                yield ReachAvoidBatch(
                    data.observations[idx],      # 当前小批量的状态观测
                    data.actions[idx],           # 当前小批量的动作
                    data.old_log_probs[idx],     # 当前小批量的旧对数概率（用于重要性采样）
                    data.old_values[idx],        # 当前小批量的旧价值估计（用于值函数裁剪）
                    data.advantages[idx],        # 当前小批量的优势估计
                    data.returns[idx],           # 当前小批量的回报（Q值目标）
                )



class ReachAvoidPPO:
    def __init__(
        self,
        actor_critic,
        learning_rate: float = 3e-4,
        gamma: float = 0.999,
        lam: float = 0.95,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.buffer = None
        self.last_value_stats = {}

    def init_storage(self, num_envs: int, horizon: int, obs_shape, action_shape) -> None:
        obs_shape = tuple(obs_shape) if isinstance(obs_shape, (list, tuple)) else (obs_shape,)
        action_shape = tuple(action_shape) if isinstance(action_shape, (list, tuple)) else (action_shape,)
        self.buffer = ReachAvoidBuffer(num_envs, horizon, obs_shape, action_shape, self.device)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            actions = self.actor_critic.act(obs)
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            values = self.actor_critic.evaluate(obs).squeeze(-1)
        return actions, log_probs, values

    # 角色说明：
    # `update()` 是在每次收集完一段完整 rollout（由 ReachAvoidBuffer 存储）后调用的训练步骤。
    # 在整个强化学习训练流程中，它负责：
    # - 从缓冲区读取已计算的优势值与回报（returns）；
    # - 统计并记录价值函数相关指标（均值、方差、解释方差等）；
    # - 对优势进行归一化以稳定训练；
    # - 按 mini-batch 和多轮 epoch 运行 PPO 的剪切目标与价值函数更新（带梯度裁剪与熵正则）。
    # 返回值：平均策略损失（policy loss）与平均价值损失（value loss），便于监控训练趋势。
    def update(self) -> Tuple[float, float]:
        """
        PPO 算法核心更新步骤（针对 Reach-Avoid 任务的变体）。
        该函数不负责数据采集（由外部环境和 buffer 完成），只在完整 rollout 可用时被调用以执行参数更新。
        """
        assert self.buffer is not None

        advantages = self.buffer.advantages
        with torch.no_grad():  # 计算价值统计量，无需梯度
            values_flat = self.buffer.values.reshape(-1)
            returns_flat = self.buffer.returns.reshape(-1)
            diff = returns_flat - values_flat
            value_mean = values_flat.mean()
            value_std = values_flat.std(unbiased=False)
            return_mean = returns_flat.mean()
            return_std = returns_flat.std(unbiased=False)
            value_rmse = diff.pow(2).mean().sqrt()
            var_returns = returns_flat.var(unbiased=False)
            diff_var = diff.var(unbiased=False)
            # 计算解释方差：1 - 预测误差方差 / 回报方差，衡量价值函数预测能力
            if var_returns.item() > 1e-8:
                explained_variance = 1.0 - diff_var / var_returns
            else:
                explained_variance = torch.tensor(0.0, device=values_flat.device)
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            self.last_value_stats = {
                "value_mean": value_mean.item(),
                "value_std": value_std.item(),
                "return_mean": return_mean.item(),
                "return_std": return_std.item(),
                "value_rmse": value_rmse.item(),
                "explained_variance": explained_variance.item(),
                "adv_mean": adv_mean.item(),
                "adv_std": adv_std.item(),
            }
        # 优势值标准化：减去均值，除以标准差，稳定训练（防止优势尺度过大或过小影响梯度）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages.copy_(advantages)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        batch_count = 0

        # PPO 更新循环：对存储的轨迹进行多轮多 mini-batch 的随机抽样优化。
        # 这里采用常见的实现：在每个 epoch 随机打乱样本并按 mini-batch 运行 loss、反向传播和优化器步。
        for batch in self.buffer.iter_batches(self.num_mini_batches, self.num_learning_epochs):
            obs_batch = batch.observations
            act_batch = batch.actions
            old_log_probs = batch.old_log_probs
            returns_batch = batch.returns
            old_values = batch.old_values
            adv_batch = batch.advantages

            # ======================= PPO核心计算步骤 =======================
            # 步骤1: 更新策略分布 - 基于当前小批量的观测，计算策略网络的输出分布参数
            #        - 对于高斯策略，计算均值和标准差
            #        - 这一步设置了 actor_critic 内部的分布状态，供后续方法使用
            self.actor_critic.update_distribution(obs_batch)

            # 步骤2: 计算动作的对数概率 - 在更新后的策略分布下，计算小批量中每个动作的对数概率
            #        - 用于重要性采样比: ratio = exp(new_log_prob - old_log_prob)
            #        - old_log_probs 是采样时的概率，log_probs 是当前策略下的概率
            log_probs = self.actor_critic.get_actions_log_prob(act_batch)

            # 步骤3: 计算策略熵 - 策略分布的熵的平均值，用于鼓励探索
            #        - 熵越高表示策略越随机，探索性越强
            #        - 在损失函数中加入负熵项 (-entropy_coef * entropy) 鼓励探索
            entropy = self.actor_critic.entropy.mean()

            # 步骤4: 评估状态价值 - Critic网络输出当前观测的状态价值估计
            #        - 用于计算值损失: (values - returns)^2
            #        - squeeze(-1) 移除可能存在的多余维度
            values = self.actor_critic.evaluate(obs_batch).squeeze(-1)

            # 注意：这里将优势取负（gae_batch = -adv_batch），因为实现上将策略目标以“最小化损失”的方式编码。
            # 原始 PPO 目标是最大化 E[ ratio * advantage ]，等价于最小化 -E[ ratio * advantage ]。
            gae_batch = -adv_batch

            # ======================= PPO策略损失计算 =======================
            # 步骤1: 计算重要性采样比 (importance sampling ratio)
            #        ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ(a|s) - log π_θ_old(a|s))
            #        - 如果 ratio > 1: 当前策略选择该动作的概率高于采样时
            #        - 如果 ratio < 1: 当前策略选择该动作的概率低于采样时
            #        - 如果 ratio ≈ 1: 策略变化不大
            ratio = torch.exp(log_probs - old_log_probs)

            # 步骤2: 计算两个候选损失函数，实现PPO的clipped objective
            #        loss_actor1 = ratio * A  (标准重要性采样损失)
            #        loss_actor2 = clip(ratio, 1-ε, 1+ε) * A  (裁剪后的损失)
            #        其中 A = gae_batch = -advantage (因为我们要最小化损失)
            #        PPO目标: L^CLIP(θ) = E[min(ratio·A, clip(ratio,1-ε,1+ε)·A)]
            loss_actor1 = ratio * gae_batch
            loss_actor2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * gae_batch

            # 步骤3: 取两个损失中的较小值，然后取负得到最终的策略损失
            #        - torch.min(loss_actor1, loss_actor2): 选择较小的损失（更保守的更新）
            #        - .mean(): 对小批量样本求平均
            #        - 负号: 因为gae_batch = -advantage，这里取负相当于最大化优势加权概率
            #        最终: policy_loss = -E[min(ratio·A, clip(ratio,1-ε,1+ε)·A)]
            policy_loss = -torch.min(loss_actor1, loss_actor2).mean()

            # ======================= PPO值损失计算 =======================
            # 步骤1: 计算裁剪后的价值估计，防止critic网络更新过大
            #        values_clipped = old_values + clamp(values - old_values, -ε, +ε)
            #        - old_values: 采样时的价值估计（存储在缓冲区中）
            #        - values: 当前critic网络的预测值
            #        - clamp操作限制更新的幅度在 [-ε, +ε] 范围内
            values_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)

            # 步骤2: 计算未裁剪和裁剪后的值损失
            #        - value_loss_unclipped = (values - returns)^2  (标准MSE损失)
            #        - value_loss_clipped = (values_clipped - returns)^2  (裁剪后的MSE损失)
            #        其中 returns = q_targets 是通过GAE计算得到的Q值目标
            value_loss_unclipped = (values - returns_batch).pow(2)
            value_loss_clipped = (values_clipped - returns_batch).pow(2)

            # 步骤3: 取两个损失中的较大值作为最终值损失
            #        value_loss = 0.5 * max(unclipped_loss, clipped_loss).mean()
            #        - 取max确保当critic预测误差较大时，使用clipped loss限制更新
            #        - 当预测误差较小时，使用unclipped loss进行正常更新
            #        - 0.5系数是MSE损失的常规系数（求导后为1）
            #        这种设计称为"clipped value loss"，防止value网络训练不稳定
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # ======================= 总损失计算与反向传播 =======================
            # 步骤1: 计算总损失，组合策略损失、值损失和熵奖励
            #        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            #        - policy_loss: 策略损失（负的期望优势加权概率）
            #        - value_loss_coef: 值损失系数，控制值网络训练强度（通常为1.0）
            #        - value_loss: 值网络损失，使critic准确预测状态价值
            #        - entropy_coef: 熵系数，控制探索强度（本项目为0.0，不使用熵奖励）
            #        - entropy: 策略熵，鼓励探索（熵越高策略越随机）
            #        注意：entropy_coef * entropy前有负号，所以是 -entropy_coef * entropy
            #              这相当于在损失函数中加入 entropy_coef * entropy 作为奖励项
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # 步骤2: 反向传播与优化
            #        a) 清零梯度: 防止梯度累积
            #        b) 反向传播: 计算损失相对于网络参数的梯度
            #        c) 梯度裁剪: 限制梯度范数，防止梯度爆炸
            #        d) 优化器步: 根据梯度更新网络参数
            self.optimizer.zero_grad()  # 清零所有参数的梯度
            loss.backward()             # 反向传播，计算梯度

            # 梯度裁剪: 将所有参数的梯度范数限制在 max_grad_norm 以内
            #           clip_grad_norm_ 原地修改梯度值
            #           防止梯度爆炸，提高训练稳定性
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

            # 优化器步: 根据梯度更新网络参数（Adam优化器）
            self.optimizer.step()

            # 累加损失用于计算本轮平均损失
            policy_loss_acc += policy_loss.item()
            value_loss_acc += value_loss.item()
            batch_count += 1

        # 计算本轮平均策略损失和值损失
        mean_policy_loss = policy_loss_acc / max(batch_count, 1)
        mean_value_loss = value_loss_acc / max(batch_count, 1)

        # 清空缓冲区，准备下一轮数据收集
        self.buffer.clear()
        return mean_policy_loss, mean_value_loss











