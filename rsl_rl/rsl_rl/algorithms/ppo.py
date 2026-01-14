# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 value_clip_param=None,
                 min_lr=1e-5,
                 max_lr=1e-3,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.max_lr = max_lr

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.value_clip_param = clip_param if value_clip_param is None else value_clip_param

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        if hasattr(self.actor_critic, "action_mean_raw"):
            self.transition.action_mean = self.actor_critic.action_mean_raw.detach()
        else:
            self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    # 处理环境步进后的奖励与终止信息。
    def process_env_step(
        self, rewards, dones, infos  # 传入奖励、终止标记与环境信息字典。
    ):
        # 函数作用: 收集本步奖励/终止, 处理超时 bootstrap, 并写入 rollout 缓冲。
        # 输入 rewards: 环境返回的奖励张量。
        # 输入 dones: 环境返回的终止标记张量。
        # 输入 infos: 环境返回的信息字典, 可能包含 time_outs。
        # 输出: 无, 直接更新内部 transition 与 storage 状态。
        self.transition.rewards = rewards.clone()  # 复制奖励避免后续原地修改影响存储。
        self.transition.dones = dones  # 记录终止标记用于回放与重置。
        # 若存在 time_outs 字段, 对超时终止进行 bootstrap 回报修正。
        if 'time_outs' in infos:
            # 将超时标记映射到设备, 用值函数补全被截断的回报。
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # 记录本次 transition 到 rollout storage。
        self.storage.add_transitions(self.transition)
        # 清理临时 transition, 为下一步采样做准备。
        self.transition.clear()
        # 根据 dones 重置 actor-critic 的内部状态 (如 RNN 隐状态)。
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_approx_kl = 0
        mean_clip_fraction = 0
        mean_entropy = 0
        mean_ratio = 0
        mean_approx_kl_raw = 0
        mean_value_clip_frac = 0
        mean_grad_norm = 0
        max_ratio = 0
        max_logp_diff_raw = 0
        updates_count = 0
        stop_update = False
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            if hasattr(self.actor_critic, "action_mean_raw"):
                mu_batch = self.actor_critic.action_mean_raw
            else:
                mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Surrogate loss
            logp_diff_raw = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            logp_diff = torch.clamp(logp_diff_raw, -20.0, 20.0)
            ratio = torch.exp(logp_diff)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                              1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float())
            approx_kl = 0.5 * torch.mean(logp_diff.pow(2))
            approx_kl_raw = 0.5 * torch.mean(logp_diff_raw.pow(2))

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.value_clip_param,
                                                                                                self.value_clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
                value_clip_frac = torch.mean(
                    (torch.abs(value_batch - target_values_batch) > self.value_clip_param).float()
                )
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                value_clip_frac = torch.zeros(1, device=self.device)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_approx_kl += approx_kl.item()
            mean_clip_fraction += clip_fraction.item()
            mean_entropy += entropy_batch.mean().item()
            mean_ratio += ratio.mean().item()
            mean_approx_kl_raw += approx_kl_raw.item()
            mean_value_clip_frac += value_clip_frac.item()
            mean_grad_norm += float(grad_norm)
            max_ratio = max(max_ratio, ratio.max().item())
            max_logp_diff_raw = max(max_logp_diff_raw, logp_diff_raw.abs().max().item())
            updates_count += 1

        num_updates = max(1, updates_count)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_approx_kl /= num_updates
        mean_clip_fraction /= num_updates
        mean_entropy /= num_updates
        mean_ratio /= num_updates
        mean_approx_kl_raw /= num_updates
        mean_value_clip_frac /= num_updates
        mean_grad_norm /= num_updates
        if self.desired_kl is not None and self.schedule == 'adaptive':
            if mean_approx_kl_raw > self.desired_kl * 2.0:
                self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
            elif mean_approx_kl_raw < self.desired_kl / 2.0 and mean_approx_kl_raw > 0.0:
                self.learning_rate = min(self.max_lr, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        self.storage.clear()

        self.last_stats = {
            "approx_kl_raw": mean_approx_kl_raw,
            "ratio_mean": mean_ratio,
            "ratio_max": max_ratio,
            "logp_diff_raw_max": max_logp_diff_raw,
            "entropy": mean_entropy,
            "grad_norm": mean_grad_norm,
            "value_clip_frac": mean_value_clip_frac,
            "update_steps": updates_count,
            "early_stop": stop_update,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return mean_value_loss, mean_surrogate_loss, mean_approx_kl, mean_clip_fraction
