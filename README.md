# Go2 Hierarchical Reach-Avoid RL

## 📋 项目概述

Go2 Hierarchical Reach-Avoid RL 是一个基于强化学习的机器人导航系统，专为 Unitree Go2 机器人设计。该项目实现了一个分层控制架构，结合了预训练的低层级运动策略和可训练的高层级导航策略，使机器人能够在复杂环境中安全导航并到达目标位置。

## 🔧 核心功能

### 🎯 1. 分层控制架构

#### 低层级（Locomotion）
- **功能**：将速度指令（线速度和角速度）转换为机器人的关节动作
- **实现**：预训练的运动控制策略，基于 PPO 算法训练
- **输入**：机器人状态（关节角度、速度、IMU 数据等）和速度指令
- **输出**：关节目标位置或力矩
- **文件位置**：
  - 环境封装：`legged_gym/envs/go2/go2_env.py`
  - 低层级策略加载：`legged_gym/envs/go2/hierarchical_go2_env.py` 中的 `_load_low_level_policy` 方法

#### 高层级（Navigation）
- **功能**：将环境观测转换为速度指令，实现避障导航
- **实现**：可训练的导航策略，基于 Reach-Avoid PPO 算法
- **输入**：环境观测（机器人位置、障碍物信息、目标位置、速度等）
- **输出**：速度指令（线速度和角速度）
- **文件位置**：
  - 导航环境：`legged_gym/envs/go2/high_level_navigation_env.py`
  - 高层级配置：`legged_gym/envs/go2/high_level_navigation_env.py` 中的 `HighLevelNavigationConfig` 类

#### 层级连接机制
- **动作重复机制**：高层级输出的速度指令在低层级重复执行多次
- **参数配置**：通过 `high_level_action_repeat` 参数控制，默认为 1
- **文件位置**：`legged_gym/envs/go2/hierarchical_go2_env.py` 中的 `low_level_action_repeat` 属性

#### 分层环境封装
- **统一接口**：`HierarchicalVecEnv` 类封装了分层环境，提供标准的 RL 环境接口
- **文件位置**：`legged_gym/scripts/train_reach_avoid.py` 中的 `HierarchicalVecEnv` 类
- **核心方法**：`reset()`、`step()`、`close()`

### 🧠 2. Reach-Avoid 强化学习算法

Reach-Avoid 强化学习算法是专为避障导航任务设计的 PPO（Proximal Policy Optimization）扩展版本，实现了到达目标位置（reach）和避免障碍物（avoid）的双重目标。

#### 2.1 核心问题定义

- **Reach 目标**：机器人需要到达指定目标位置，由 `g_values` 表示（负值表示成功到达）
- **Avoid 约束**：机器人必须避免与障碍物碰撞，由 `h_values` 表示（非负值表示碰撞）
- **状态空间**：机器人位置、速度、障碍物信息、目标位置等环境观测
- **动作空间**：速度指令（线速度和角速度）

#### 2.2 算法核心组件

##### ReachAvoidPPO 类
- **功能**：实现完整的 Reach-Avoid PPO 算法
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py`
- **核心方法**：
  - `act()`：根据观测生成动作
  - `update()`：更新策略网络
  - `init_storage()`：初始化经验缓冲区

##### 自定义优势函数计算
- **功能**：计算适用于 Reach-Avoid 任务的广义优势估计（GAE）
- **实现**：`_calculate_reach_gae` 函数
- **特点**：
  - 考虑了 reach 和 avoid 双重目标
  - 基于 JAX 参考实现的 PyTorch 移植
  - 支持多环境并行计算
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py` 中的 `_calculate_reach_gae` 函数

##### 经验回放缓冲区
- **功能**：存储和管理训练数据
- **实现**：`ReachAvoidBuffer` 类
- **特点**：
  - 存储 `g_values` 和 `h_values` 用于避障任务
  - 支持批量采样和多轮训练
  - 实现了自定义的优势函数计算
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py` 中的 `ReachAvoidBuffer` 类

##### 数据批次处理
- **功能**：将经验数据转换为训练批次
- **实现**：`ReachAvoidBatch` 数据类
- **特点**：
  - 封装了训练所需的所有数据
  - 支持扁平化数据视图
  - 便于批量采样和训练
- **文件位置**：`rsl_rl/algorithms/reach_avoid_ppo.py` 中的 `ReachAvoidBatch` 类

#### 2.3 算法更新流程

1. **经验收集**：通过与环境交互收集轨迹数据
2. **完成标志计算**：
   - 环境完成：`env_dones = self.dones`
   - 安全违规：`safety_dones = self.h_values[:-1] >= 0`
   - 最终完成标志：`done_seq = torch.logical_or(env_dones, safety_dones)`
3. **优势函数计算**：调用 `_calculate_reach_gae` 计算优势和目标值
4. **优势归一化**：对优势函数进行归一化处理
5. **策略更新**：
   - 计算新的动作概率和价值估计
   - 计算策略损失（裁剪目标）
   - 计算价值损失（裁剪目标）
   - 计算熵正则化项
   - 总损失：`loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy`
   - 反向传播和梯度裁剪
   - 优化器更新

#### 2.4 关键参数配置

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `learning_rate` | 学习率 | 3e-4 |
| `gamma` | 折扣因子 | 0.999 |
| `lam` | GAE 参数 | 0.95 |
| `num_learning_epochs` | 每次迭代的训练轮数 | 4 |
| `num_mini_batches` | 每次迭代的 mini-batch 数量 | 4 |
| `clip_param` | PPO 裁剪参数 | 0.2 |
| `value_loss_coef` | 价值损失权重 | 1.0 |
| `entropy_coef` | 熵正则化权重 | 0.0 |
| `max_grad_norm` | 梯度裁剪阈值 | 1.0 |

#### 2.5 成功指标计算

- **功能**：评估机器人在轨迹中的避障导航成功率
- **实现**：`compute_reach_avoid_success_rate` 函数
- **计算逻辑**：
  1. 检查是否成功到达目标（`g_values < 0`）
  2. 记录首次成功到达的时间步
  3. 检查在成功前是否发生碰撞（`h_values >= 0`）
  4. 成功率 = （成功到达目标且未碰撞的环境数）/ 总环境数
- **文件位置**：`legged_gym/scripts/train_reach_avoid.py` 中的 `compute_reach_avoid_success_rate` 函数

#### 2.6 与标准 PPO 的区别

1. **目标函数**：
   - 标准 PPO：单一奖励信号
   - ReachAvoidPPO：双重目标（reach + avoid）

2. **优势函数**：
   - 标准 PPO：基于奖励信号的 GAE
   - ReachAvoidPPO：基于 `g_values` 和 `h_values` 的自定义 GAE

3. **完成标志**：
   - 标准 PPO：仅环境完成
   - ReachAvoidPPO：环境完成或安全违规

4. **经验缓冲区**：
   - 标准 PPO：存储奖励和完成标志
   - ReachAvoidPPO：额外存储 `g_values` 和 `h_values`

5. **训练流程**：
   - 标准 PPO：基于奖励的策略更新
   - ReachAvoidPPO：考虑双重目标的策略更新

### 🌍 3. 环境与仿真

- 基于 Isaac Gym 物理引擎的高性能仿真环境
- 支持多环境并行训练
- 提供丰富的环境观测和奖励信号

### 📋 4. 训练流程

执行 `train_reach_avoid.py` 后，项目将启动完整的训练流程：

1. **环境初始化**：创建 Go2 机器人仿真环境
2. **低层级策略加载**：加载预训练的运动控制策略
3. **高层级策略初始化**：初始化导航策略网络
4. **经验收集**：通过与环境交互收集轨迹数据
5. **策略更新**：使用 PPO 算法更新高层级导航策略
6. **性能评估**：计算并监控避障任务成功率
7. **模型保存**：定期保存训练好的模型检查点

## 📐 技术架构

### 🧩 主要组件

| 组件 | 描述 | 位置 |
|------|------|------|
| HierarchicalGO2Env | 分层环境封装 | `legged_gym/envs/go2/hierarchical_go2_env.py` |
| ReachAvoidPPO | 避障强化学习算法 | `rsl_rl/algorithms/reach_avoid_ppo.py` |
| ActorCritic | 策略网络架构 | `rsl_rl/modules/actor_critic.py` |
| HighLevelNavigationEnv | 高层级导航封装 | `legged_gym/envs/go2/high_level_navigation_env.py` |

### 🏗️ 网络架构

- **Actor 网络**：4 层全连接网络，每层 512 个单元
- **Critic 网络**：4 层全连接网络，每层 512 个单元
- 激活函数：ReLU
- 初始化噪声：标准差为 0.1 的高斯噪声

## 📦 环境配置

### 📥 安装步骤

#### 系统要求
- **操作系统**：推荐 Ubuntu 18.04 或更高版本
- **GPU**：NVIDIA GPU
- **驱动版本**：推荐 525 或更高版本

#### 详细安装步骤

1. **克隆项目仓库**
   ```bash
   git clone https://github.com/haoyangc2001/Go2HierarchicalReachAvoidRL.git
   cd Go2HierarchicalReachAvoidRL
   ```

2. **创建虚拟环境（推荐使用 Conda）**
   
   建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 2.1。

   #### 2.1 下载并安装 MiniConda
   MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：
   ```bash
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```

   安装完成后，初始化 Conda：
   ```bash
   ~/miniconda3/bin/conda init --all
   source ~/.bashrc
   ```

   #### 2.2 创建新环境
   ```bash
   conda create -n unitree-rl python=3.8
   ```

   #### 2.3 激活虚拟环境
   ```bash
   conda activate unitree-rl
   ```

3. **安装依赖项**
   
   #### 3.1 安装 PyTorch
   PyTorch 是一个神经网络计算框架，用于模型训练和推理。使用以下命令安装：
   ```bash
   conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
   
   #### 3.2 安装 Isaac Gym
   Isaac Gym 是 Nvidia 提供的刚体仿真和训练框架。

   ##### 3.2.1 下载
   从 Nvidia 官网下载 [Isaac Gym](https://developer.nvidia.com/isaac-gym)。

   ##### 3.2.2 安装
   解压后进入 `isaacgym/python` 文件夹，执行以下命令安装：
   ```bash
   cd isaacgym/python
   pip install -e .
   ```

   ##### 3.2.3 验证安装
   运行以下命令，若弹出窗口并显示 1080 个球下落，则安装成功：
   ```bash
   cd examples
   python 1080_balls_of_solitude.py
   ```

   如有问题，可参考 `isaacgym/docs/index.html` 中的官方文档。

   #### 3.3 安装 rsl_rl
   `rsl_rl` 是一个强化学习算法库。

   ##### 3.3.1 下载
   通过 Git 克隆仓库：
   ```bash
   cd ..
   cd ..
   git clone https://github.com/leggedrobotics/rsl_rl.git
   ```

   ##### 3.3.2 切换分支
   切换到 v1.0.2 分支：
   ```bash
   cd rsl_rl
   git checkout v1.0.2
   ```

   ##### 3.3.3 安装
   ```bash
   pip install -e .
   cd ..
   ```

   #### 3.4 安装项目依赖
   ```bash
   cd legged_gym_go2
   pip install -e .
   
   # 安装其他依赖
   pip install numpy matplotlib torchvision
   ```


## 🛠️ 使用说明

### 📜 脚本说明

`legged_gym_go2/legged_gym/scripts/` 目录下包含以下核心脚本：

#### 1. 训练脚本

##### `train_reach_avoid.py`
- **功能**：实现分层避障任务的完整训练流程，结合预训练的低层级运动策略和可训练的高层级导航策略
- **算法**：基于 Reach-Avoid PPO 算法
- **使用命令**：
  ```bash
  python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py
  ```
- **主要参数**：
  - `--headless`：是否在无头模式下运行（默认：True）
  - `--resume`：是否从 checkpoint 恢复训练
  - `--experiment_name`：实验名称
  - `--num_envs`：并行训练环境数量

#### 2. 可视化与测试脚本

##### `play_reach_avoid.py`
- **功能**：可视化训练后的分层避障策略效果，展示机器人在复杂环境中导航避障的能力
- **使用命令**：
  ```bash
  python legged_gym_go2/legged_gym/scripts/play_reach_avoid.py
  ```
- **主要参数**：
  - `--model_path`：指定模型路径
  - `--headless`：是否在无头模式下运行

##### `play_fixed_commands_with_video.py`
- **功能**：使用固定命令序列测试机器人的运动控制能力，并支持视频录制
- **测试命令**：包含停止、前进、右移、右转等预设命令序列
- **使用命令**：
  ```bash
  python legged_gym_go2/legged_gym/scripts/play_fixed_commands_with_video.py --task=go2
  ```
- **主要参数**：
  - `--task`：机器人类型（固定为 go2）
  - `--headless`：是否在无头模式下运行

##### `plot_env_layout.py`
- **功能**：可视化 GO2 高层级环境布局（目标+障碍物），并可叠加测试轨迹
- **使用命令**：
  ```bash
  python legged_gym_go2/legged_gym/scripts/plot_env_layout.py
  ```
- **主要参数**：
  - `--traj-file`：可选参数，指定由 test_reach_avoid.py 生成的 JSON 轨迹文件
  - `--save`：可选参数，指定图像保存路径
  - `--no-show`：跳过交互式窗口（用于保存到磁盘时）

##### `test_reach_avoid.py`
- **功能**：加载训练好的高层级reach-avoid策略，生成多个随机轨迹，并将XY路径保存到JSON文件中，供后续使用plot_env_layout.py可视化
- **使用命令**：
  ```bash
  python legged_gym_go2/legged_gym/scripts/test_reach_avoid.py --checkpoint-path <model_path> --output traj_run.json
  ```
- **主要参数**：
  - `--checkpoint-path`：必须参数，指定训练好的高层级checkpoint路径（.pt文件）
  - `--num-trajs`：要记录的轨迹数量（默认：10）
  - `--max-steps`：每个轨迹的最大高层级步数（默认：500）
  - `--output`：存储XY路径的JSON文件名称（默认：reach_avoid_rollouts.json）
  - `--low-level-model`：可选的预训练低层级运动策略覆盖
  - `--num-envs`：用于rollouts的并行环境数量（默认：1）
  - `--render`：在rollouts期间打开Isaac Gym查看器
  - `--max-reset-attempts`：环境重置尝试的安全上限（默认：20）

### ⚙️ 命令行参数

所有脚本支持的通用参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| --headless | 是否在无头模式下运行 | True |
| --rl_device | RL 计算设备 | cuda:1 |
| --sim_device | 仿真设备 | cuda:1 |
| --compute_device_id | 计算设备 ID | 1 |
| --sim_device_id | 仿真设备 ID | 1 |
| --task | 任务名称（固定为 go2） | go2 |
| --resume | 是否从 checkpoint 恢复训练 | False |
| --experiment_name | 实验名称 | high_level_go2 |
| --run_name | 运行名称 | 自动生成时间戳 |
| --num_envs | 并行训练环境数量 | 32 |
| --seed | 随机种子 | 42 |
| --max_iterations | 最大训练迭代次数 | 10000 |

### 📝 示例命令

#### 基本训练
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py
```

#### 无头模式训练（更高效率）
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --headless=true
```

#### 使用不同 GPU
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --rl_device=cuda:0 --sim_device=cuda:0 --compute_device_id=0 --sim_device_id=0
```

#### 恢复训练
```bash
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --resume=true --experiment_name=high_level_go2
```

#### 可视化训练效果
```bash
python legged_gym_go2/legged_gym/scripts/play_reach_avoid.py
```

### 📄 配置文件

- 环境配置：`GO2HighLevelCfg`
- 训练配置：`GO2HighLevelCfgPPO`
- 可通过命令行参数覆盖配置值

## 📊 训练结果

### 📤 输出信息

训练过程中，脚本会输出以下关键信息：

```
iter 00001 | success 0.000 | policy_loss -0.00123 | value_loss 0.12345 | Vmean 0.567 | Rmean 0.890 | Vrmse 0.123 | VexpVar 0.456 | adv_std 0.789 | elapsed 1.23s
```

### 💾 生成文件

- **模型检查点**：`logs/<experiment_name>/<timestamp>/model_<iteration>.pt`
- **训练日志**：控制台输出，可重定向到文件
- **GH 快照**：（可选）定期保存的状态快照

## 📁 项目结构

```
Go2HierarchicalReachAvoidRL/
├── legged_gym_go2/
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   └── go2/
│   │   │       ├── hierarchical_go2_env.py    # 分层环境封装
│   │   │       ├── high_level_navigation_env.py # 高层级导航环境
│   │   │       └── go2_env.py                   # 基础 Go2 环境
│   │   ├── scripts/
│   │   │   ├── train_reach_avoid.py            # 分层避障任务训练脚本
│   │   │   ├── play_reach_avoid.py             # 避障策略可视化脚本
│   │   │   ├── play_fixed_commands_with_video.py # 固定命令测试与视频录制脚本
│   │   │   ├── plot_env_layout.py              # 环境布局可视化脚本
│   │   │   └── test_reach_avoid.py             # 生成轨迹并保存到JSON文件的测试脚本
│   │   └── utils/
│   └── rsl_rl/
│       ├── algorithms/
│       │   └── reach_avoid_ppo.py              # 避障 PPO 算法
│       └── modules/
│           └── actor_critic.py                 # 策略网络
└── logs/                                       # 训练日志和模型保存目录
```

## 🔄 训练流程详解

### 🟢 1. 环境创建

`create_env` 函数创建分层环境：

```python
def create_env(env_cfg, train_cfg, args, device) -> HierarchicalVecEnv:
    base_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )
    return HierarchicalVecEnv(base_env)
```

### 📊 2. 经验收集

训练循环中，脚本收集固定长度的轨迹数据：

```python
for step in range(horizon):
    actions, log_probs, values = alg.act(rollout_obs[step])
    next_obs, next_g, next_h, dones, _ = env.step(actions)
    # 存储轨迹数据
```

### ✅ 3. 成功率计算

`compute_reach_avoid_success_rate` 函数评估避障任务成功率：

```python
def compute_reach_avoid_success_rate(g_sequence, h_sequence) -> float:
    # g_sequence: 目标达成指标序列
    # h_sequence: 障碍物规避指标序列
    # 返回成功率
```

### � 4. 策略更新

使用 PPO 算法更新高层级策略：

```python
policy_loss, value_loss = alg.update()
```

## 🚀 扩展与自定义

### 🏗️ 1. 修改网络架构

在 `train_reach_avoid` 函数中修改网络尺寸：

```python
train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]  # Actor 网络
 train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]  # Critic 网络
```

### ⚙️ 2. 调整训练参数

修改 `GO2HighLevelCfgPPO` 配置：

- 学习率
- 折扣因子
- gae 参数
- 批量大小
- 训练迭代次数

### 🌍 3. 自定义环境

扩展 `HierarchicalGO2Env` 类或修改 `HighLevelNavigationConfig` 配置，自定义环境特性：

- 障碍物类型和分布
- 奖励函数
- 观测空间
- 动作空间

## 📈 性能指标

训练过程中监控的关键指标：

- **成功率（success）**：成功到达目标且未碰撞的环境比例
- **策略损失（policy_loss）**：Actor 网络的损失值
- **价值损失（value_loss）**：Critic 网络的损失值
- **价值均值（Vmean）**：价值函数的均值
- **回报均值（Rmean）**：轨迹回报的均值
- **价值 RMSE（Vrmse）**：价值函数预测误差
- **解释方差（VexpVar）**：价值函数解释回报变化的能力
- **优势标准差（adv_std）**：优势函数的标准差

## 🎯 示例应用场景

1. **室内导航**：在办公室或家庭环境中自主导航
2. **仓储物流**：在仓库环境中搬运物品
3. **搜索救援**：在复杂环境中搜索目标
4. **环境监测**：在特定区域内进行环境监测

## 🎉 致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：
- [https://github.com/littlebearqqq/legged_gym_go2.git](https://github.com/littlebearqqq/legged_gym_go2.git): 构建训练与运行代码的基础。
- rsl_rl: 强化学习算法实现。
- mujoco: 提供强大仿真功能。
- unitree_sdk2_python: 实物部署硬件通信接口。

