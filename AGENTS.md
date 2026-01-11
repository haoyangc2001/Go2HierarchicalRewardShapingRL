# Repository Guidelines


## Project Overview

MCRA_RL is a hierarchical reinforcement learning system for Unitree Go2 quadruped robot navigation. The project implements a Reach-Avoid PPO algorithm for obstacle avoidance navigation in complex environments. The system combines pre-trained low-level locomotion policies with trainable high-level navigation policies.

## Architecture

### Hierarchical RL Structure
- **Low-level**: Pre-trained locomotion policy (velocity → joint actions) in `legged_gym/envs/go2/go2_env.py`
- **High-level**: Trainable navigation policy (observations → velocity commands) in `legged_gym/envs/go2/high_level_navigation_env.py`
- **Hierarchical wrapper**: `legged_gym/envs/go2/hierarchical_go2_env.py` connects both levels
- **Action repeat**: High-level actions repeated at low-level via `high_level_action_repeat` parameter

### Reach-Avoid PPO Algorithm
- **Custom implementation**: `rsl_rl/algorithms/reach_avoid_ppo.py`
- **Dual objectives**: `g_values` for reaching target, `h_values` for avoiding obstacles
- **Custom buffer**: `ReachAvoidBuffer` stores both objective values
- **Safety constraints**: Terminates episodes on safety violations (`h_values >= 0`)

### Key Components
1. **Environments**: `legged_gym/envs/go2/` - Go2 robot environment implementations
2. **RL Algorithms**: `rsl_rl/algorithms/` - Reinforcement learning algorithms
3. **Training Scripts**: `legged_gym/scripts/` - Training, testing, and visualization
4. **Deployment**: `legged_gym_go2/deploy/` - Simulation and real robot deployment
5. **Configuration**: YAML configs in `deploy/*/configs/` and Python config classes

## Algorithm Design

Detailed Reach-Avoid PPO design (value targets, losses, entropy regularization) is documented in `REACH_AVOID_PPO_DESIGN.md` and should be kept in sync with this section.

### 1. Reach-Avoid Problem Formulation
The Reach-Avoid task requires the agent to reach a target region while avoiding obstacles:
- **Reach objective**: Terminally reach target region \( \mathcal{G} \)
- **Avoid constraint**: Always stay outside obstacle region \( \mathcal{O} \)
- **Mathematical form**: Find \( \pi \) s.t. \( x_t \notin \mathcal{O} \ \forall t < T, \ x_T \in \mathcal{G} \)

#### Dual Objective Functions
Two auxiliary functions quantify task progress:

```python
# g_values: Reach objective function
# Negative values indicate inside target region
g(x) = {
    g_target_value,                     if dist(x, G) ≤ target_radius
    g_distance_scale × dist(x, G),     otherwise
}

# h_values: Safety constraint function
# Negative values indicate safe, non-negative indicates constraint violation
h(x) = {
    h_safe_value,    if dist(x, O) > unsafe_radius
    h_unsafe_value,  if dist(x, O) ≤ unsafe_radius
}
```

**Default parameters** (`high_level_navigation_env.py:332-337`):
- `g_target_value = -300.0`    # g-value inside target region
- `g_distance_scale = 100.0`   # Distance scaling factor
- `h_safe_value = -300.0`      # h-value in safe region
- `h_unsafe_value = 300.0`     # h-value in unsafe region

### 2. Reach-Avoid PPO Core Innovation

#### 2.1 Standard PPO Limitation
Traditional PPO optimizes expected return:
\[
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
\]
But cannot guarantee **safety constraints**.

#### 2.2 Reach-Avoid Value Function Design
Introduces **dual value functions** to optimize both safety and task completion:

**Core formula** (`reach_avoid_ppo.py:116-119`):
\[
Q_t = \max\left( h_t, \min\left( g_t, \gamma \cdot V_{t+1} \right) \right)
\]

**Physical meaning**:
1. **Inner min**: Takes the smaller between reach cost \( g_t \) and future discounted value \( \gamma V_{t+1} \)
   - If currently close to target (small \( g_t \)), prioritize current reward
   - Otherwise consider potentially better future states
2. **Outer max**: Takes the larger between safety constraint \( h_t \) and inner result
   - **Safety-first principle**: When \( h_t \geq 0 \) (safety violation), \( Q_t = h_t \geq 0 \)
   - Ensures value function reflects safety constraint priority

**Implementation note**: The code forms multiple \(Q_t^{(k)}\) candidates via a value table and computes the value target \(\hat{Q}_t\) as a GAE-weighted average; advantages use \(A_t=\hat{Q}_t-V_t\). See `REACH_AVOID_PPO_DESIGN.md` for the full target and loss definitions.

#### 2.3 Generalized Advantage Estimation (GAE) Extension
**Reach-Avoid GAE formula** (`_calculate_reach_gae` function):

```python
# Key steps (reach_avoid_ppo.py:84-146)
for idx in range(horizon - 1, -1, -1):
    # 1. Update GAE coefficients considering termination states
    gae_coeffs = (rolled * lam * (1.0 - prev_done)
                 + rolled * lam_ratio * prev_done) * (1.0 - done_row)

    # 2. Compute Reach-Avoid Q-values
    vhs_row = torch.maximum(
        h_seq[idx],                    # Safety constraint
        torch.minimum(g_seq[idx], gamma * value_table)  # Min of target and future values
    )

    # 3. Compute Q-value targets: weighted average of future vhs_row
    q_targets[idx] = (vhs_row * norm_coeffs).sum(dim=0)
```

**Termination conditions**:
- **Environment termination**: `done_seq` marks episode end
- **Safety violation**: Force termination when `h_values >= 0`
- **Combined termination**: `done_seq = env_dones OR (h_values >= 0)`

### 3. Hierarchical Learning Architecture

#### 3.1 Two-Level Policy Structure
```
High-level policy π_high (trainable)
    ↓ Output: [vx, vy, vyaw] ∈ [-1, 1]³
    ↓ Scaling: × [0.6, 0.2, 0.8] → actual velocity commands
Low-level policy π_low (pre-trained fixed)
    ↓ Output: 12-dimensional joint torques
    ↓ Execution: Go2 robot locomotion
```

#### 3.2 Action Repeat Mechanism
```python
# hierarchical_go2_env.py:148-159
for _ in range(self.low_level_action_repeat):
    self.base_env.commands[:, :3] = desired_velocity_commands
    current_base_obs = self.base_env.get_observations()
    low_level_actions = self.low_level_policy(current_base_obs)
    # Execute low-level policy for one step
```

**Design considerations**:
- High-level decisions at lower frequency (computational efficiency)
- Low-level execution at high frequency (motion stability)
- `high_level_action_repeat` controls temporal scale separation

### 4. Observation Space Design

#### 4.1 High-Level Observation Composition (`high_level_navigation_env.py:42-49`)
```
Observation dimension = 8 (base features) + target_lidar_bins + obstacle_lidar_bins

Base features (8-dimensional):
1. cos(heading)                     # Heading cosine
2. sin(heading)                     # Heading sine
3. body_vx × 2.0                    # Body X-velocity (scaled)
4. body_vy × 2.0                    # Body Y-velocity
5. yaw_rate × 0.25                  # Yaw angular velocity
6. dist_to_target_normalized        # Normalized target distance
7. target_dir_body_x                # Target direction in body X-axis
8. target_dir_body_y                # Target direction in body Y-axis

Lidar features:
- target_lidar: "Soft" encoding of target direction (optional)
- obstacle_lidar: Sector encoding of obstacle distances
```

#### 4.2 Manual Lidar Implementation
**Innovation**: Instead of using real sensors, compute **simulated lidar features** based on known obstacle positions, reducing simulation computational cost.

```python
# high_level_navigation_env.py:201-228
if self.use_manual_lidar and self.lidar_num_bins > 0:
    # 1. Compute obstacle positions in body coordinate system
    rel_x_body = rel_xy[:, :, 0] * heading_cos + rel_xy[:, :, 1] * heading_sin
    rel_y_body = -rel_xy[:, :, 0] * heading_sin + rel_xy[:, :, 1] * heading_cos

    # 2. Compute distances and angles
    planar_dist = sqrt(rel_x_body² + rel_y_body²)
    surface_dist = max(planar_dist - obstacle_radius, 0)
    intensity = 1.0 - clamp(surface_dist / lidar_max_range, 0, 1)

    # 3. Angular bin encoding
    angles = atan2(rel_y_body, rel_x_body)
    bin_indices = floor((angles + π) / bin_size)
```

### 5. Training Algorithm Flow

#### 5.1 Overall Training Loop (`train_reach_avoid.py:222-338`)
```python
Algorithm 1: Reach-Avoid PPO Training Procedure
Input: Environment env, Algorithm alg, Maximum iterations max_iterations
Output: Trained policy

1: obs, g_vals, h_vals ← env.reset()
2: for iteration = 1 to max_iterations do
3:   # Data collection phase (horizon steps)
4:   for step = 1 to horizon do
5:     actions, log_probs, values ← alg.act(obs)
6:     next_obs, next_g, next_h, dones, _ ← env.step(actions)
7:     Store to rollout buffer
8:     obs, g_vals, h_vals ← next_obs, next_g, next_h
9:   end for
10:
11:  # Force horizon truncation as termination
12:  rollout_dones[-1] ← True
13:
14:  # Advantage computation phase
15:  last_values ← alg.actor_critic.evaluate(obs)
16:  alg.buffer.compute_advantages(last_values, γ, λ)
17:
18:  # Success rate computation
19:  success_rate ← compute_reach_avoid_success_rate(g_seq, h_seq)
20:
21:  # Policy update phase
22:  policy_loss, value_loss ← alg.update()
23:
24:  # Environment reset
25:  if iteration < max_iterations then
26:    obs, g_vals, h_vals ← env.reset()
27:  end if
28: end for
```

#### 5.2 Success Rate Computation Algorithm
```python
def compute_reach_avoid_success_rate(g_sequence, h_sequence):
    # 1. Check if target reached (g < 0)
    g_negative = g_sequence < 0
    has_success = g_negative.any(dim=0)  # Reached target at any timestep

    # 2. Find first reaching time
    first_success = argmax(g_negative.long(), dim=0)

    # 3. Check safety before reaching (h < 0)
    before_success = time_index < first_success
    h_violation = (h_sequence >= 0) & before_success
    safe_before = ~h_violation.any(dim=0)

    # 4. Success condition: Reached target AND safe before reaching
    success = has_success & safe_before

    return success.float().mean()  # Success rate
```

**Key metrics**:
- **Success rate**: Proportion of environments satisfying both Reach and Avoid conditions
- **Execution cost**: Average timesteps required for successful environments to reach target

### 6. Safety Constraint Handling

#### 6.1 Hard vs Soft Constraints
| Constraint Type | Implementation | Advantages | Disadvantages |
|----------------|---------------|------------|---------------|
| **Hard constraint** | Terminate episode when `h_values >= 0` | Absolute safety guarantee | Sparse training data |
| **Soft constraint** | Penalize in reward function | Stable training | May violate constraints |

#### 6.2 Hybrid Approach in This Project
```python
# 1. During training: Soft constraint + hard termination
h_values = {
    h_safe_value = -300,     # Large negative value for safe region
    h_unsafe_value = 300,    # Large positive value for unsafe region
}

# 2. Force termination on safety violation
safety_dones = h_values[:-1] >= 0
done_seq = torch.logical_or(env_dones, safety_dones)  # reach_avoid_ppo.py:259
```

**Design trade-off**:
- Large numerical difference (-300 vs 300) provides clear learning signal
- Hard termination prevents learning dangerous behaviors
- GAE computation handles safety termination, correctly propagating value functions


## Development Workflow

### Common Commands

#### Training
```bash
# Basic training (headless mode)
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --headless=true --num_envs=32

# Resume training from checkpoint
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --resume=true --experiment_name=high_level_go2

# Training with visualization
python legged_gym_go2/legged_gym/scripts/train_reach_avoid.py --headless=false
```

#### Evaluation and Visualization
```bash
# Visualize trained policy
python legged_gym_go2/legged_gym/scripts/play_reach_avoid.py --model_path=logs/high_level_go2/.../model_1000.pt

# Test policy and generate trajectories
python legged_gym_go2/legged_gym/scripts/test_reach_avoid.py --checkpoint-path=model.pt --output=trajectories.json

# Plot environment layout with trajectories
python legged_gym_go2/legged_gym/scripts/plot_env_layout.py --traj-file=traj_run.json
```

#### Deployment
```bash
# MuJoCo deployment
python legged_gym_go2/deploy/deploy_mujoco/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml

# Real robot deployment
python legged_gym_go2/deploy/deploy_real/deploy.py --checkpoint=model.pt --cfg=configs/go2.yaml
```

### Key Parameters
- `--headless`: Run without visualization (faster training)
- `--num_envs`: Number of parallel environments (default: 32)
- `--experiment_name`: Name for log directory in `logs/`
- `--resume`: Resume training from latest checkpoint
- `--checkpoint`: Path to model checkpoint file (.pt)

## Code Organization Patterns

### File Naming Conventions
- **Environment files**: `*_env.py` (e.g., `hierarchical_go2_env.py`)
- **Configuration files**: `*_config.py` and `*.yaml`
- **Training scripts**: `train_*.py`
- **Evaluation scripts**: `play_*.py`, `test_*.py`
- **Deployment scripts**: `deploy.py` in `deploy/` subdirectories

### Configuration System
- **Python config classes**: Define default parameters in `*_config.py` files
- **YAML configs**: Override defaults for specific deployments
- **Command-line args**: Further override via `get_args()` utility
- **Hierarchy**: YAML → Python config class → command-line args

### Logging and Checkpoints
- **Log directory**: `logs/{experiment_name}/{timestamp}/`
- **Checkpoints**: Saved as `model_{iteration}.pt`
- **TensorBoard logs**: Automatically generated in log directory
- **Trajectory files**: JSON format for analysis (`traj_run.json`)

## Key Dependencies

### Core Dependencies (from setup.py)
- `isaacgym`: NVIDIA physics simulator (proprietary)
- `torch>=1.4.0`: PyTorch for neural networks
- `numpy==1.20`: Numerical computations
- `mujoco==3.2.3`: Physics engine for deployment
- `pyyaml`: Configuration file parsing
- `tensorboard`: Training visualization

### Python Version
- **Primary**: Python 3.7 (from conda environment)
- **Compatibility**: Python >=3.6 (rsl_rl requirement)

## Important Paths

### Absolute Paths
- Main training script: `legged_gym_go2/legged_gym/scripts/train_reach_avoid.py`
- Custom RL algorithm: `rsl_rl/algorithms/reach_avoid_ppo.py`
- Hierarchical environment: `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`
- Log directory: `logs/high_level_go2/` (example)

### Relative Paths (from repository root)
- `legged_gym_go2/`: Main RL environment and training code
- `rsl_rl/`: Reinforcement learning algorithms library
- `unitree_rl_gym/`: Unitree robot RL base library
- `isaacgym/`: NVIDIA Isaac Gym physics simulator

## Development Notes

### Testing Approach
- No formal unit test suite
- Integration testing through training scripts
- Visual validation with `play_*` scripts
- Trajectory analysis via JSON output

### Git Practices
- Logs and checkpoints excluded via `.gitignore`
- Model files (.pt) should be managed separately
- Configuration changes tracked in YAML files

### Performance Considerations
- **GPU required**: Isaac Gym requires NVIDIA GPU
- **Headless mode**: Use `--headless=true` for faster training
- **Parallel environments**: Increase `--num_envs` for better GPU utilization
- **Memory**: Large models may require GPU memory management

## Troubleshooting

### Common Issues
1. **Isaac Gym not found**: Ensure Isaac Gym is downloaded and installed separately
2. **CUDA errors**: Verify CUDA version matches Isaac Gym requirements
3. **Import errors**: Install all packages with `-e` (editable) flag
4. **Memory issues**: Reduce `--num_envs` or use headless mode

### Debug Commands
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify package installations
python -c "import isaacgym; import rsl_rl; import legged_gym"

# Test basic environment
python legged_gym_go2/legged_gym/scripts/play_fixed_commands_with_video.py
```

### notice
执行所有脚本之前，必须先使用 conda activate unitree-rl 激活  unitree-rl 环境，之后使用 python 执行脚本。
