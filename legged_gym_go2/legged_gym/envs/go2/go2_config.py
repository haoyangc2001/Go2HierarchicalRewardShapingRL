from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env:
        # num_envs = 4096
        num_envs = 2048
        # num_envs = 2
        num_observations = 45
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0

            'FL_hip_joint': 0.1,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]


            'FR_hip_joint': -0.1 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]



            'RL_hip_joint': 0.1,   # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]


            'RR_hip_joint': -0.1,   # [rad]
            'RR_thigh_joint': 1.,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]



        }

    # 定义target和unsafe sphere
    class rewards_ext:
        # 多障碍物配置 - 从 point_goal_avoid.py 等比例放大 (缩放比例: 6/1.8 ≈ 3.33)
        # 原始环境: boundary=1.8m, 障碍物半径=0.2m, 目标半径=0.3m
        # 缩放后环境: boundary=6m, 障碍物半径≈0.67m, 目标半径≈1.0m

        # 8个障碍物位置 (从原始位置 × 3.33 缩放)
        unsafe_spheres_pos = [
            [-2.66, -3.66, 0.4],  # 原: [-0.8, -1.1]
            [ 4.00, -3.00, 0.4],  # 原: [1.2, -0.9]
            [-3.66,  2.33, 0.4],  # 原: [-1.1, 0.7]
            [ 3.00,  4.33, 0.4],  # 原: [0.9, 1.3]
            [ 0.33, -4.00, 0.4],  # 原: [0.1, -1.2]
            [-4.33, -0.33, 0.4],  # 原: [-1.3, -0.1]
            [ 2.66,  0.33, 0.4],  # 原: [0.8, 0.1]
            [-0.66,  3.33, 0.4],
            [-2.00,  2.00, 0.4],
            [-2.00,  1.00, 0.4],
            [5.00,  0.00, 0.4],
            [2.00,  -2.00, 0.4],
            [-0.5,  -2.5, 0.4],
            [-4.00,  -2.00, 0.4], #原: [-0.2, 1.0]
        ]
        unsafe_sphere_radius = 0.25  # 原: 0.2m × 3.33 ≈ 0.67m
        boundary_margin = 0.25

        # 目标位置 (保持在原点附近)
        target_sphere_pos = [0.0, 0.0, 0.4]    # [m]
        target_sphere_radius = 0.3             # 目标判定半径（参考环境为0.3m）

        # # 兼容性: 保留单个障碍物配置 (使用第一个障碍物)
        # unsafe_sphere_pos = [-2.66, -3.66, 0.4]
        unsafe_radius_h_eval_scale = 2.0  # 扩大判定半径来计算安全指标h(x)
        unsafe_radius_reward_scale = 1.0  # 奖励 shaping 使用的半径缩放
        goal_reached_dist = 0.3
        collision_dist = 0.35
        terminate_on_reach_avoid = False

    class terrain(LeggedRobotCfg.terrain ):
        #使用平地而非三维地形
        # mesh_type = 'heightfield' # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]

        #关闭课程学习
        # curriculum = True
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:

        #关闭地面高度起伏
        # measure_heights = True
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 12.
        terrain_width = 12.
        num_rows= 20 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.6, 0.2, 0.2, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["base","thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.


    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        touch_thr= 8 #N
        command_dead = 0.1
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.4
            lin_vel_z = -2.0
            #ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.3
            feet_air_time =  1.0 #这个最好保持为1 如果修改太了 腿可能会颤抖
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -10.0

            hip_pos = -0.5
            #feet_contact_number=-0.2
            orientation_eular = 0.32
            feet_contact_forces = -0.01
            foot_slip = -0.05
            vel_mismatch_exp = 0.4 #速度权重
            no_fly  = 0.05
            #action_smoothness=-0.01

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        max_iterations = 16000
        experiment_name = 'rough_go2'
        save_interval = 500 # check for potential saves every this many iterations
        resume = True
        resume_path = "/home/caohy/repositories/MCRA_RL/logs/rough_go2/Sep08_11-57-26_/model_18500.pt" # updated from load_run and chkpt


class GO2HighLevelCfg(GO2RoughCfg):
    """高层导航策略配置"""

    seed = 1  # keep interfaces stable by using a fixed seed
    reach_metric_scale = 0.2
    action_scale = [1.3, 1.0, 1.0]
    enable_manual_lidar = True
    lidar_max_range = 8.0
    lidar_num_bins = 16
    target_lidar_num_bins = 16
    target_lidar_max_range = 8.0

    class reward_shaping:
        # Reward shaping parameters for high-level navigation.
        goal_reached_dist = 0.3
        collision_dist = 0.35
        obstacle_avoid_dist = 1.2
        progress_scale = 45.0
        alignment_scale = 2.0
        obstacle_penalty_scale = 0.6
        yaw_rate_scale = 0.02
        action_smooth_scale = 0.08
        body_speed_scale = 1.0
        command_alignment_scale = 0.6
        idle_speed_threshold = 0.1
        idle_distance_threshold = 0.5
        idle_penalty_scale = 0.6
        success_reward = 100.0
        collision_penalty = 50.0
        timeout_penalty = 10.0
        reward_scale = 1.0
        reward_clip = 200.0
        terminate_on_safety_violation = True
        terminate_on_success = True

    class env(GO2RoughCfg.env):
        num_observations = 7  # placeholder; overwritten below
        num_actions = 3       # [vx, vy, vyaw]
        high_level_action_repeat = 10  # number of low-level steps per high-level action
        episode_length_s = 40

class GO2HighLevelCfgPPO(LeggedRobotCfgPPO):
    """高层导航策略PPO配置"""

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003
        learning_rate = 3e-4
        clip_param = 0.2
        value_clip_param = 0.2
        value_loss_coef = 0.5
        schedule = 'adaptive'
        desired_kl = 0.03
        min_lr = 5e-6
        max_lr = 5e-4
        num_learning_epochs = 3
        num_mini_batches = 4
        num_steps_per_env = 200 # increase horizon to give more time to reach the goal
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'high_level_go2'
        max_iterations = 30000
        save_interval = 200
        # gh_dump_interval = 50  # iteration interval for dumping g/h tensors
        resume = False
        resume_path = "/home/caohy/repositories/MCRA_RL/legged_gym_go2/legged_gym/scripts/logs/high_level_go2/20260105-102613/model_1300.pt"  # 你的checkpoint路径
        # 底层策略模型路径
        low_level_model_path = "/home/caohy/repositories/MCRA_RL/logs/rough_go2/Sep08_11-57-26_/model_18500.pt"

_base_high_level_obs = 8
_target_dim = GO2HighLevelCfg.target_lidar_num_bins
_lidar_dim = GO2HighLevelCfg.lidar_num_bins if GO2HighLevelCfg.enable_manual_lidar else 0
GO2HighLevelCfg.env.num_observations = _base_high_level_obs + _target_dim + _lidar_dim
