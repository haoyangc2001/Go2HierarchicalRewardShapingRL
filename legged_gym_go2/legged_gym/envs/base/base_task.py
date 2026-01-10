
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()       #

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        # 检查是否需要创建图形化界面（非无头模式）
        if self.headless == False:
            # 以下代码用于设置可视化查看器和键盘事件监听
            # 创建一个可视化查看器，需要传入仿真环境对象和相机属性配置
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())        # 创建一个可视化窗口
            # 订阅ESC键的键盘事件，用于退出仿真
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")        # 设置ESC键退出功能
            # 订阅V键的键盘事件，用于切换查看器的同步模式
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")  # 监听V键，切换同步渲染模式

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        # 检查是否存在视图窗口
        if self.viewer:
            # 检查视图窗口是否已关闭
            if self.gym.query_viewer_has_closed(self.viewer):
                # 如果窗口已关闭，退出程序
                sys.exit()

            # 检查并处理键盘事件
            for evt in self.gym.query_viewer_action_events(self.viewer):
                # 处理退出事件
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                # 处理切换视图同步模式事件
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    # 切换视图同步模式的启用状态
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # 当运行在GPU设备上时，获取仿真结果
            if self.device != 'cpu':
                # 从GPU同步仿真结果到CPU内存
                self.gym.fetch_results(self.sim, True)

            # 执行图形渲染步骤
            if self.enable_viewer_sync:
                # 推进图形系统状态更新
                self.gym.step_graphics(self.sim)
                # 将当前仿真状态绘制到视图窗口
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # 根据需要同步帧时间，保持稳定的渲染帧率
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                # 非同步模式下，仅处理视图事件而不更新渲染
                self.gym.poll_viewer_events(self.viewer)