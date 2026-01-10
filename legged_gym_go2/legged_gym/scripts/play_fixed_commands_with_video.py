import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
from datetime import datetime
import imageio


def play_fixed_commands_with_video(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    # 避免命令被周期性重采样与随机heading影响：
    env_cfg.commands.resampling_time = 1e9  # 设为极大，几乎不再重采样
    env_cfg.commands.heading_command = False  # 直接使用我们设置的yaw速度指令

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 视频录制设置（基于相机传感器，稳定可靠）
    if RECORD_FRAMES and not env.headless:
        # 创建视频保存目录
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        os.makedirs(video_dir, exist_ok=True)

        # 生成视频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"fixed_commands_{args.task}_{timestamp}.mp4"
        video_path = os.path.join(video_dir, video_filename)

        print(f"开始录制视频: {video_path}")

        # 创建环境相机（挂在第一个子环境上）
        cam_props = gymapi.CameraProperties()
        cam_props.width = 1280
        cam_props.height = 720
        cam_props.enable_tensors = False
        cam_props.horizontal_fov = 60.0
        cam_props.near_plane = 0.01
        cam_props.far_plane = 50.0

        camera_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
        cam_w, cam_h = cam_props.width, cam_props.height
        # 设定相机位置与朝向（可根据需求调整/动态更新）
        cam_pos = gymapi.Vec3(3.0, 0.0, 1.5)  # 增加距离和高度
        cam_target = gymapi.Vec3(0.0, 0.0, 0.3)  # 稍微降低目标高度
        env.gym.set_camera_location(camera_handle, env.envs[0], cam_pos, cam_target)

        # 存储帧数据
        frames = []
        frame_count = 0
        # 预先步进一次图形，避免首帧为空
        env.gym.step_graphics(env.sim)

    # 定义测试命令序列
    test_commands = [
        [-1.0, 0.0, 0.0],   # 停止
        [-1.0, 0.0, 0.0],  # 停止
        [1.0, 0.0, 0.0],   # 前进
        [0.0, 1.0, 0.0],   # 右移
        [0.0, 0.0, 1.0],   # 右转
        [1.0, 0.0, 0.5],   # 前进+右转
        [0.5, 0.5, 0.0],   # 前进+右移
    ]
    
    command_duration = 150  # 每个命令持续200步
    current_command_idx = 0
    step_count = 0
    
    print("=== GO2 机器狗固定命令测试（带视频录制）===")
    print("将依次执行以下命令:")
    for i, cmd in enumerate(test_commands):
        print(f"  {i+1}. 前进={cmd[0]:.1f}, 左右={cmd[1]:.1f}, 转向={cmd[2]:.1f}")
    print("=" * 50)

    for i in range(10*int(env.max_episode_length)):
        # 更新命令
        if step_count % command_duration == 0:
            current_command_idx = (current_command_idx + 1) % len(test_commands)
            # 如果所有命令都执行完了，直接结束
            if current_command_idx == 0:
                print("所有测试命令执行完毕，结束测试")
                break
            current_command = test_commands[current_command_idx]
            print(f"切换到命令 {current_command_idx + 1}: 前进={current_command[0]:.1f}, 左右={current_command[1]:.1f}, 转向={current_command[2]:.1f}")
        
        # 设置当前命令
        current_command = test_commands[current_command_idx]
        env.commands[:, 0] = current_command[0]  # 前进速度
        env.commands[:, 1] = current_command[1]  # 左右速度  
        env.commands[:, 2] = current_command[2]  # 转向速度
        
        # 执行动作
        actions = policy(obs.detach())
        step_result = env.step(actions.detach())
        # 兼容不同环境的返回值长度（5 或 7）
        if isinstance(step_result, tuple) and len(step_result) >= 7:
            obs, _, rews, dones, infos, _, _ = step_result
        else:
            obs, _, rews, dones, infos = step_result
        
        # 录制视频帧（从相机传感器抓帧）
        if RECORD_FRAMES and not env.headless:
            # 动态跟随机器人：相机看向第一个机器人的位置
            try:
                target_pos = env.root_states[0, :3].detach().cpu().numpy()
                # 相机放在目标后上方，增加距离和高度，调整视角
                cam_target = gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]) + 0.2)
                cam_pos = gymapi.Vec3(float(target_pos[0]) - 3.0, float(target_pos[1]) + 0.0,
                                      float(target_pos[2]) + 1.5)
                env.gym.set_camera_location(camera_handle, env.envs[0], cam_pos, cam_target)
            except Exception:
                pass

            # 先步进图形，再渲染相机
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            # 读取颜色图像（RGBA, uint8）
            img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
            if isinstance(img, np.ndarray) and img.size > 0:
                # Isaac Gym 可能返回打包的 uint32 缓冲或 1D 数组，这里统一重解释为 (H, W, 4) 的 uint8 RGBA
                if img.dtype != np.uint8:
                    img = img.view(np.uint8)
                if img.ndim == 1:
                    try:
                        img = np.reshape(img, (cam_h, cam_w, 4))
                    except Exception:
                        # 宽高对调的兜底
                        img = np.reshape(img, (cam_w, cam_h, 4)).transpose(1, 0, 2)
                elif img.ndim == 2 and img.shape[-1] == cam_w * 4:
                    img = np.reshape(img, (cam_h, cam_w, 4))
                # 丢弃 alpha 通道，保持 RGB
                frame_rgb = img[..., :3].copy()
                frames.append(frame_rgb)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"已录制 {frame_count} 帧")
        
        # 显示状态
        if step_count % 50 == 0:  # 每50步显示一次
            base_vel = env.base_lin_vel[0].cpu().numpy()
            base_ang_vel = env.base_ang_vel[0].cpu().numpy()
            print(f"实际速度: 前进={base_vel[0]:.2f}, 左右={base_vel[1]:.2f}, 转向={base_ang_vel[2]:.2f}")
        
        # 如果环境重置，重新设置命令
        if dones[0]:
            env.commands[:, 0] = current_command[0]
            env.commands[:, 1] = current_command[1] 
            env.commands[:, 2] = current_command[2]
        
        step_count += 1

    # 完成视频录制
    if RECORD_FRAMES and 'frames' in locals() and frames:
        try:
            print("正在生成视频...")
            # 使用imageio创建视频
            with imageio.get_writer(video_path, fps=30) as writer:
                for frame in frames:
                    writer.append_data(frame)

            print(f"视频已保存到: {video_path}")
            print(f"共录制 {len(frames)} 帧")

        except Exception as e:
            print(f"视频生成错误: {e}")
            print("请检查是否安装了imageio和ffmpeg")

    print("固定命令测试（带视频录制）结束")


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True  # 启用视频录制
    MOVE_CAMERA = False
    args = get_args()
    play_fixed_commands_with_video(args)
