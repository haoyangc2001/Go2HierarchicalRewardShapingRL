import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Tuple

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import update_cfg_from_args

import numpy as np
import torch
import time
from datetime import datetime
import imageio

from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from rsl_rl.algorithms.reach_avoid_ppo import ReachAvoidPPO
from rsl_rl.modules import ActorCritic




class HierarchicalVecEnv:
    def __init__(self, env: HierarchicalGO2Env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.device = env.device

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, g_vals, h_vals = self.env.reset()
        return obs, g_vals, h_vals

    def step(self, actions: torch.Tensor):
        obs, g_vals, h_vals, dones, infos = self.env.step(actions)
        return obs, g_vals, h_vals, dones, infos

    def close(self) -> None:
        self.env.close()





def create_env(env_cfg, train_cfg, args, device) -> HierarchicalVecEnv:
    base_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )
    return HierarchicalVecEnv(base_env)




def play(args):
    # 声明使用的全局变量
    global RECORD_FRAMES
    global MOVE_CAMERA
    
    # 初始化环境和资源变量
    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()
    train_cfg.policy.actor_hidden_dims = [512, 512, 512, 512]
    train_cfg.policy.critic_hidden_dims = [512, 512, 512, 512]
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    device = torch.device(args.rl_device)

    # 准备环境
    env = create_env(env_cfg, train_cfg, args, device)
    
    # 获取初始观测值
    obs, _, _ = env.reset()
    
    # 视频录制和可视化相关变量
    camera_handle = None
    frames = []
    cam_w, cam_h = 640, 480  # 降低分辨率以提高性能
    visual_elements = []  # 存储可视化元素句柄，用于清理
    env_origin = None  # 环境原点，用于坐标转换

    # 加载策略 - 使用ReachAvoidPPO算法来加载训练好的模型
    if hasattr(args, 'model_path') and args.model_path:
        model_path = args.model_path
    else:
        # 首先尝试从用户指定的路径查找高层导航策略模型
        custom_high_level_path = "/home/wutr/IsaacGym/logs/high_level_go2"
        model_path = None
        
        if os.path.exists(custom_high_level_path):
            print(f"在用户指定路径中搜索高层导航策略模型: {custom_high_level_path}")
            # 遍历high_level_go2目录下的所有子目录
            for dir_name in os.listdir(custom_high_level_path):
                dir_path = os.path.join(custom_high_level_path, dir_name)
                if os.path.isdir(dir_path):
                    # 检查该目录下是否有model_final.pt文件
                    candidate_path = os.path.join(dir_path, 'model_final.pt')
                    if os.path.exists(candidate_path):
                        model_path = candidate_path
                        print(f"找到高层导航策略模型: {model_path}")
                        break
                    
                    # 如果没有model_final.pt，查找其他模型文件
                    model_files = [f for f in os.listdir(dir_path) if f.startswith('model_') and f.endswith('.pt')]
                    if model_files:
                        # 按文件名排序，找到最新的迭代模型
                        model_files.sort(reverse=True)
                        model_path = os.path.join(dir_path, model_files[0])
                        print(f"自动选择高层导航策略模型: {model_path}")
                        break
    
    print(f"加载模型: {model_path}")
    


    # 创建ActorCritic网络
    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std,
    ).to(device)

    alg = ReachAvoidPPO(
        actor_critic=actor_critic,
        device=device,
        **train_cfg.algorithm.__dict__,
    )


    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"模型文件中的键: {list(checkpoint.keys())}")  # 打印模型文件中的所有键
        
        # 尝试加载模型权重，处理不同的模型格式
        if 'actor_critic' in checkpoint:
            alg.actor_critic.load_state_dict(checkpoint["actor_critic"])
            print("成功加载actor_critic权重")
        elif 'model_state_dict' in checkpoint:
            # 尝试从model_state_dict加载
            alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
            print("成功从model_state_dict加载权重")
        else:
            # 尝试直接加载到actor_critic
            print("模型文件中没有'actor_critic'或'model_state_dict'键，尝试直接加载")
            alg.actor_critic.load_state_dict(checkpoint)
            print("成功直接加载模型权重")
            
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 尝试其他加载方式
        try:
            # 尝试直接加载模型
            model_data = torch.load(model_path, map_location=device)
            print("尝试不同的加载策略...")
            
            # 检查是否有嵌套的model_state_dict
            if 'model_state_dict' in model_data:
                alg.actor_critic.load_state_dict(model_data['model_state_dict'])
                print("成功从model_state_dict加载权重")
            elif 'actor' in model_data and 'critic' in model_data:
                # 尝试分别加载actor和critic
                alg.actor_critic.actor.load_state_dict(model_data['actor'])
                alg.actor_critic.critic.load_state_dict(model_data['critic'])
                print("成功分别加载actor和critic权重")
            else:
                # 最后的尝试：打印所有可能的键结构
                print("模型文件结构分析:")
                for key, value in model_data.items():
                    if hasattr(value, 'keys'):
                        print(f"  {key}: {list(value.keys())}")
                    else:
                        print(f"  {key}: {type(value)}")
                
                print("无法加载模型，请检查模型文件格式是否正确")
                sys.exit(1)
        except Exception as e2:
            print(f"所有加载尝试都失败: {e2}")
            sys.exit(1)


    # 创建策略推理函数
    def policy(obs):
        with torch.no_grad():
            actions = alg.act(obs)[0]  # 使用ReachAvoidPPO的act方法
        return actions

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(alg.actor_critic, path)  # 导出actor_critic网络
        print('Exported policy as jit script to: ', path)

    # 视频录制设置（优化版，提高稳定性和性能）
    if RECORD_FRAMES and not env.env.base_env.headless:
        try:
            # 打印LEGGED_GYM_ROOT_DIR的值进行调试
            print(f"LEGGED_GYM_ROOT_DIR = {LEGGED_GYM_ROOT_DIR}")
            
            # 创建视频保存目录
            video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
            print(f"视频保存目录: {video_dir}")
            os.makedirs(video_dir, exist_ok=True)

            # 生成视频文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"play_go2_{timestamp}.mp4"
            video_path = os.path.join(video_dir, video_filename)

            print(f"开始录制视频: {video_path}")

            # 创建环境相机（挂在第一个子环境上）
            cam_props = gymapi.CameraProperties()
            cam_props.width = cam_w
            cam_props.height = cam_h
            cam_props.enable_tensors = False
            cam_props.horizontal_fov = 75.0  # 更宽的视角
            cam_props.near_plane = 0.01
            cam_props.far_plane = 50.0

            camera_handle = env.env.base_env.gym.create_camera_sensor(env.env.base_env.envs[0], cam_props)
            
            # 设定相机初始位置与朝向 - 改为从上往下看的斜俯视视角
            cam_pos = gymapi.Vec3(0.0, -5.0, 4.0)  # 更远更高的位置，从上方斜向下看
            cam_target = gymapi.Vec3(0.0, 0.0, 0.4)  # 保持目标点不变
            env.env.base_env.gym.set_camera_location(camera_handle, env.env.base_env.envs[0], cam_pos, cam_target)
            
            # 获取环境原点位置
            env_origin = env.env.base_env.env_origins[0].cpu().numpy()
            print(f"环境原点位置: {env_origin}")
            
            # 获取目标和障碍物信息并保存，供后续在帧上绘制使用
            try:
                # 确保numpy可用
                import numpy as np
                
                # 从high_level_config获取目标和障碍物的位置和半径
                print(f"尝试访问env.env.high_level_config...")
                high_level_config = env.env.high_level_config
                print(f"成功访问high_level_config")
                
                # 保存目标信息
                target_pos = high_level_config.target_sphere_pos
                target_radius = high_level_config.target_radius
                print(f"目标位置: {target_pos}, 半径: {target_radius}")
                
                # 调整目标位置，使其与环境原点对齐
                target_world_pos = np.array([
                    float(target_pos[0] + env_origin[0]),
                    float(target_pos[1] + env_origin[1]),
                    float(target_pos[2])
                ])
                print(f"调整后的目标世界位置: {target_world_pos}")
                
                # 保存障碍物信息
                obstacle_pos = high_level_config.unsafe_sphere_pos
                obstacle_radius = high_level_config.unsafe_sphere_radius
                print(f"障碍物位置: {obstacle_pos}, 半径: {obstacle_radius}")
                
                # 调整障碍物位置，使其与环境原点对齐
                obstacle_world_pos = np.array([
                    float(obstacle_pos[0] + env_origin[0]),
                    float(obstacle_pos[1] + env_origin[1]),
                    float(obstacle_pos[2])
                ])
                print(f"调整后的障碍物世界位置: {obstacle_world_pos}")
                
                # 保存这些信息供后续绘制使用
                visual_info = {
                    'target_pos': target_world_pos,
                    'target_radius': target_radius,
                    'obstacle_pos': obstacle_world_pos,
                    'obstacle_radius': obstacle_radius
                }
                print(f"已保存目标和障碍物信息，准备在视频帧上绘制")
                
            except Exception as vis_error:
                print(f"获取可视化信息时出错: {vis_error}")
                # 输出详细的错误信息以帮助调试
                import traceback
                traceback.print_exc()
                visual_info = None
                # 即使可视化失败，也继续执行
                pass
            
            # 预先步进一次图形，避免首帧为空
            env.env.base_env.gym.step_graphics(env.env.base_env.sim)
            
        except Exception as e:
            print(f"初始化视频录制失败: {e}")
            # 输出详细的错误信息以帮助调试
            import traceback
            traceback.print_exc()
            RECORD_FRAMES = False

        frame_count = 0




    # 导入必要的绘图库
    try:
        import cv2
        has_opencv = True
        print("成功导入OpenCV库，将用于在视频帧上绘制目标和障碍物")
    except ImportError:
        has_opencv = False
        print("警告：未找到OpenCV库，将使用纯numpy绘制目标和障碍物")
        
    # 定义3D到2D坐标转换函数
    def world_to_image(world_pos, camera_pos, camera_target, camera_up, fov, img_w, img_h):
        # 简单的透视投影实现
        # 计算相机坐标系
        forward = np.array(camera_target) - np.array(camera_pos)
        forward = forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 0 else np.array([0,0,1])
        
        right = np.cross(forward, np.array(camera_up))
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1,0,0])
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0,1,0])
        
        # 计算点相对于相机的位置
        point = np.array(world_pos) - np.array(camera_pos)
        
        # 投影到相机平面
        x = np.dot(point, right)
        y = np.dot(point, up)
        z = np.dot(point, forward)
        
        if z <= 0:  # 点在相机后面，不会显示
            return None
        
        # 应用透视投影
        scale = img_h / (2 * np.tan(fov * np.pi / 360))
        
        u = img_w / 2 + x * scale / z
        v = img_h / 2 - y * scale / z  # y轴反转，因为图像坐标的y是向下的
        
        # 检查点是否在图像范围内
        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            return None
        
        return (int(u), int(v))
        
    # 定义在图像上绘制圆柱体的函数
    def draw_cylinder_on_image(img, center_2d, radius_2d, height_2d, color, has_opencv):
        if center_2d is None:
            return img
        
        cx, cy = center_2d
        r = int(radius_2d)
        h = int(height_2d)
        
        if has_opencv:
            # 使用OpenCV绘制圆柱体
            # 绘制底部圆
            cv2.circle(img, (cx, cy), r, color, 2)
            # 绘制顶部圆
            cv2.circle(img, (cx, cy - h), r, color, 2)
            # 绘制连接的垂直线
            cv2.line(img, (cx - r, cy), (cx - r, cy - h), color, 2)
            cv2.line(img, (cx + r, cy), (cx + r, cy - h), color, 2)
            cv2.line(img, (cx, cy - r), (cx, cy - h - r), color, 2)
            cv2.line(img, (cx, cy + r), (cx, cy - h + r), color, 2)
        else:
            # 使用numpy手动绘制简单的圆柱体轮廓
            # 简化为绘制十字标记
            for i in range(max(0, cx - r), min(img.shape[1], cx + r + 1)):
                if 0 <= cy < img.shape[0]:
                    img[cy, i] = color
            for i in range(max(0, cy - r), min(img.shape[0], cy + r + 1)):
                if 0 <= cx < img.shape[1]:
                    img[i, cx] = color
        
        return img

    # 初始化frames和visual_info变量，但不覆盖camera_handle的值
    frames = []
    if 'visual_info' not in locals():
        visual_info = None

    # 主循环
    for i in range(100):
        actions = policy(obs.detach())    # 策略推理生成动作
        step_result = env.step(actions.detach())  # 执行动作并获取环境反馈
        # 兼容不同环境的返回值长度（5 或 7）
        if isinstance(step_result, tuple) and len(step_result) >= 7:
            obs, _, rews, dones, infos, _, _ = step_result
        else:
            obs, _, rews, dones, infos = step_result

        # 录制视频帧（从相机传感器抓帧）
        if RECORD_FRAMES and not env.env.base_env.headless:
            try:
                # 确保camera_handle已初始化
                if camera_handle is None:
                    print("警告：camera_handle未初始化，尝试重新初始化...")
                    try:
                        # 创建环境相机（挂在第一个子环境上）
                        cam_props = gymapi.CameraProperties()
                        cam_props.width = cam_w
                        cam_props.height = cam_h
                        cam_props.enable_tensors = False
                        cam_props.horizontal_fov = 75.0  # 更宽的视角
                        cam_props.near_plane = 0.01
                        cam_props.far_plane = 50.0

                        camera_handle = env.env.base_env.gym.create_camera_sensor(env.env.base_env.envs[0], cam_props)
                        
                        # 设定相机初始位置与朝向 - 改为从上往下看的斜俯视视角
                        cam_pos = gymapi.Vec3(0.0, -5.0, 4.0)  # 更远更高的位置，从上方斜向下看
                        cam_target = gymapi.Vec3(0.0, 0.0, 0.4)  # 保持目标点不变
                        env.env.base_env.gym.set_camera_location(camera_handle, env.env.base_env.envs[0], cam_pos, cam_target)
                        print("camera_handle已成功重新初始化")
                    except Exception as cam_error:
                        print(f"重新初始化camera_handle失败: {cam_error}")
                        continue

                # 确保numpy可用
                import numpy as np
                # 动态跟随机器人：相机看向第一个机器人的位置
                if i % 5 == 0:  # 每5帧更新一次相机位置，减少计算量
                    try:
                        target_pos = env.env.base_env.root_states[0, :3].detach().cpu().numpy()
                        # 相机放在目标后上方，保持更稳定的视角 - 改为从上往下看的斜俯视视角
                        cam_target = gymapi.Vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]) + 0.4)
                        # 调整相机距离，确保可以看到目标和障碍物 - 更远更高的位置
                        cam_pos = gymapi.Vec3(
                            float(target_pos[0]),  # 水平位置与机器人对齐
                            float(target_pos[1]) - 5.0,  # 更远的距离，从机器人后方
                            float(target_pos[2]) + 4.0   # 更高的视角
                        )
                        env.env.base_env.gym.set_camera_location(camera_handle, env.env.base_env.envs[0], cam_pos, cam_target)
                        
                        # 保存当前相机参数，用于后续坐标转换
                        current_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                        current_cam_target = np.array([cam_target.x, cam_target.y, cam_target.z])
                        current_cam_up = np.array([0, 0, 1])  # 假设相机朝上为z轴
                    except Exception:
                        pass

                # 先步进图形，再渲染相机
                env.env.base_env.gym.step_graphics(env.env.base_env.sim)
                env.env.base_env.gym.render_all_camera_sensors(env.env.base_env.sim)
                
                # 读取颜色图像（RGBA, uint8）
                img = env.env.base_env.gym.get_camera_image(env.env.base_env.sim, env.env.base_env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
                if isinstance(img, np.ndarray) and img.size > 0:
                    # 确保图像数据格式正确
                    try:
                        # Isaac Gym 可能返回打包的 uint32 缓冲或 1D 数组，这里统一处理
                        if img.dtype != np.uint8:
                            img = img.view(np.uint8)
                        
                        # 重塑图像为正确的形状
                        if img.ndim == 1:
                            try:
                                img = np.reshape(img, (cam_h, cam_w, 4))
                            except Exception:
                                # 宽高对调的兜底方案
                                img = np.reshape(img, (cam_w, cam_h, 4)).transpose(1, 0, 2)
                        elif img.ndim == 2 and img.shape[-1] == cam_w * 4:
                            img = np.reshape(img, (cam_h, cam_w, 4))
                        
                        # 确保图像维度正确
                        if img.ndim == 3 and img.shape[2] >= 3:
                            # 丢弃 alpha 通道，保持 RGB
                            frame_rgb = img[..., :3].copy()
                            
                            # 在图像上绘制目标和障碍物
                            if visual_info is not None and 'current_cam_pos' in locals():
                                try:
                                    # 计算目标和障碍物在图像上的投影位置
                                    # 简化计算：假设半径和高度在图像上的大小与实际大小成正比
                                    # 这里使用一个近似因子，根据相机距离调整大小
                                    camera_distance = np.linalg.norm(current_cam_pos - current_cam_target)
                                    scale_factor = 30 * camera_distance  # 调整这个因子以获得合适的大小
                                    
                                    # 转换目标位置到图像坐标
                                    target_2d = world_to_image(
                                        visual_info['target_pos'], 
                                        current_cam_pos, 
                                        current_cam_target, 
                                        current_cam_up, 
                                        75.0,  # 相机水平FOV
                                        cam_w, 
                                        cam_h
                                    )
                                    
                                    # 绘制绿色目标圆柱体
                                    target_radius_2d = visual_info['target_radius'] * scale_factor
                                    target_height_2d = 0.8 * scale_factor  # 圆柱体高度
                                    if target_2d is not None:
                                        frame_rgb = draw_cylinder_on_image(
                                            frame_rgb, 
                                            target_2d, 
                                            target_radius_2d, 
                                            target_height_2d, 
                                            [0, 255, 0],  # 绿色
                                            has_opencv
                                        )
                                    
                                    # 转换障碍物位置到图像坐标
                                    obstacle_2d = world_to_image(
                                        visual_info['obstacle_pos'], 
                                        current_cam_pos, 
                                        current_cam_target, 
                                        current_cam_up, 
                                        75.0,  # 相机水平FOV
                                        cam_w, 
                                        cam_h
                                    )
                                    
                                    # 绘制红色障碍物圆柱体
                                    obstacle_radius_2d = visual_info['obstacle_radius'] * scale_factor
                                    obstacle_height_2d = 0.8 * scale_factor  # 圆柱体高度
                                    if obstacle_2d is not None:
                                        frame_rgb = draw_cylinder_on_image(
                                            frame_rgb, 
                                            obstacle_2d, 
                                            obstacle_radius_2d, 
                                            obstacle_height_2d, 
                                            [255, 0, 0],  # 红色
                                            has_opencv
                                        )
                                except Exception as draw_error:
                                    print(f"绘制目标和障碍物时出错: {draw_error}")
                                    # 即使绘制失败，也继续执行
                                    pass
                            
                            # 添加帧到列表
                            frames.append(frame_rgb)
                            frame_count += 1
                            # 减少打印频率，提高性能
                            if frame_count % 200 == 0:
                                print(f"已录制 {frame_count} 帧")
                    except Exception as frame_error:
                        # 静默处理单帧错误，确保仿真继续
                        if frame_count % 50 == 0:  # 只在特定间隔打印错误
                            print(f"处理帧时出现非严重错误: {frame_error}")
            except Exception as e:
                # 捕获并处理任何录制相关的异常
                print(f"视频录制过程中出错: {e}")
                print("已暂停视频录制，但仿真将继续")
                RECORD_FRAMES = False



    # 完成视频录制
    if RECORD_FRAMES and 'frames' in locals() and frames:
        try:
            print("正在生成视频...")
            # 使用imageio创建视频，降低帧率以提高稳定性
            video_fps = 15  # 降低帧率以减少处理负担
            with imageio.get_writer(video_path, fps=video_fps) as writer:
                for i, frame in enumerate(frames):
                    writer.append_data(frame)
                    # 进度提示，减少打印频率
                    if i % 50 == 0:
                        print(f"正在写入视频帧: {i}/{len(frames)}")

            print(f"视频已保存到: {video_path}")
            print(f"共录制 {len(frames)} 帧")

        except Exception as e:
            print(f"视频生成错误: {e}")
            print("请检查是否安装了imageio和ffmpeg")
            # 尝试保存单张图像作为备份
            if frames:
                try:
                    backup_img_path = video_path.replace('.mp4', '_frame0.png')
                    imageio.imwrite(backup_img_path, frames[0])
                    print(f"已保存第一帧作为备份: {backup_img_path}")
                except Exception:
                    pass

    # 清理资源
    try:
        print("正在清理资源...")
        
        # 清理可视化元素
        if 'visual_elements' in locals() and visual_elements:
            for element in visual_elements:
                try:
                    env.env.base_env.gym.destroy_lines(element)
                except Exception:
                    pass
            visual_elements.clear()
        
        # 清理相机
        if camera_handle is not None:
            try:
                env.env.base_env.gym.destroy_camera_sensor(env.env.base_env.envs[0], camera_handle)
                camera_handle = None
            except Exception:
                pass
        
        # 清理帧数据
        if frames:
            frames.clear()
            # 触发垃圾回收
            import gc
            gc.collect()
            
    except Exception as cleanup_error:
        print(f"资源清理过程中出错: {cleanup_error}")

    print("仿真播放结束")





if __name__ == '__main__':
    # 导入必要的模块
    import sys
    import os
    import signal
    
    # 设置信号处理器来捕获段错误
    def signal_handler(sig, frame):
        print(f"捕获到信号: {sig}, 正在安全退出...")
        try:
            # 强制资源清理
            import gc
            gc.collect()
        except:
            pass
        sys.exit(1)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    # 全局设置
    global EXPORT_POLICY
    global RECORD_FRAMES
    global MOVE_CAMERA
    
    EXPORT_POLICY = True
    RECORD_FRAMES = True  # 启用视频录制
    MOVE_CAMERA = False
    
    try:
        args = get_args()
        play(args)
        
        # 强制资源清理
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 尝试强制清理后退出
        try:
            import gc
            gc.collect()
        except:
            pass
        sys.exit(1)
    
    # 使用os._exit强制退出，避免分段错误
    os._exit(0)
