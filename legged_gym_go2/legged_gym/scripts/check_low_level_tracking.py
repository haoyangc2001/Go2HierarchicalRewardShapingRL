#!/usr/bin/env python3
import os

import isaacgym
from isaacgym import gymutil
import torch


from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyRunner
from legged_gym.envs.go2.go2_config import GO2HighLevelCfgPPO


def _parse_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Task name."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL algorithm."},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--cmd", "type": str, "default": "0.4,0.0,0.0",
         "help": "Commanded [vx, vy, vyaw], format: 'vx,vy,vyaw'."},
        {"name": "--steps", "type": int, "default": 500, "help": "Total steps to run."},
        {"name": "--warmup", "type": int, "default": 50, "help": "Warmup steps skipped from stats."},
        {"name": "--log_interval", "type": int, "default": 50, "help": "Step interval for logging."},
        {"name": "--low_level_checkpoint", "type": str, "default": "",
         "help": "Override low-level policy checkpoint path."},
        {"name": "--disable_domain_rand", "action": "store_true", "default": False,
         "help": "Disable friction randomization and pushes."},
    ]
    args = gymutil.parse_arguments(
        description="Check low-level tracking (cmd_speed vs body_speed).",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def _load_low_level_policy(env, train_cfg, checkpoint_path, device):
    train_cfg.runner.resume = False
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg_dict, device=device)
    runner.load(checkpoint_path, load_optimizer=False)
    return runner.get_inference_policy(device=env.device)


def _parse_cmd(cmd_str):
    parts = [p.strip() for p in cmd_str.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--cmd expects 3 comma-separated floats, got: {cmd_str}")
    return [float(p) for p in parts]


def main():
    args = _parse_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed

    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.commands.resampling_time = 1e9
    env_cfg.commands.heading_command = False
    if args.disable_domain_rand:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    default_ckpt = GO2HighLevelCfgPPO.runner.low_level_model_path
    checkpoint_path = args.low_level_checkpoint or default_ckpt
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Low-level checkpoint not found: {checkpoint_path}")

    policy = _load_low_level_policy(env, train_cfg, checkpoint_path, args.rl_device)

    cmd_list = _parse_cmd(args.cmd)
    cmd = torch.tensor(cmd_list, dtype=torch.float, device=env.device)
    cmd_buf = cmd.unsqueeze(0).repeat(env.num_envs, 1)

    obs, _ = env.reset()
    cmd_speed_sum = 0.0
    body_speed_sum = 0.0
    ratio_sum = 0.0
    count = 0

    print("Low-level tracking check")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  envs       : {env.num_envs}")
    print(f"  cmd        : {cmd_list}")
    print(f"  steps      : {args.steps} (warmup {args.warmup})")

    for step in range(args.steps):
        env.commands[:, :3] = cmd_buf
        env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
        step_result = env.step(actions)
        if isinstance(step_result, tuple) and len(step_result) >= 5:
            obs, _, _, dones, _infos = step_result[:5]
        else:
            obs, _ = env.get_observations(), None
            dones = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        body_speed = torch.norm(env.base_lin_vel[:, :2], dim=1)
        cmd_speed = torch.norm(cmd_buf[:, :2], dim=1)
        ratio = body_speed / cmd_speed.clamp_min(1e-6)

        if step >= args.warmup:
            valid = ~dones.bool()
            if valid.any():
                cmd_speed_sum += cmd_speed[valid].sum().item()
                body_speed_sum += body_speed[valid].sum().item()
                ratio_sum += ratio[valid].sum().item()
                count += int(valid.sum().item())

        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            vx = env.base_lin_vel[:, 0].mean().item()
            vy = env.base_lin_vel[:, 1].mean().item()
            vyaw = env.base_ang_vel[:, 2].mean().item()
            print(
                f"step {step + 1:04d} | cmd_speed {cmd_speed.mean().item():.3f} | "
                f"body_speed {body_speed.mean().item():.3f} | ratio {ratio.mean().item():.3f} | "
                f"vx {vx:.3f} vy {vy:.3f} vyaw {vyaw:.3f}"
            )

    if count == 0:
        print("No valid samples collected (all done or warmup too long).")
        return

    print("Averages (post-warmup, excluding dones)")
    print(f"  cmd_speed  : {cmd_speed_sum / count:.3f}")
    print(f"  body_speed : {body_speed_sum / count:.3f}")
    print(f"  ratio      : {ratio_sum / count:.3f}")


if __name__ == "__main__":
    main()
