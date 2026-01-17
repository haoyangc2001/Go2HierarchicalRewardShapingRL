#!/usr/bin/env python3
import argparse
import math
import sys
import time

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args
from legged_gym.utils.hierarchical_env_utils import create_env


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, default=200, help="Number of high-level steps to run.")
    parser.add_argument("--vx", type=float, default=0.6, help="Fixed forward vx command in [-1, 1].")
    parser.add_argument("--vyaw", type=float, default=1.0, help="Yaw command gain for target heading.")
    parser.add_argument(
        "--report_interval",
        type=int,
        default=50,
        help="Print interim stats every N steps (0 disables).",
    )
    diag_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    return base_args, diag_args


def main() -> None:
    args, diag = _parse_args()
    device = torch.device(args.rl_device)

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    env = create_env(env_cfg, train_cfg, args, device)

    obs = env.reset()
    obs = obs.to(device)

    progress_sum = 0.0
    progress_abs_sum = 0.0
    progress_count = 0.0
    cmd_speed_sum = 0.0
    body_speed_sum = 0.0
    goal_dist_sum = 0.0
    reach_metric_sum = 0.0
    min_hazard_sum = 0.0
    done_frac_sum = 0.0

    vx_cmd = torch.clamp(torch.tensor(float(diag.vx), device=device), -1.0, 1.0)
    start_time = time.time()
    for step in range(diag.steps):
        target_dir = obs[:, 6:8].clamp(-1.0, 1.0)
        target_angle = torch.atan2(target_dir[:, 1], target_dir[:, 0])
        yaw_cmd = torch.clamp(target_angle / math.pi, -1.0, 1.0) * float(diag.vyaw)
        command = torch.zeros(env.num_envs, 3, device=device)
        command[:, 0] = vx_cmd
        command[:, 2] = yaw_cmd
        obs, rewards, dones, infos = env.step(command)
        target_distance = infos.get(
            "target_distance_est", torch.zeros(env.num_envs, device=device)
        )
        reach_metric = infos.get("reach_metric", env.env.base_env.reach_metric.to(device))
        min_hazard = infos.get("min_hazard_distance", torch.zeros(env.num_envs, device=device))
        progress = infos.get("progress", torch.zeros(env.num_envs, device=device))
        cmd_speed = infos.get("command_speed", torch.zeros(env.num_envs, device=device))
        body_speed = infos.get("body_speed", torch.zeros(env.num_envs, device=device))

        valid = (~dones).float()

        progress_sum += (progress * valid).sum().item()
        progress_abs_sum += (progress.abs() * valid).sum().item()
        progress_count += valid.sum().item()

        cmd_speed_sum += cmd_speed.mean().item()
        body_speed_sum += body_speed.mean().item()
        goal_dist_sum += target_distance.mean().item()
        reach_metric_sum += reach_metric.mean().item()
        min_hazard_sum += min_hazard.mean().item()
        done_frac_sum += dones.float().mean().item()

        if diag.report_interval and (step + 1) % diag.report_interval == 0:
            count = max(progress_count, 1.0)
            print(
                f"step {step + 1:04d} | progress {progress_sum / count:.6f} | "
                f"progress_abs {progress_abs_sum / count:.6f} | "
                f"cmd_speed {cmd_speed_sum / (step + 1):.3f} | "
                f"body_speed {body_speed_sum / (step + 1):.3f} | "
                f"goal_dist {goal_dist_sum / (step + 1):.3f} | "
                f"reach_metric {reach_metric_sum / (step + 1):.3f} | "
                f"min_hazard {min_hazard_sum / (step + 1):.3f} | "
                f"done_frac {done_frac_sum / (step + 1):.3f}"
            )

        obs = obs.to(device)

    elapsed = time.time() - start_time
    count = max(progress_count, 1.0)
    print("Command tracking summary")
    print(f"  steps       : {diag.steps}")
    print(f"  num_envs    : {env.num_envs}")
    print(f"  cmd (norm)  : vx = {diag.vx:.3f}, vy = 0.000, vyaw = atan2(target_dir) * {diag.vyaw:.3f}")
    print(f"  progress    : {progress_sum / count:.6f}")
    print(f"  progress_abs: {progress_abs_sum / count:.6f}")
    print(f"  cmd_speed   : {cmd_speed_sum / diag.steps:.3f}")
    print(f"  body_speed  : {body_speed_sum / diag.steps:.3f}")
    print(f"  goal_dist   : {goal_dist_sum / diag.steps:.3f}")
    print(f"  reach_metric: {reach_metric_sum / diag.steps:.3f}")
    print(f"  min_hazard  : {min_hazard_sum / diag.steps:.3f}")
    print(f"  done_frac   : {done_frac_sum / diag.steps:.3f}")
    print(f"  elapsed     : {elapsed:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
