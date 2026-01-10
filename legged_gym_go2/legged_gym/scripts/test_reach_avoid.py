#!/usr/bin/env python3
"""
Eval script: load a trained high-level reach-avoid policy, roll out several random
trajectories, and dump the XY paths to a JSON file that can be visualized later.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

import isaacgym
from isaacgym import gymtorch
import torch
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2HighLevelCfgPPO
from legged_gym.scripts.train_reach_avoid import HierarchicalVecEnv, create_env
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args
from rsl_rl.modules import ActorCritic

FIXED_SPAWN_POINTS = [(-5, 5), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)]


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reach-avoid policy rollouts and store the resulting trajectories."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the trained high-level checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-trajs",
        type=int,
        default=10,
        help="How many trajectories to record.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max high-level steps per trajectory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reach_avoid_rollouts.json",
        help="JSON file used to store the XY paths.",
    )
    parser.add_argument(
        "--low-level-model",
        type=str,
        default=None,
        help="Optional override for the pretrained low-level locomotion policy.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel envs for rollouts (each env produces one trajectory).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open Isaac Gym viewer during rollouts.",
    )
    parser.add_argument(
        "--max-reset-attempts",
        type=int,
        default=20,
        help="Safety cap on how many environment resets we attempt.",
    )
    eval_args, remaining = parser.parse_known_args()
    # Let Isaac Gym / legged_gym parse their usual CLI flags using the remaining argv.
    sys.argv = [sys.argv[0]] + remaining
    if eval_args.num_envs != 1:
        print("[INFO] For sequential trajectory collection we force num_envs=1.")
    eval_args.num_envs = 1
    return eval_args


def _resolve_low_level_model_path(
    eval_args: argparse.Namespace,
    train_cfg: GO2HighLevelCfgPPO,
    checkpoint: dict,
) -> str:
    candidates = []
    if eval_args.low_level_model:
        candidates.append(("argument --low-level-model", eval_args.low_level_model))
    ckpt_path = checkpoint.get("low_level_model_path") if isinstance(checkpoint, dict) else None
    if ckpt_path:
        candidates.append(("checkpoint metadata", ckpt_path))
    cfg_path = getattr(train_cfg.runner, "low_level_model_path", None)
    if cfg_path:
        candidates.append(("config default", cfg_path))

    checked_sources = []
    for source, path in candidates:
        expanded = os.path.expanduser(str(path))
        checked_sources.append(f"{source}: {expanded}")
        if os.path.isfile(expanded):
            print(f"Using low-level policy from {source}: {expanded}")
            return expanded
        else:
            print(f"[WARN] Low-level checkpoint not found at {expanded} ({source}).")

    detail = "\n  ".join(checked_sources) if checked_sources else "  (no candidates available)"
    raise FileNotFoundError(
        "Unable to locate a valid low-level policy checkpoint. Checked:\n"
        f"{detail}\n"
        "Please pass --low-level-model with an existing file, or update the config/checkpoint metadata."
    )


def _set_spawn_position(env: HierarchicalVecEnv, spawn_xy: Sequence[float]):
    base_env = env.env.base_env
    high_level_env = env.env.high_level_env
    device = base_env.device
    spawn_tensor = torch.tensor(spawn_xy, device=device, dtype=torch.float)

    base_env.root_states[0, 0] = spawn_tensor[0]
    base_env.root_states[0, 1] = spawn_tensor[1]
    env_id = torch.tensor([0], device=device, dtype=torch.int32)
    base_env.gym.set_actor_root_state_tensor_indexed(
        base_env.sim,
        gymtorch.unwrap_tensor(base_env.root_states),
        gymtorch.unwrap_tensor(env_id),
        len(env_id),
    )
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.compute_observations()
    high_level_env._compute_high_level_observations()
    obs = high_level_env.get_observations()
    avoid_metric, reach_metric = high_level_env._get_current_metrics()
    g_vals, h_vals = high_level_env.compute_g_h_values(avoid_metric, reach_metric)
    return obs, g_vals, h_vals


def _infer_hidden_dims_from_state_dict(state_dict: dict, prefix: str, expected_output_dim: int) -> List[int]:
    """Infer hidden layer sizes from a Sequential actor/critic stored in the checkpoint."""
    if not state_dict:
        return []
    dims = []
    layer_idx = 0
    while True:
        weight_key = f"{prefix}.{layer_idx}.weight"
        if weight_key not in state_dict:
            break
        weight = state_dict[weight_key]
        dims.append(weight.shape[0])
        layer_idx += 2  # skip activation modules that follow each Linear
    if not dims:
        return []
    if expected_output_dim is not None and dims[-1] != expected_output_dim:
        return []
    hidden = dims[:-1]
    return hidden


def _override_policy_dims_from_checkpoint(
    train_cfg: GO2HighLevelCfgPPO, checkpoint: dict, num_actions: int
) -> None:
    state_dict = checkpoint.get("actor_critic")
    if not isinstance(state_dict, dict):
        return

    actor_hidden = _infer_hidden_dims_from_state_dict(state_dict, "actor", num_actions)
    if actor_hidden:
        train_cfg.policy.actor_hidden_dims = actor_hidden
        print(f"Inferred actor hidden dims from checkpoint: {actor_hidden}")
    critic_hidden = _infer_hidden_dims_from_state_dict(state_dict, "critic", 1)
    if critic_hidden:
        train_cfg.policy.critic_hidden_dims = critic_hidden
        print(f"Inferred critic hidden dims from checkpoint: {critic_hidden}")


def build_env_and_policy(eval_args: argparse.Namespace):
    """Create the hierarchical env and load the trained policy."""
    sim_args = get_args()
    sim_args.headless = not eval_args.render
    sim_args.num_envs = eval_args.num_envs
    device = torch.device(sim_args.rl_device)

    checkpoint = torch.load(eval_args.checkpoint_path, map_location="cpu")

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2HighLevelCfgPPO()
    env_cfg.env.num_envs = eval_args.num_envs
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, sim_args)
    resolved_low_level = _resolve_low_level_model_path(eval_args, train_cfg, checkpoint)
    train_cfg.runner.low_level_model_path = resolved_low_level

    env = create_env(env_cfg, train_cfg, sim_args, device)
    _override_policy_dims_from_checkpoint(train_cfg, checkpoint, env.num_actions)

    actor_critic = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=train_cfg.policy.init_noise_std,
    ).to(device)

    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

    return env, actor_critic, device, train_cfg


def _snapshot_xy(env: HierarchicalVecEnv) -> torch.Tensor:
    """Return the current XY base positions for every env."""
    return env.env.base_env.base_pos[:, :2].detach().clone()


def collect_trajectories(
    env: HierarchicalVecEnv,
    actor_critic: ActorCritic,
    device: torch.device,
    num_trajs: int,
    max_steps: int,
    max_reset_attempts: int,
) -> List[List[Sequence[float]]]:
    """Roll out until we gather `num_trajs` trajectories."""
    recorded: List[List[Sequence[float]]] = []
    envs_per_batch = env.num_envs
    resets_attempted = 0
    if num_trajs < 1:
        return recorded

    spawn_sequence: List[Sequence[float]] = []
    while len(spawn_sequence) < num_trajs:
        spawn_sequence.extend(FIXED_SPAWN_POINTS)
    spawn_sequence = spawn_sequence[:num_trajs]

    for spawn_xy in spawn_sequence:
        if resets_attempted >= max_reset_attempts:
            break
        resets_attempted += 1
        obs, _, _ = env.reset()
        obs, _, _ = _set_spawn_position(env, spawn_xy)
        obs = obs.to(device)

        batch_paths: List[List[Sequence[float]]] = [[] for _ in range(envs_per_batch)]
        xy = _snapshot_xy(env).cpu().numpy()
        for env_idx in range(envs_per_batch):
            batch_paths[env_idx].append([float(xy[env_idx, 0]), float(xy[env_idx, 1])])

        dones = torch.zeros(envs_per_batch, dtype=torch.bool, device=device)
        for _ in range(max_steps):
            with torch.no_grad():
                actions = actor_critic.act_inference(obs)
            obs, _, _, step_dones, _ = env.step(actions)
            obs = obs.to(device)
            xy = _snapshot_xy(env).cpu().numpy()
            for env_idx in range(envs_per_batch):
                batch_paths[env_idx].append([float(xy[env_idx, 0]), float(xy[env_idx, 1])])
            dones |= step_dones.to(device).bool()
            if torch.all(dones):
                break

        for path in batch_paths:
            if len(recorded) < num_trajs:
                recorded.append(path)

    if len(recorded) < num_trajs:
        raise RuntimeError(
            f"Only collected {len(recorded)} trajectories after {resets_attempted} resets "
            f"(requested {num_trajs}). Increase --max-reset-attempts."
        )
    return recorded


def save_trajectories(
    trajectories: List[List[Sequence[float]]],
    eval_args: argparse.Namespace,
    train_cfg: GO2HighLevelCfgPPO,
) -> Path:
    out_path = Path(eval_args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_path": os.path.abspath(eval_args.checkpoint_path),
        "num_trajs": len(trajectories),
        "max_steps": eval_args.max_steps,
        "low_level_model": train_cfg.runner.low_level_model_path,
        "trajectories": trajectories,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> None:
    eval_args = parse_eval_args()
    env, actor_critic, device, train_cfg = build_env_and_policy(eval_args)
    try:
        trajectories = collect_trajectories(
            env,
            actor_critic,
            device,
            num_trajs=eval_args.num_trajs,
            max_steps=eval_args.max_steps,
            max_reset_attempts=eval_args.max_reset_attempts,
        )
        out_path = save_trajectories(trajectories, eval_args, train_cfg)
        print(f"Saved {len(trajectories)} trajectories to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
