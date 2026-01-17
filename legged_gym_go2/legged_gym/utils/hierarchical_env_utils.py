import torch

from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env


class HierarchicalVecEnv:
    """Thin wrapper to expose a vectorized interface for the hierarchical GO2 env."""

    def __init__(self, env: HierarchicalGO2Env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.device = env.device
        self.num_privileged_obs = None

    def reset(self) -> torch.Tensor:
        return self.env.reset()

    def step(self, actions: torch.Tensor):
        obs, rewards, dones, infos = self.env.step(actions)
        return obs, rewards, dones, infos

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
