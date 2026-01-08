"""JAX SAC 訓練模組

提供 MJX 環境的 SAC 訓練功能。

Usage:
    from src.training import SACConfig, train_sac

    config = SACConfig(total_timesteps=10_000_000)
    train_sac(config)
"""

from .config import SACConfig
from .replay_buffer import ReplayBuffer
from .sac_agent import SACAgent
from .train_sac import train_sac

__all__ = [
    "SACConfig",
    "ReplayBuffer",
    "SACAgent",
    "train_sac",
]
