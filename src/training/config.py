"""SAC 訓練配置

使用 dataclass 提供類型安全的配置管理。
所有超參數都在這裡定義，便於實驗追蹤和復現。

## 實驗筆記（2026-01-09）

### 問題診斷：Entropy Collapse

第一次 180k 步訓練（Session 7）發現以下問題：
- `train/episode_reward`：44.5-47.5 波動，**無上升趨勢**
- `train/entropy`：8.2 → 7.6 持續下降
- `log_alpha`：-4.9（α ≈ 0.007），entropy 權重過低
- `eval/std_reward`：0.28，行為過於確定性

### 根本原因
`target_entropy = -12.0`（= -action_dim）導致 SAC 過早將策略收斂到確定性行為，
停止探索，陷入局部最優。

### 解決方案（方案 A：保守調整）
| 參數 | 原值 | 新值 | 原因 |
|------|------|------|------|
| target_entropy | -12.0 | -6.0 | 保持更高 entropy（-dim/2） |
| init_alpha | 0.2 | 1.0 | 初始探索更多 |
| actor_lr | 3e-4 | 1e-4 | 穩定 actor 學習 |
| updates_per_step | 4 | 2 | 減少過擬合當前數據 |

### 理論背景
- 12 維 Gaussian 的初始 entropy ≈ 8.4
- target_entropy = -6：允許中等隨機性
- target_entropy = -12：幾乎無隨機性（過於確定）
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class SACConfig:
    """SAC 訓練完整配置

    Attributes:
        # 環境參數
        num_envs: 並行環境數量（預設 512，JIT 編譯通過後可嘗試 1024）
        max_episode_steps: 每個 episode 最大步數
        obs_dim: 觀察維度（87，CLAUDE.md 約束）
        action_dim: 動作維度（12，下肢關節）

        # SAC 超參數
        gamma: 折扣因子
        tau: Target network soft update 係數
        actor_lr: Actor 學習率
        critic_lr: Critic 學習率
        alpha_lr: Entropy temperature 學習率
        target_entropy: 目標 entropy（自動設為 -action_dim）

        # 網路架構
        hidden_dims: 隱藏層維度（與 jax2torch 兼容）

        # 訓練流程
        total_timesteps: 總訓練步數
        learning_starts: 開始學習前的隨機探索步數
        batch_size: 每次更新的 batch size
        buffer_size: Replay Buffer 容量

        # Domain Randomization
        dr_level: DR 強度級別（1=基礎, 2=進階, 3=激進）
        random_task_index: 是否隨機化 task_index

        # Checkpoint & Logging
        save_frequency: 保存 checkpoint 頻率（步數）
        log_frequency: 記錄日誌頻率（步數）
        eval_frequency: 評估頻率（步數）
        checkpoint_dir: Checkpoint 保存目錄

        # 監控
        use_wandb: 是否使用 W&B
        use_mlflow: 是否使用 MLflow
        wandb_project: W&B 專案名稱
        mlflow_experiment: MLflow 實驗名稱
    """

    # === 環境參數 ===
    num_envs: int = 32  # 降低以避免 L4 GPU JIT 編譯時 OOM
    max_episode_steps: int = 1000
    obs_dim: int = 87  # CLAUDE.md 約束：固定 87 維
    action_dim: int = 12  # 12 個下肢關節

    # === SAC 超參數 ===
    gamma: float = 0.99
    tau: float = 0.005

    # 學習率設定
    # [實驗發現] actor_lr=3e-4 時 Q1_mean 先升後降，actor 學習不穩定
    actor_lr: float = 1e-4  # 降低以穩定 actor 學習（原 3e-4）
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # Entropy 調節（解決 Entropy Collapse 的關鍵）
    # [實驗發現] target_entropy=-12 導致 log_alpha 降到 -4.9，策略過早確定性收斂
    # [理論] 12 維 Gaussian 初始 entropy ≈ 8.4，target=-6 允許中等隨機性
    target_entropy: float = -6.0  # 提高目標（-dim/2），避免過早收斂（原 -12）
    init_alpha: float = 1.0  # 增加初始探索（原 0.2）

    # === 網路架構 ===
    # 與 booster_soccer_showdown/imitation_learning/utils/networks.py 一致
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    log_std_min: float = -5.0  # CLAUDE.md 約束
    log_std_max: float = 2.0   # CLAUDE.md 約束

    # === 訓練流程 ===
    total_timesteps: int = 10_000_000
    learning_starts: int = 10_000  # 前 10k 步隨機探索
    batch_size: int = 512  # 增大 batch 減少梯度噪聲（原 256）
    buffer_size: int = 1_000_000

    # 樣本效率 vs 過擬合權衡
    # [實驗發現] updates_per_step=4 時 episode_reward 無上升趨勢，可能過擬合當前數據
    # [理論] 較低的 UTD（Update-To-Data ratio）讓策略有更多時間收集多樣化經驗
    updates_per_step: int = 2  # 減少過擬合（原 4）

    # === Domain Randomization ===
    dr_level: int = 1  # 1=基礎(±5%), 2=進階(±10%), 3=激進(±20%)
    random_task_index: bool = True  # 每次 reset 隨機選擇任務

    # === Checkpoint & Logging ===
    # [實驗發現] save_frequency=500k 時，140k 步訓練崩潰後無 checkpoint
    # 建議：設為訓練總步數的 1/4 ~ 1/3
    save_frequency: int = 50_000  # 更頻繁保存，避免訓練損失（原 500k）
    log_frequency: int = 1_000     # 每 1k 步記錄
    eval_frequency: int = 50_000   # 每 50k 步評估
    eval_episodes: int = 10        # 評估時運行的 episode 數
    checkpoint_dir: str = "exp/sac_mjx/checkpoints"

    # === 監控 ===
    use_wandb: bool = True
    use_mlflow: bool = False  # Databricks MLflow 需要絕對路徑，暫時禁用（WandB 已足夠）
    wandb_project: str = "booster_soccer_mjx"
    mlflow_experiment: str = "/Users/adamlin@cheerstech.com.tw/mjx_sac_training"  # Databricks 需要絕對路徑

    # === 隨機種子 ===
    seed: int = 42

    def __post_init__(self):
        """初始化後處理"""
        # 驗證 CLAUDE.md 約束
        assert self.obs_dim == 87, f"obs_dim 必須為 87（CLAUDE.md 約束），got {self.obs_dim}"
        assert self.action_dim == 12, f"action_dim 必須為 12，got {self.action_dim}"
        assert self.log_std_min == -5.0, "log_std_min 必須為 -5.0（CLAUDE.md 約束）"
        assert self.log_std_max == 2.0, "log_std_max 必須為 2.0（CLAUDE.md 約束）"

    def get_dr_config(self) -> dict:
        """根據 DR 級別返回隨機化參數範圍"""
        configs = {
            1: {  # 基礎
                "mass_range": (0.95, 1.05),
                "friction_range": (0.9, 1.1),
                "damping_range": (0.95, 1.05),
                "obs_noise_std": 0.005,
            },
            2: {  # 進階
                "mass_range": (0.9, 1.1),
                "friction_range": (0.7, 1.3),
                "damping_range": (0.85, 1.15),
                "obs_noise_std": 0.02,
            },
            3: {  # 激進
                "mass_range": (0.8, 1.2),
                "friction_range": (0.5, 1.5),
                "damping_range": (0.8, 1.2),
                "obs_noise_std": 0.03,
            },
        }
        return configs.get(self.dr_level, configs[1])

    def to_dict(self) -> dict:
        """轉換為字典（用於 logging）"""
        return {
            "num_envs": self.num_envs,
            "max_episode_steps": self.max_episode_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "alpha_lr": self.alpha_lr,
            "target_entropy": self.target_entropy,
            "init_alpha": self.init_alpha,
            "hidden_dims": self.hidden_dims,
            "total_timesteps": self.total_timesteps,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "dr_level": self.dr_level,
            "random_task_index": self.random_task_index,
            "seed": self.seed,
        }
