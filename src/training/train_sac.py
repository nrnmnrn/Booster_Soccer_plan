"""SAC 訓練主腳本

提供完整的 MJX 環境 SAC 訓練流程。

Features:
- 支持並行環境（num_envs）
- W&B + MLflow 雙重追蹤
- 定期 checkpoint 保存（與 jax2torch 兼容）
- 訓練進度恢復
- Domain Randomization
- Task index 隨機化

Usage:
    # 在 Databricks 中
    from src.training import SACConfig, train_sac

    config = SACConfig(total_timesteps=10_000_000)
    final_state, checkpoint_path = train_sac(config)

    # 命令行
    python -m src.training.train_sac --total_timesteps 10000000
"""

import os
import time
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp

# 可選依賴
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed, W&B logging disabled")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("Warning: mlflow not installed, MLflow logging disabled")

from .config import SACConfig
from .sac_agent import SACAgent, SACState
from .replay_buffer import ReplayBuffer, BufferState

# 延遲導入 MJX 環境（避免循環導入）
MJXSoccerEnv = None


def _get_env_class():
    """延遲導入 MJX 環境類"""
    global MJXSoccerEnv
    if MJXSoccerEnv is None:
        from ..mjx.soccer_env import MJXSoccerEnv as _MJXSoccerEnv
        MJXSoccerEnv = _MJXSoccerEnv
    return MJXSoccerEnv


# =============================================================================
# Logging
# =============================================================================

def setup_logging(config: SACConfig) -> Optional[str]:
    """設置 W&B 和 MLflow

    Args:
        config: 訓練配置

    Returns:
        run_id（用於 checkpoint 目錄命名）
    """
    run_id = f"sac_{int(time.time())}"

    if config.use_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            config=config.to_dict(),
            name=f"sac_{config.total_timesteps // 1_000_000}M_dr{config.dr_level}",
        )
        run_id = wandb.run.id

    if config.use_mlflow and HAS_MLFLOW:
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=f"sac_dr{config.dr_level}")
        mlflow.log_params(config.to_dict())

    return run_id


def log_metrics(metrics: Dict[str, float], step: int, config: SACConfig):
    """記錄指標到 W&B 和 MLflow"""
    if config.use_wandb and HAS_WANDB:
        wandb.log(metrics, step=step)

    if config.use_mlflow and HAS_MLFLOW:
        for key, value in metrics.items():
            # MLflow 不支持 "/" 在 metric name 中
            mlflow_key = key.replace("/", "_")
            mlflow.log_metric(mlflow_key, float(value), step=step)


def finish_logging(config: SACConfig):
    """結束 logging sessions"""
    if config.use_wandb and HAS_WANDB:
        wandb.finish()

    if config.use_mlflow and HAS_MLFLOW:
        mlflow.end_run()


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    agent: SACAgent,
    state: SACState,
    step: int,
    config: SACConfig,
    run_id: str,
    is_final: bool = False,
) -> str:
    """保存 checkpoint（與 jax2torch 兼容）

    Args:
        agent: SACAgent 實例
        state: 當前 SAC 狀態
        step: 當前步數
        config: 訓練配置
        run_id: 運行 ID
        is_final: 是否為最終 checkpoint

    Returns:
        checkpoint 路徑
    """
    checkpoint_dir = Path(config.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_final:
        filename = "final_checkpoint.pkl"
    else:
        filename = f"checkpoint_{step}.pkl"

    checkpoint_path = checkpoint_dir / filename

    # 獲取 jax2torch 兼容的 checkpoint
    checkpoint = agent.get_checkpoint(state)
    checkpoint["step"] = step
    checkpoint["run_id"] = run_id

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"✓ Checkpoint saved: {checkpoint_path}")

    # 上傳到 MLflow
    if config.use_mlflow and HAS_MLFLOW:
        mlflow.log_artifact(str(checkpoint_path))

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    agent: SACAgent,
    rng: jnp.ndarray,
) -> Tuple[SACState, int]:
    """載入 checkpoint

    Args:
        checkpoint_path: Checkpoint 文件路徑
        agent: SACAgent 實例
        rng: JAX random key

    Returns:
        (SACState, step)
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    state = agent.load_checkpoint(checkpoint, rng)
    step = checkpoint.get("step", 0)

    print(f"✓ Loaded checkpoint from step {step}")

    return state, step


# =============================================================================
# Training Loop
# =============================================================================

def train_sac(
    config: SACConfig,
    resume_from: Optional[str] = None,
) -> Tuple[SACState, str]:
    """SAC 訓練主循環

    Args:
        config: 訓練配置
        resume_from: 可選的 checkpoint 路徑，用於恢復訓練

    Returns:
        (final_state, checkpoint_path)
    """
    print("=" * 60)
    print("SAC Training on MJX Soccer Environment")
    print("=" * 60)
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Num envs: {config.num_envs}")
    print(f"DR level: {config.dr_level}")
    print(f"Random task index: {config.random_task_index}")
    print("=" * 60)

    # 設置 MuJoCo 環境變數（Databricks 需要）
    os.environ.setdefault("MUJOCO_GL", "disabled")

    # 設置 logging
    run_id = setup_logging(config)

    # 初始化 RNG
    rng = jax.random.PRNGKey(config.seed)
    rng, env_rng, agent_rng, buffer_rng = jax.random.split(rng, 4)

    # 創建環境
    print("\n[1/4] Creating MJX environment...")
    EnvClass = _get_env_class()
    env = EnvClass(
        num_envs=config.num_envs,
        max_steps=config.max_episode_steps,
    )
    print(f"✓ Environment created: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    # 創建 Agent
    print("\n[2/4] Creating SAC agent...")
    agent = SACAgent(config)

    # 初始化或恢復 Agent
    if resume_from:
        sac_state, start_step = load_checkpoint(resume_from, agent, agent_rng)
    else:
        sac_state = agent.init(agent_rng)
        start_step = 0
    print(f"✓ SAC agent initialized")

    # 創建 Replay Buffer
    print("\n[3/4] Creating replay buffer...")
    buffer = ReplayBuffer(
        capacity=config.buffer_size,
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
    )
    buffer_state = buffer.init()
    print(f"✓ Replay buffer created: capacity={config.buffer_size:,}")

    # Reset 環境
    print("\n[4/4] Resetting environment...")
    env_state, obs = env.reset(env_rng)
    print(f"✓ Environment reset: obs.shape={obs.shape}")

    # 訓練統計
    episode_rewards = jnp.zeros(config.num_envs)
    episode_lengths = jnp.zeros(config.num_envs, dtype=jnp.int32)
    episode_count = 0
    total_episodes_completed = 0

    # 開始訓練
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")

    start_time = time.time()
    global_step = start_step

    while global_step < config.total_timesteps:
        # ===== 1. 收集數據 =====
        rng, action_rng = jax.random.split(rng)

        # 選擇動作
        if global_step < config.learning_starts:
            # 隨機探索
            action = jax.random.uniform(
                action_rng,
                (config.num_envs, config.action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        else:
            # 使用 policy
            action = agent.select_action(sac_state, obs, action_rng, deterministic=False)

        # 執行環境 step
        env_state, next_obs, reward, done, info = env.step(env_state, action)

        # 添加到 buffer（批量）
        buffer_state = buffer.add_batch(
            buffer_state,
            obs,
            action,
            reward,
            next_obs,
            done.astype(jnp.float32),
        )

        # 更新 episode 統計
        episode_rewards = episode_rewards + reward
        episode_lengths = episode_lengths + 1

        # 處理 episode 結束
        finished_mask = done.astype(bool)
        if jnp.any(finished_mask):
            finished_count = int(jnp.sum(finished_mask))
            total_episodes_completed += finished_count

            # 計算完成 episodes 的統計
            finished_rewards = episode_rewards[finished_mask]
            finished_lengths = episode_lengths[finished_mask]

            avg_reward = float(jnp.mean(finished_rewards))
            avg_length = float(jnp.mean(finished_lengths))

            # 記錄
            log_metrics({
                "train/episode_reward": avg_reward,
                "train/episode_length": avg_length,
                "train/episodes_completed": total_episodes_completed,
            }, global_step, config)

            # 重置已完成 episodes 的統計
            episode_rewards = jnp.where(finished_mask, 0.0, episode_rewards)
            episode_lengths = jnp.where(finished_mask, 0, episode_lengths)

        # 更新 obs
        obs = next_obs
        global_step += config.num_envs

        # ===== 2. 更新 Agent =====
        if global_step >= config.learning_starts and buffer.can_sample(buffer_state, config.batch_size):
            for _ in range(config.updates_per_step):
                rng, sample_rng, update_rng = jax.random.split(rng, 3)

                # 採樣 batch
                batch = buffer.sample(buffer_state, config.batch_size, sample_rng)

                # SAC 更新
                sac_state, update_info = agent.update(sac_state, batch, update_rng)

            # 記錄訓練指標
            if global_step % config.log_frequency == 0:
                elapsed_time = time.time() - start_time
                sps = global_step / elapsed_time  # Steps per second

                metrics = {
                    "train/critic_loss": float(update_info["critic/loss"]),
                    "train/actor_loss": float(update_info["actor/loss"]),
                    "train/alpha": float(update_info["alpha/value"]),
                    "train/entropy": float(update_info["actor/entropy"]),
                    "train/q1_mean": float(update_info["critic/q1_mean"]),
                    "perf/sps": sps,
                    "perf/buffer_size": int(buffer_state.size),
                    "perf/elapsed_hours": elapsed_time / 3600,
                }

                log_metrics(metrics, global_step, config)

                # 進度輸出
                progress = global_step / config.total_timesteps * 100
                eta_hours = (config.total_timesteps - global_step) / sps / 3600 if sps > 0 else 0
                print(
                    f"Step {global_step:,}/{config.total_timesteps:,} ({progress:.1f}%) | "
                    f"SPS: {sps:.0f} | ETA: {eta_hours:.1f}h | "
                    f"Episodes: {total_episodes_completed} | "
                    f"Q: {update_info['critic/q1_mean']:.2f} | "
                    f"α: {update_info['alpha/value']:.3f}"
                )

        # ===== 3. 評估 =====
        if global_step % config.eval_frequency == 0 and global_step > 0:
            eval_metrics = evaluate_policy(
                agent, sac_state, env, config, rng
            )
            log_metrics(eval_metrics, global_step, config)
            print(f"  [Eval] Mean reward: {eval_metrics['eval/mean_reward']:.2f}")

        # ===== 4. 保存 Checkpoint =====
        if global_step % config.save_frequency == 0 and global_step > 0:
            save_checkpoint(agent, sac_state, global_step, config, run_id)

    # 訓練結束
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # 保存最終 checkpoint
    final_path = save_checkpoint(agent, sac_state, global_step, config, run_id, is_final=True)

    # 結束 logging
    finish_logging(config)

    # 打印總結
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 3600:.2f} hours")
    print(f"Total episodes: {total_episodes_completed}")
    print(f"Average SPS: {global_step / total_time:.0f}")
    print(f"Final checkpoint: {final_path}")

    return sac_state, final_path


def evaluate_policy(
    agent: SACAgent,
    state: SACState,
    env,
    config: SACConfig,
    rng: jnp.ndarray,
) -> Dict[str, float]:
    """評估當前策略（高效並行版本）

    使用所有並行環境進行評估，充分利用 GPU 資源。
    每個環境運行一個完整 episode，收集所有環境的 reward 統計。

    Args:
        agent: SACAgent
        state: SACState
        env: MJX 環境
        config: 訓練配置
        rng: JAX random key

    Returns:
        評估指標字典
    """
    rng, reset_rng = jax.random.split(rng)

    # 重置所有環境
    env_state, obs = env.reset(reset_rng)

    # 追蹤每個環境的累積 reward 和完成狀態
    num_envs = config.num_envs
    episode_rewards = jnp.zeros(num_envs)
    episode_done = jnp.zeros(num_envs, dtype=jnp.bool_)

    for step in range(config.max_episode_steps):
        rng, action_rng = jax.random.split(rng)

        # 為所有環境生成確定性動作（批量處理）
        # 使用 vmap 對所有觀察並行計算動作
        action_keys = jax.random.split(action_rng, num_envs)
        actions = jax.vmap(
            lambda o, k: agent.select_action(state, o, k, deterministic=True)
        )(obs, action_keys)

        # 環境 step
        env_state, obs, reward, done, info = env.step(env_state, actions)

        # 累積 reward（只對未完成的環境）
        episode_rewards = episode_rewards + reward * (~episode_done)

        # 更新完成狀態
        episode_done = episode_done | done

        # 如果所有環境都完成，提前結束
        if jnp.all(episode_done):
            break

    # 計算統計（只使用前 eval_episodes 個環境的結果，或全部）
    n_eval = min(config.eval_episodes, num_envs)
    eval_rewards = episode_rewards[:n_eval]

    return {
        "eval/mean_reward": float(jnp.mean(eval_rewards)),
        "eval/std_reward": float(jnp.std(eval_rewards)),
        "eval/min_reward": float(jnp.min(eval_rewards)),
        "eval/max_reward": float(jnp.max(eval_rewards)),
        "eval/num_episodes": n_eval,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Train SAC on MJX Soccer Environment")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--dr_level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="exp/sac_mjx/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_mlflow", action="store_true")

    args = parser.parse_args()

    config = SACConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        dr_level=args.dr_level,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
        use_mlflow=not args.no_mlflow,
    )

    train_sac(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
