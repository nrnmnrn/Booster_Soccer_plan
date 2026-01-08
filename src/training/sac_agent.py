"""JAX SAC Agent

Soft Actor-Critic 實現，使用 Flax 和 Optax。
網路結構與 booster_soccer_showdown/imitation_learning 保持一致，
確保與 jax2torch.py 轉換腳本兼容。

Features:
- Twin Q-Networks（雙 Critic 減少過估計）
- Automatic Entropy Tuning（自動調整 temperature）
- Tanh-squashed Gaussian Policy
- Soft Target Update
- JIT 編譯的 update 函數

References:
- SAC 論文: https://arxiv.org/abs/1801.01290
- SAC v2 (自動 alpha): https://arxiv.org/abs/1812.05905
"""

from typing import Dict, Tuple, Any, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax

from .config import SACConfig


# =============================================================================
# 網路定義
# =============================================================================

class MLP(nn.Module):
    """多層感知機（與 imitation_learning/utils/networks.py 一致）

    Attributes:
        hidden_dims: 隱藏層維度列表
        activate_final: 是否在最後一層後激活
        use_layer_norm: 是否使用 LayerNorm
    """
    hidden_dims: Tuple[int, ...]
    activate_final: bool = True
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim)(x)

            # 最後一層可能不激活
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                x = nn.relu(x)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)

        return x


class GaussianActor(nn.Module):
    """高斯策略 Actor（與 GCActor 結構一致）

    輸出 Tanh-squashed Gaussian 分佈。
    網路結構與 jax2torch.py 兼容：
    - actor_net: MLP backbone
    - mean_net: 輸出均值
    - log_std_net: 輸出 log 標準差

    Attributes:
        action_dim: 動作維度
        hidden_dims: 隱藏層維度
        log_std_min: log_std 下界（CLAUDE.md: -5.0）
        log_std_max: log_std 上界（CLAUDE.md: 2.0）
    """
    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> distrax.Distribution:
        # Backbone（命名與 jax2torch 一致）
        x = MLP(
            hidden_dims=self.hidden_dims,
            activate_final=True,
            use_layer_norm=True,
            name="actor_net"
        )(obs)

        # 輸出頭（命名與 jax2torch 一致）
        mean = nn.Dense(self.action_dim, name="mean_net")(x)
        log_std = nn.Dense(self.action_dim, name="log_std_net")(x)

        # Clip log_std（CLAUDE.md 約束）
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        # 創建 Gaussian 分佈
        base_dist = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.exp(log_std)
        )

        # Tanh squashing（將動作限制在 [-1, 1]）
        return distrax.Transformed(
            distribution=base_dist,
            bijector=distrax.Block(distrax.Tanh(), ndims=1)
        )

    def get_action_and_log_prob(
        self,
        obs: jnp.ndarray,
        rng: jnp.ndarray,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """獲取動作和 log probability

        Args:
            obs: 觀察
            rng: 隨機數 key
            deterministic: 是否確定性選擇（使用 mean）

        Returns:
            (action, log_prob)
        """
        dist = self(obs)

        if deterministic:
            # 確定性：使用 mean（經過 tanh）
            action = dist.mode()
            log_prob = dist.log_prob(action)
        else:
            # 隨機：從分佈採樣
            action = dist.sample(seed=rng)
            log_prob = dist.log_prob(action)

        return action, log_prob


class Critic(nn.Module):
    """Q-Network（輸入 obs+action，輸出 Q 值）

    Attributes:
        hidden_dims: 隱藏層維度
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # 拼接 obs 和 action
        x = jnp.concatenate([obs, action], axis=-1)

        # MLP
        x = MLP(
            hidden_dims=self.hidden_dims,
            activate_final=True,
            use_layer_norm=True
        )(x)

        # 輸出 Q 值（標量）
        q = nn.Dense(1)(x)

        return q.squeeze(-1)


class DoubleCritic(nn.Module):
    """Twin Q-Networks（SAC 標準配置）

    Attributes:
        hidden_dims: 隱藏層維度
    """
    hidden_dims: Tuple[int, ...] = (256, 256, 256)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Q1
        q1 = Critic(hidden_dims=self.hidden_dims, name="critic1")(obs, action)

        # Q2（獨立參數）
        q2 = Critic(hidden_dims=self.hidden_dims, name="critic2")(obs, action)

        return q1, q2


# =============================================================================
# SAC Agent
# =============================================================================

class SACState(NamedTuple):
    """SAC 訓練狀態

    Attributes:
        actor: Actor TrainState
        critic: Critic TrainState
        target_critic_params: Target Critic 參數
        log_alpha: log(entropy temperature)
        alpha_optimizer_state: Alpha 優化器狀態
    """
    actor: TrainState
    critic: TrainState
    target_critic_params: Any
    log_alpha: jnp.ndarray
    alpha_optimizer_state: Any


class SACAgent:
    """Soft Actor-Critic Agent

    Example:
        config = SACConfig()
        agent = SACAgent(config)

        # 初始化
        rng = jax.random.PRNGKey(0)
        sac_state = agent.init(rng)

        # 選擇動作
        action = agent.select_action(sac_state, obs, rng)

        # 更新
        sac_state, info = agent.update(sac_state, batch, rng)
    """

    def __init__(self, config: SACConfig):
        self.config = config

        # 創建網路
        self.actor = GaussianActor(
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        )

        self.critic = DoubleCritic(
            hidden_dims=config.hidden_dims,
        )

        # Alpha 優化器
        self.alpha_optimizer = optax.adam(config.alpha_lr)

    def init(self, rng: jnp.ndarray) -> SACState:
        """初始化 SAC 狀態

        Args:
            rng: JAX random key

        Returns:
            SACState
        """
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # 創建 dummy inputs
        dummy_obs = jnp.zeros((1, self.config.obs_dim))
        dummy_action = jnp.zeros((1, self.config.action_dim))

        # 初始化 Actor
        actor_params = self.actor.init(actor_rng, dummy_obs)
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(self.config.actor_lr),
        )

        # 初始化 Critic
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_action)
        critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(self.config.critic_lr),
        )

        # Target Critic（硬拷貝）
        target_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        # 初始化 log_alpha
        log_alpha = jnp.log(self.config.init_alpha)
        alpha_optimizer_state = self.alpha_optimizer.init(log_alpha)

        return SACState(
            actor=actor_state,
            critic=critic_state,
            target_critic_params=target_critic_params,
            log_alpha=log_alpha,
            alpha_optimizer_state=alpha_optimizer_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: SACState,
        obs: jnp.ndarray,
        rng: jnp.ndarray,
        deterministic: bool = False
    ) -> jnp.ndarray:
        """選擇動作

        Args:
            state: SAC 狀態
            obs: 觀察，shape (obs_dim,) 或 (batch_size, obs_dim)
            rng: JAX random key
            deterministic: 是否確定性選擇

        Returns:
            動作，shape (action_dim,) 或 (batch_size, action_dim)
        """
        # 確保 obs 是 2D
        squeeze_output = False
        if obs.ndim == 1:
            obs = obs[None, :]
            squeeze_output = True

        # 獲取分佈
        dist = state.actor.apply_fn(state.actor.params, obs)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample(seed=rng)

        if squeeze_output:
            action = action.squeeze(0)

        return action

    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: SACState,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> Tuple[SACState, Dict[str, float]]:
        """SAC 更新

        執行順序：
        1. 更新 Critic（TD learning）
        2. 更新 Actor（Policy gradient）
        3. 更新 Alpha（Entropy tuning）
        4. Soft update Target Critic

        Args:
            state: SAC 狀態
            batch: 包含 obs, action, reward, next_obs, done 的字典
            rng: JAX random key

        Returns:
            (new_state, info_dict)
        """
        rng, critic_rng, actor_rng, alpha_rng = jax.random.split(rng, 4)

        # 當前 alpha
        alpha = jnp.exp(state.log_alpha)

        # ===== 1. Update Critic =====
        new_critic_state, critic_info = self._update_critic(
            state, batch, alpha, critic_rng
        )

        # ===== 2. Update Actor =====
        new_actor_state, actor_info = self._update_actor(
            state, new_critic_state, batch, alpha, actor_rng
        )

        # ===== 3. Update Alpha =====
        new_log_alpha, new_alpha_opt_state, alpha_info = self._update_alpha(
            state, new_actor_state, batch, alpha_rng
        )

        # ===== 4. Soft Update Target =====
        new_target_params = self._soft_update_target(
            state.target_critic_params,
            new_critic_state.params
        )

        # 組合新狀態
        new_state = SACState(
            actor=new_actor_state,
            critic=new_critic_state,
            target_critic_params=new_target_params,
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt_state,
        )

        # 合併 info
        info = {**critic_info, **actor_info, **alpha_info}

        return new_state, info

    def _update_critic(
        self,
        state: SACState,
        batch: Dict[str, jnp.ndarray],
        alpha: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """更新 Critic"""

        def critic_loss_fn(critic_params):
            # 當前 Q 值
            q1, q2 = state.critic.apply_fn(
                critic_params, batch["obs"], batch["action"]
            )

            # 計算 target Q
            # 1. 從當前 policy 採樣 next action
            next_dist = state.actor.apply_fn(state.actor.params, batch["next_obs"])
            next_action = next_dist.sample(seed=rng)
            next_log_prob = next_dist.log_prob(next_action)

            # 2. 使用 target critic 計算 next Q
            next_q1, next_q2 = state.critic.apply_fn(
                state.target_critic_params, batch["next_obs"], next_action
            )
            next_q = jnp.minimum(next_q1, next_q2)

            # 3. TD target（包含 entropy bonus）
            target_q = batch["reward"] + self.config.gamma * (1.0 - batch["done"]) * (
                next_q - alpha * next_log_prob
            )
            target_q = jax.lax.stop_gradient(target_q)

            # MSE loss
            loss_q1 = jnp.mean((q1 - target_q) ** 2)
            loss_q2 = jnp.mean((q2 - target_q) ** 2)
            loss = loss_q1 + loss_q2

            info = {
                "critic/loss": loss,
                "critic/q1_mean": jnp.mean(q1),
                "critic/q2_mean": jnp.mean(q2),
                "critic/target_q_mean": jnp.mean(target_q),
            }

            return loss, info

        (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            state.critic.params
        )

        new_critic_state = state.critic.apply_gradients(grads=grads)

        return new_critic_state, info

    def _update_actor(
        self,
        state: SACState,
        new_critic_state: TrainState,
        batch: Dict[str, jnp.ndarray],
        alpha: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """更新 Actor"""

        def actor_loss_fn(actor_params):
            # 從當前 policy 採樣動作
            dist = state.actor.apply_fn(actor_params, batch["obs"])
            action = dist.sample(seed=rng)
            log_prob = dist.log_prob(action)

            # 計算 Q 值（使用更新後的 critic）
            q1, q2 = new_critic_state.apply_fn(
                new_critic_state.params, batch["obs"], action
            )
            q = jnp.minimum(q1, q2)

            # Actor loss：最大化 Q - alpha * log_prob
            # 等價於最小化 alpha * log_prob - Q
            loss = jnp.mean(alpha * log_prob - q)

            info = {
                "actor/loss": loss,
                "actor/entropy": -jnp.mean(log_prob),
            }

            return loss, info

        (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            state.actor.params
        )

        new_actor_state = state.actor.apply_gradients(grads=grads)

        return new_actor_state, info

    def _update_alpha(
        self,
        state: SACState,
        new_actor_state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any, Dict[str, float]]:
        """更新 entropy temperature (alpha)"""

        def alpha_loss_fn(log_alpha):
            alpha = jnp.exp(log_alpha)

            # 計算當前 policy 的 entropy
            dist = new_actor_state.apply_fn(new_actor_state.params, batch["obs"])
            action = dist.sample(seed=rng)
            log_prob = dist.log_prob(action)

            # Alpha loss：讓 entropy 接近 target_entropy
            # L = -alpha * (log_prob + target_entropy)
            loss = -jnp.mean(alpha * (log_prob + self.config.target_entropy))

            return loss

        loss, grads = jax.value_and_grad(alpha_loss_fn)(state.log_alpha)

        updates, new_alpha_opt_state = self.alpha_optimizer.update(
            grads, state.alpha_optimizer_state
        )
        new_log_alpha = optax.apply_updates(state.log_alpha, updates)

        info = {
            "alpha/loss": loss,
            "alpha/value": jnp.exp(new_log_alpha),
            "alpha/log_alpha": new_log_alpha,
        }

        return new_log_alpha, new_alpha_opt_state, info

    def _soft_update_target(
        self,
        target_params: Any,
        online_params: Any,
    ) -> Any:
        """Soft update target network"""
        return jax.tree.map(
            lambda t, o: (1.0 - self.config.tau) * t + self.config.tau * o,
            target_params,
            online_params,
        )

    def get_checkpoint(self, state: SACState) -> Dict[str, Any]:
        """獲取 checkpoint（與 jax2torch 兼容的格式）

        保存格式對齊 jax2torch.py 的期望：
        checkpoint['agent']['network']['params']['modules_actor']

        Args:
            state: SAC 狀態

        Returns:
            Checkpoint 字典
        """
        return {
            "agent": {
                "network": {
                    "params": {
                        # jax2torch.py 期望的路徑
                        "modules_actor": state.actor.params["params"],
                    }
                }
            },
            "critic": state.critic.params,
            "target_critic": state.target_critic_params,
            "log_alpha": state.log_alpha,
            "alpha_optimizer_state": state.alpha_optimizer_state,
            "config": self.config.to_dict(),
        }

    def load_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        rng: jnp.ndarray,
    ) -> SACState:
        """從 checkpoint 恢復

        Args:
            checkpoint: Checkpoint 字典
            rng: JAX random key

        Returns:
            SACState
        """
        # 先初始化獲取結構
        state = self.init(rng)

        # 恢復參數
        actor_params = {"params": checkpoint["agent"]["network"]["params"]["modules_actor"]}
        new_actor_state = state.actor.replace(params=actor_params)

        new_critic_state = state.critic.replace(params=checkpoint["critic"])

        return SACState(
            actor=new_actor_state,
            critic=new_critic_state,
            target_critic_params=checkpoint["target_critic"],
            log_alpha=checkpoint["log_alpha"],
            alpha_optimizer_state=checkpoint["alpha_optimizer_state"],
        )


# =============================================================================
# 測試
# =============================================================================

def test_sac_agent():
    """測試 SACAgent"""
    print("Testing SACAgent...")

    config = SACConfig(
        num_envs=4,
        hidden_dims=(64, 64),  # 小型網路用於測試
    )
    agent = SACAgent(config)

    # 初始化
    rng = jax.random.PRNGKey(0)
    state = agent.init(rng)
    print(f"✓ Initialized SAC state")

    # 測試 select_action
    rng, action_rng = jax.random.split(rng)
    obs = jax.random.normal(action_rng, (87,))
    action = agent.select_action(state, obs, action_rng)
    assert action.shape == (12,), f"Expected (12,), got {action.shape}"
    print(f"✓ select_action: {action.shape}")

    # 測試批量 select_action
    batch_obs = jax.random.normal(action_rng, (8, 87))
    batch_action = agent.select_action(state, batch_obs, action_rng)
    assert batch_action.shape == (8, 12), f"Expected (8, 12), got {batch_action.shape}"
    print(f"✓ batch select_action: {batch_action.shape}")

    # 測試 update
    batch = {
        "obs": jax.random.normal(rng, (32, 87)),
        "action": jax.random.uniform(rng, (32, 12), minval=-1, maxval=1),
        "reward": jax.random.normal(rng, (32,)),
        "next_obs": jax.random.normal(rng, (32, 87)),
        "done": jax.random.bernoulli(rng, 0.1, (32,)).astype(jnp.float32),
    }

    rng, update_rng = jax.random.split(rng)
    new_state, info = agent.update(state, batch, update_rng)

    print(f"✓ update: critic/loss={info['critic/loss']:.4f}, actor/loss={info['actor/loss']:.4f}")

    # 測試 checkpoint
    ckpt = agent.get_checkpoint(new_state)
    assert "agent" in ckpt, "Checkpoint 缺少 'agent' key"
    assert "modules_actor" in ckpt["agent"]["network"]["params"], "Checkpoint 格式不兼容 jax2torch"
    print(f"✓ checkpoint 格式正確（jax2torch 兼容）")

    print("✅ All SACAgent tests passed!")


if __name__ == "__main__":
    test_sac_agent()
