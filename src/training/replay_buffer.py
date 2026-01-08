"""JAX 原生 Replay Buffer

簡化實現，使用純 JAX 數組，支持 JIT 編譯。
採用 FIFO 循環覆蓋策略。

Features:
- 純 JAX 實現（無外部依賴）
- 支持批量添加（適配並行環境）
- JIT 編譯的 add 和 sample 操作
- 狀態完全由 JAX arrays 表示
"""

from typing import Dict, NamedTuple
import jax
import jax.numpy as jnp


class BufferState(NamedTuple):
    """Replay Buffer 狀態

    所有字段都是 JAX arrays，支持 JIT 和 vmap。

    Attributes:
        obs: 觀察，shape (capacity, obs_dim)
        action: 動作，shape (capacity, action_dim)
        reward: 獎勵，shape (capacity,)
        next_obs: 下一個觀察，shape (capacity, obs_dim)
        done: 終止標誌，shape (capacity,)
        ptr: 當前寫入指針
        size: 當前有效樣本數
    """
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    ptr: jnp.ndarray
    size: jnp.ndarray


class ReplayBuffer:
    """JAX 原生 Replay Buffer

    設計要點：
    1. 無狀態設計：buffer 本身不保存狀態，狀態由 BufferState 攜帶
    2. FIFO 覆蓋：當 buffer 滿時，覆蓋最舊的樣本
    3. JIT 友好：所有操作都可以 JIT 編譯
    4. 批量支持：add_batch 可一次添加多個樣本（適配並行環境）

    Example:
        buffer = ReplayBuffer(capacity=1_000_000, obs_dim=87, action_dim=12)
        state = buffer.init()

        # 添加單個樣本
        state = buffer.add(state, obs, action, reward, next_obs, done)

        # 添加批量樣本（從並行環境）
        state = buffer.add_batch(state, obs_batch, action_batch, ...)

        # 採樣
        batch = buffer.sample(state, batch_size=256, rng=key)
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        """初始化 Buffer 配置

        Args:
            capacity: Buffer 容量
            obs_dim: 觀察維度
            action_dim: 動作維度
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def init(self) -> BufferState:
        """初始化 Buffer 狀態

        Returns:
            空的 BufferState
        """
        return BufferState(
            obs=jnp.zeros((self.capacity, self.obs_dim), dtype=jnp.float32),
            action=jnp.zeros((self.capacity, self.action_dim), dtype=jnp.float32),
            reward=jnp.zeros(self.capacity, dtype=jnp.float32),
            next_obs=jnp.zeros((self.capacity, self.obs_dim), dtype=jnp.float32),
            done=jnp.zeros(self.capacity, dtype=jnp.float32),
            ptr=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    @staticmethod
    @jax.jit
    def add(
        state: BufferState,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_obs: jnp.ndarray,
        done: jnp.ndarray,
    ) -> BufferState:
        """添加單個 transition

        Args:
            state: 當前 buffer 狀態
            obs: 觀察，shape (obs_dim,)
            action: 動作，shape (action_dim,)
            reward: 獎勵，scalar
            next_obs: 下一個觀察，shape (obs_dim,)
            done: 終止標誌，scalar (0.0 或 1.0)

        Returns:
            更新後的 BufferState
        """
        ptr = state.ptr
        capacity = state.obs.shape[0]

        # 更新數據
        new_obs = state.obs.at[ptr].set(obs)
        new_action = state.action.at[ptr].set(action)
        new_reward = state.reward.at[ptr].set(reward)
        new_next_obs = state.next_obs.at[ptr].set(next_obs)
        new_done = state.done.at[ptr].set(done)

        # 更新指針（循環）
        new_ptr = (ptr + 1) % capacity

        # 更新有效大小
        new_size = jnp.minimum(state.size + 1, capacity)

        return BufferState(
            obs=new_obs,
            action=new_action,
            reward=new_reward,
            next_obs=new_next_obs,
            done=new_done,
            ptr=new_ptr,
            size=new_size,
        )

    @staticmethod
    @jax.jit
    def add_batch(
        state: BufferState,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_obs: jnp.ndarray,
        done: jnp.ndarray,
    ) -> BufferState:
        """添加批量 transitions

        用於並行環境，一次添加多個樣本。

        Args:
            state: 當前 buffer 狀態
            obs: 觀察，shape (batch_size, obs_dim)
            action: 動作，shape (batch_size, action_dim)
            reward: 獎勵，shape (batch_size,)
            next_obs: 下一個觀察，shape (batch_size, obs_dim)
            done: 終止標誌，shape (batch_size,)

        Returns:
            更新後的 BufferState
        """
        batch_size = obs.shape[0]
        ptr = state.ptr
        capacity = state.obs.shape[0]

        # 計算新的索引（可能跨越 buffer 邊界）
        indices = (jnp.arange(batch_size) + ptr) % capacity

        # 批量更新（使用 scatter）
        new_obs = state.obs.at[indices].set(obs)
        new_action = state.action.at[indices].set(action)
        new_reward = state.reward.at[indices].set(reward)
        new_next_obs = state.next_obs.at[indices].set(next_obs)
        new_done = state.done.at[indices].set(done)

        # 更新指針
        new_ptr = (ptr + batch_size) % capacity

        # 更新有效大小
        new_size = jnp.minimum(state.size + batch_size, capacity)

        return BufferState(
            obs=new_obs,
            action=new_action,
            reward=new_reward,
            next_obs=new_next_obs,
            done=new_done,
            ptr=new_ptr,
            size=new_size,
        )

    @staticmethod
    @jax.jit
    def sample(
        state: BufferState,
        batch_size: int,
        rng: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """從 buffer 中隨機採樣

        Args:
            state: 當前 buffer 狀態
            batch_size: 採樣數量
            rng: JAX random key

        Returns:
            包含 obs, action, reward, next_obs, done 的字典
        """
        # 從有效範圍內採樣索引
        indices = jax.random.randint(
            rng, shape=(batch_size,), minval=0, maxval=state.size
        )

        return {
            "obs": state.obs[indices],
            "action": state.action[indices],
            "reward": state.reward[indices],
            "next_obs": state.next_obs[indices],
            "done": state.done[indices],
        }

    def can_sample(self, state: BufferState, batch_size: int) -> bool:
        """檢查是否有足夠樣本可採樣

        Args:
            state: Buffer 狀態
            batch_size: 需要的樣本數

        Returns:
            是否可以採樣
        """
        return int(state.size) >= batch_size


# =============================================================================
# 測試工具
# =============================================================================

def test_replay_buffer():
    """測試 ReplayBuffer 功能"""
    print("Testing ReplayBuffer...")

    buffer = ReplayBuffer(capacity=100, obs_dim=87, action_dim=12)
    state = buffer.init()

    # 測試單個添加
    key = jax.random.PRNGKey(0)
    for i in range(50):
        key, subkey = jax.random.split(key)
        obs = jax.random.normal(subkey, (87,))
        action = jax.random.uniform(subkey, (12,), minval=-1, maxval=1)
        reward = jnp.array(1.0)
        next_obs = jax.random.normal(subkey, (87,))
        done = jnp.array(0.0)

        state = buffer.add(state, obs, action, reward, next_obs, done)

    assert int(state.size) == 50, f"Expected size 50, got {state.size}"
    assert int(state.ptr) == 50, f"Expected ptr 50, got {state.ptr}"
    print(f"✓ Single add: size={state.size}, ptr={state.ptr}")

    # 測試批量添加
    batch_obs = jax.random.normal(key, (30, 87))
    batch_action = jax.random.uniform(key, (30, 12), minval=-1, maxval=1)
    batch_reward = jnp.ones(30)
    batch_next_obs = jax.random.normal(key, (30, 87))
    batch_done = jnp.zeros(30)

    state = buffer.add_batch(state, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
    assert int(state.size) == 80, f"Expected size 80, got {state.size}"
    print(f"✓ Batch add: size={state.size}, ptr={state.ptr}")

    # 測試採樣
    key, sample_key = jax.random.split(key)
    batch = buffer.sample(state, batch_size=32, rng=sample_key)
    assert batch["obs"].shape == (32, 87), f"Expected (32, 87), got {batch['obs'].shape}"
    assert batch["action"].shape == (32, 12), f"Expected (32, 12), got {batch['action'].shape}"
    print(f"✓ Sample: obs={batch['obs'].shape}, action={batch['action'].shape}")

    # 測試循環覆蓋
    state = buffer.add_batch(
        state,
        jax.random.normal(key, (50, 87)),
        jax.random.uniform(key, (50, 12)),
        jnp.ones(50),
        jax.random.normal(key, (50, 87)),
        jnp.zeros(50),
    )
    assert int(state.size) == 100, f"Expected size 100 (capacity), got {state.size}"
    print(f"✓ Circular overwrite: size={state.size}")

    print("✅ All ReplayBuffer tests passed!")


if __name__ == "__main__":
    test_replay_buffer()
