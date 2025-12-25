# 驗證關卡

## 概述

本文件定義三個必須通過的驗證關卡，確保 MJX → PyTorch 轉換流程的正確性。

**原則：** 每個關卡未通過前，不進入下一階段。

---

## Gate 1: Preprocessor 對齊

**時間點：** Week 1 Day 2
**目標：** 確保 JAX Preprocessor 與 NumPy Preprocessor 輸出完全一致

### 測試腳本

```python
# scripts/test_preprocessor_alignment.py

import numpy as np
import jax.numpy as jnp
from training_scripts.main import Preprocessor as NumpyPreprocessor
from training_scripts.preprocessor_jax import PreprocessorJAX

def test_preprocessor_alignment():
    """驗證 JAX 和 NumPy Preprocessor 輸出一致"""

    # 1. 錄製官方環境數據（或使用預錄數據）
    # 建議：錄製 100 個不同的 obs/info 組合
    test_data = np.load("test_data/preprocessor_test_cases.npz")

    np_preprocessor = NumpyPreprocessor()
    jax_preprocessor = PreprocessorJAX()

    max_diff = 0.0

    for i in range(len(test_data['obs'])):
        obs = test_data['obs'][i]
        info = test_data['info'][i]  # 需要正確序列化 info dict

        # NumPy 版本
        np_result = np_preprocessor.modify_state(obs, info)

        # JAX 版本（單一樣本，需要添加 batch 維度）
        jax_result = jax_preprocessor.modify_state(
            jnp.array(obs)[None, :],
            {k: jnp.array(v)[None, :] for k, v in info.items()},
            task_onehot=jnp.zeros((1, 3))  # 根據實際任務調整
        )
        jax_result = np.array(jax_result[0])

        diff = np.abs(np_result - jax_result).max()
        max_diff = max(max_diff, diff)

        if diff > 1e-6:
            print(f"Case {i}: diff = {diff}")
            print(f"  NumPy shape: {np_result.shape}")
            print(f"  JAX shape: {jax_result.shape}")

    print(f"\n最大誤差: {max_diff}")
    assert max_diff < 1e-6, f"Preprocessor 對齊失敗！最大誤差: {max_diff}"
    print("✅ Gate 1 通過：Preprocessor 對齊")

if __name__ == "__main__":
    test_preprocessor_alignment()
```

### 數據錄製腳本

```python
# scripts/record_test_data.py

import numpy as np
import gymnasium as gym
import sai_mujoco  # noqa: F401
import pickle

def record_preprocessor_test_cases(n_cases=100):
    """錄製官方環境的 obs/info 用於對齊測試"""

    env = gym.make("LowerT1GoaliePenaltyKick-v0")

    obs_list = []
    info_list = []

    obs, info = env.reset()

    for _ in range(n_cases):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs.copy())
        # 注意：info 中的值需要是可序列化的 numpy arrays
        info_list.append({k: np.array(v) for k, v in info.items()})

        if terminated or truncated:
            obs, info = env.reset()

    # 保存
    np.savez(
        "test_data/preprocessor_test_cases.npz",
        obs=np.array(obs_list),
    )
    with open("test_data/preprocessor_info.pkl", "wb") as f:
        pickle.dump(info_list, f)

    print(f"✅ 錄製完成：{n_cases} 個測試案例")
    env.close()

if __name__ == "__main__":
    record_preprocessor_test_cases()
```

### 通過標準

| 指標 | 標準 |
|------|------|
| 最大絕對誤差 | < 1e-6 |
| 輸出維度 | 完全一致 |
| 測試案例數 | >= 100 |

### Quaternion 順序驗證（Opus + Gemini 共識）

**問題：** MuJoCo 使用 `[w, x, y, z]` 順序，某些庫可能使用 `[x, y, z, w]`。

**驗證方法：** 讓機器人旋轉 90 度，檢查 quaternion 數值變化。

```python
# scripts/test_quaternion_order.py

import numpy as np
import gymnasium as gym
import sai_mujoco  # noqa: F401

def test_quaternion_order():
    """
    驗證 quaternion 順序是 [w, x, y, z] (MuJoCo 標準)

    測試方法：
    1. 初始狀態應該是 quat ≈ [1, 0, 0, 0] (identity)
    2. 繞 z 軸旋轉 90 度應該是 quat ≈ [0.707, 0, 0, 0.707]
    """
    env = gym.make("LowerT1GoaliePenaltyKick-v0")
    obs, info = env.reset()

    robot_quat = info['robot_quat']
    print(f"Initial quaternion: {robot_quat}")

    # 檢查初始狀態接近 identity
    # MuJoCo [w, x, y, z]: identity = [1, 0, 0, 0]
    expected_identity = np.array([1.0, 0.0, 0.0, 0.0])

    # 允許一些誤差（機器人可能有初始姿態）
    w_component = robot_quat[0]  # 如果是 [w,x,y,z]，這應該接近 1
    print(f"W component (should be close to 1 for upright): {w_component}")

    if abs(w_component) > 0.9:
        print("✅ Quaternion order appears to be [w, x, y, z] (MuJoCo standard)")
    else:
        print("⚠️ Quaternion order may be [x, y, z, w] - need manual verification")
        print("   Check if robot_quat[-1] is close to 1 instead")
        print(f"   Last component: {robot_quat[-1]}")

    env.close()

if __name__ == "__main__":
    test_quaternion_order()
```

**如果順序錯誤，修正方式：**

```python
# preprocessor_jax.py
def quat_rotate_inverse(self, q, v):
    # 如果官方環境使用 [x, y, z, w] 順序，需要調整
    # 假設輸入是 [x, y, z, w]
    # q_w = q[:, -1:]   # w 在最後
    # q_vec = q[:, :3]  # xyz 在前

    # 如果是 MuJoCo 標準 [w, x, y, z]
    q_w = q[:, :1]    # w 在前
    q_vec = q[:, 1:]  # xyz 在後
    # ...
```

---

## Gate 2: 模型輸出對齊

**時間點：** Week 1 Day 4
**目標：** 確保相同權重下，JAX 和 PyTorch 模型輸出一致

### 測試腳本

```python
# scripts/test_model_alignment.py

import numpy as np
import jax
import jax.numpy as jnp
import torch

def test_actor_alignment(jax_actor, jax_params, torch_actor, n_tests=100):
    """驗證 JAX 和 PyTorch Actor 輸出一致"""

    # 1. 確保 PyTorch 模型載入了相同的權重
    # （這需要先執行 jax2torch 轉換）

    max_diff = 0.0

    for seed in range(n_tests):
        # 生成隨機觀測
        np.random.seed(seed)
        obs = np.random.randn(1, 87).astype(np.float32)

        # JAX 前向傳播
        jax_out = jax_actor.apply(jax_params, jnp.array(obs))
        jax_out = np.array(jax_out)

        # PyTorch 前向傳播
        torch_actor.eval()
        with torch.no_grad():
            torch_out = torch_actor(torch.tensor(obs))
            torch_out = torch_out.numpy()

        diff = np.abs(jax_out - torch_out).max()
        max_diff = max(max_diff, diff)

        if diff > 1e-5:
            print(f"Seed {seed}: diff = {diff}")
            print(f"  JAX output: {jax_out[:5]}...")
            print(f"  Torch output: {torch_out[:5]}...")

    print(f"\n最大誤差: {max_diff}")
    assert max_diff < 1e-5, f"模型對齊失敗！最大誤差: {max_diff}"
    print("✅ Gate 2 通過：模型輸出對齊")

def test_weight_transfer():
    """驗證權重轉換的正確性"""

    # 1. 初始化 JAX 模型
    from imitation_learning.utils.networks import MLP
    jax_actor = MLP(input_dim=87, output_dim=12, hidden_dims=[256, 256])

    rng = jax.random.PRNGKey(42)
    jax_params = jax_actor.init(rng, jnp.zeros((1, 87)))

    # 2. 轉換到 PyTorch
    from imitation_learning.scripts.jax2torch import convert_params
    torch_state_dict = convert_params(jax_params)

    # 3. 載入到 PyTorch 模型
    from training_scripts.ddpg import Actor
    torch_actor = Actor(n_features=87, action_dim=12, neurons=[256, 256])
    torch_actor.load_state_dict(torch_state_dict)

    # 4. 驗證
    test_actor_alignment(jax_actor, jax_params, torch_actor)

if __name__ == "__main__":
    test_weight_transfer()
```

### 注意事項

1. **權重初始化差異：** JAX (Flax) 和 PyTorch 的預設初始化不同
   - Flax: `lecun_normal()`
   - PyTorch: `kaiming_uniform()`
   - 解決方案：在 PyTorch 模型中使用相同的初始化，或直接載入 JAX 權重

2. **LayerNorm epsilon：**
   - Flax: `1e-6`
   - PyTorch: `1e-5`
   - 如果使用 LayerNorm，需要統一

### 通過標準

| 指標 | 標準 |
|------|------|
| 最大絕對誤差 | < 1e-5 |
| 測試案例數 | >= 100 |

---

## Gate 3: 行為定性一致

**時間點：** Week 2 Day 5
**目標：** 確保轉換後的 PyTorch 模型在官方環境中表現「定性一致」

### 測試腳本

```python
# scripts/test_behavior_consistency.py

import numpy as np
import gymnasium as gym
import sai_mujoco  # noqa: F401
import torch

def evaluate_episode(env, model, preprocessor, max_steps=500):
    """評估單個 episode"""
    obs, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        processed_obs = preprocessor.modify_state(obs, info)

        with torch.no_grad():
            state = torch.from_numpy(processed_obs).float().unsqueeze(0)
            action = model(state).numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, step + 1

def test_behavior_consistency(model, preprocessor, n_episodes=10):
    """
    定性一致性測試

    標準：
    - 機器人能站立（不在前幾步就倒下）
    - 總體 reward 方向正確（正值為主）
    - 行為穩定（多次運行結果方差合理）
    """

    env = gym.make("LowerT1GoaliePenaltyKick-v0")

    rewards = []
    steps = []

    for ep in range(n_episodes):
        reward, n_steps = evaluate_episode(env, model, preprocessor)
        rewards.append(reward)
        steps.append(n_steps)
        print(f"Episode {ep+1}: reward={reward:.2f}, steps={n_steps}")

    env.close()

    # 定性一致性檢查
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    std_reward = np.std(rewards)

    print(f"\n統計結果:")
    print(f"  平均 reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  平均 steps: {avg_steps:.1f}")

    # 定性標準（根據 MJX 訓練結果調整）
    checks = []

    # 1. 能站立一段時間（至少 100 步）
    checks.append(("站立能力", avg_steps > 100, f"avg_steps={avg_steps:.1f}"))

    # 2. 有正向行為（reward 不是全負）
    checks.append(("正向行為", avg_reward > -100, f"avg_reward={avg_reward:.2f}"))

    # 3. 行為穩定（方差不會太大）
    checks.append(("行為穩定", std_reward < abs(avg_reward) * 2, f"std={std_reward:.2f}"))

    print("\n定性一致性檢查:")
    all_passed = True
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {detail}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n✅ Gate 3 通過：行為定性一致")
    else:
        print("\n❌ Gate 3 未通過，需要調試")

    return all_passed

if __name__ == "__main__":
    from training_scripts.ddpg import DDPG_FF
    from training_scripts.main import Preprocessor

    # 載入轉換後的模型
    model = DDPG_FF(n_features=87, action_space=None, neurons=[256, 256])
    model.actor.load_state_dict(torch.load("model_pretrained.pt")['actor'])

    preprocessor = Preprocessor()

    test_behavior_consistency(model.actor, preprocessor)
```

### 通過標準

| 指標 | 標準 | 說明 |
|------|------|------|
| 平均存活步數 | > 100 | 機器人不會立刻倒下 |
| 平均 reward | > -100 | 有正向行為傾向 |
| Reward 方差 | < 2× 平均值 | 行為穩定 |

**注意：** 這些數值需要根據 MJX 訓練結果調整。如果 MJX 訓練出的 agent 平均 reward 是 500，那轉換後 300-700 都是可接受的。

---

## 驗證流程圖

```
Week 1 Day 2:  Gate 1 (Preprocessor)
                    │
                    ▼ 通過
Week 1 Day 4:  Gate 2 (模型輸出)
                    │
                    ▼ 通過
Week 2 Day 1-3: MJX 大規模訓練
                    │
                    ▼
Week 2 Day 4:  JAX → PyTorch 轉換
                    │
                    ▼
Week 2 Day 5:  Gate 3 (行為一致)
                    │
                    ▼ 通過
Week 3:        官方環境微調 → 提交
```

---

## 常見問題

### Q: Gate 1 失敗，誤差在 1e-3 左右？

常見原因：
1. **維度順序不同：** JAX 是 batch-first，確保 concatenate 順序一致
2. **四元數旋轉實現不同：** 檢查 `quat_rotate_inverse` 的公式

### Q: Gate 2 失敗，誤差在 1e-2 左右？

常見原因：
1. **權重轉置問題：** JAX Dense layer 是 `(in, out)`，PyTorch Linear 是 `(out, in)`
2. **偏置載入錯誤：** 確保 bias 對應正確的 layer

### Q: Gate 3 機器人立刻倒下？

可能原因：
1. **Action 範圍不對：** 檢查是否需要 `tanh` 或其他 scaling
2. **Preprocessor 輸出維度錯誤：** 某些 info 欄位可能缺失

---

## 下一步

所有 Gate 通過後，前往 [03-finetuning-submission.md](./03-finetuning-submission.md) 進行微調。
