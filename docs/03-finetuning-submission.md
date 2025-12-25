# 微調與提交

## 概述

本文件說明如何將 JAX 預訓練模型轉換為 PyTorch，在官方環境微調，並提交到 SAI 平台。

---

## JAX → PyTorch 轉換

### ⚠️ 重要：SAC → DDPG 架構差異

**警告：** 現有的 `imitation_learning/scripts/jax2torch.py` 是為 IL 模型設計的，**不適用於 SAC Actor 轉換！**

| 來源 | Actor 輸出維度 | 說明 |
|------|---------------|------|
| JAX SAC Actor | 24 (12 mean + 12 log_std) | SAC 需要輸出 mean 和 std |
| PyTorch DDPG Actor | 12 | DDPG 只需要 deterministic action |

**必須使用下方 `convert_sac_actor_to_ddpg()` 函數，不能直接使用現有腳本！**

### 驗證轉換

```python
import torch

# 載入轉換後的權重
model = torch.load("model_pretrained.pt")

# 檢查 layer 名稱和維度
print("Keys:", model.keys())
for key, value in model.items():
    print(f"{key}: {value.shape}")
```

### SAC → DDPG 權重轉換（Critical）

**問題：架構不匹配**
- SAC Actor 輸出: `action_dim * 2` = 24（12 mean + 12 log_std）
- DDPG Actor 輸出: `action_dim` = 12（只有 action）

**解決方案：只取 mean 部分，捨棄 log_std**

```python
# jax2torch.py 修改範例
import torch
import numpy as np

def convert_sac_actor_to_ddpg(jax_params):
    """
    將 JAX SAC Actor 參數轉換為 PyTorch DDPG 格式

    關鍵：SAC 最後一層輸出 24 維（mean + log_std）
         DDPG 最後一層只需要 12 維（mean）
    """
    state_dict = {}

    # 假設網路結構: Dense_0 (87→256) → Dense_1 (256→256) → Dense_2 (256→24)
    layer_names = ['Dense_0', 'Dense_1', 'Dense_2']

    for i, name in enumerate(layer_names):
        kernel = jax_params['params'][name]['kernel']  # JAX: (in, out)
        bias = jax_params['params'][name]['bias']

        # === 最後一層特殊處理：只取 mean 部分 ===
        if i == len(layer_names) - 1:
            # SAC 最後一層: (256, 24) → 取前 12 維
            kernel = kernel[:, :12]  # (256, 12)
            bias = bias[:12]          # (12,)

        # JAX kernel 是 (in, out)，PyTorch 是 (out, in)
        state_dict[f'layers.{i}.weight'] = torch.from_numpy(kernel.T.copy())
        state_dict[f'layers.{i}.bias'] = torch.from_numpy(bias.copy())

    return state_dict

# 使用範例
import pickle
with open("exp/mjx_sac/checkpoint.pkl", "rb") as f:
    jax_params = pickle.load(f)

state_dict = convert_sac_actor_to_ddpg(jax_params['actor'])
torch.save({'actor': state_dict}, "model_pretrained.pt")
```

### 驗證轉換正確性

```python
# 確認維度匹配
state_dict = torch.load("model_pretrained.pt")['actor']
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

# 預期輸出：
# layers.0.weight: torch.Size([256, 87])
# layers.0.bias: torch.Size([256])
# layers.1.weight: torch.Size([256, 256])
# layers.1.bias: torch.Size([256])
# layers.2.weight: torch.Size([12, 256])  # 注意：12 而不是 24！
# layers.2.bias: torch.Size([12])
```

---

## 官方環境驗證

### 載入預訓練權重

```python
import torch
import gymnasium as gym
import sai_mujoco  # noqa: F401

from training_scripts.ddpg import DDPG_FF
from training_scripts.main import Preprocessor

# 建立環境
env = gym.make("LowerT1GoaliePenaltyKick-v0")

# 建立模型
model = DDPG_FF(
    n_features=87,
    action_space=env.action_space,
    neurons=[256, 256],
    learning_rate=3e-4
)

# 載入預訓練權重（只載入 Actor）
pretrained = torch.load("model_pretrained.pt")
model.actor.load_state_dict(pretrained['actor'], strict=False)

print("預訓練權重載入成功！")
```

### 測試行為

```python
preprocessor = Preprocessor()

obs, info = env.reset()
total_reward = 0

for step in range(500):
    # 預處理觀測
    processed_obs = preprocessor.modify_state(obs, info)

    # 選擇動作
    with torch.no_grad():
        state = torch.from_numpy(processed_obs).float().unsqueeze(0)
        action = model(state).numpy().squeeze()

    # 執行動作
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Episode reward: {total_reward}")
env.close()
```

**預期行為：**
- 機器人應能站立
- 機器人應能走向球
- 可能還不會精準踢球（需要微調）

---

## 官方環境微調

### 修改 main.py 支援預訓練載入

```python
# training_scripts/main.py 添加

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default=None,
                    help='Path to pretrained weights')
parser.add_argument('--timesteps', type=int, default=1000000,
                    help='Training timesteps')
args = parser.parse_args()

# 建立模型後載入預訓練權重
model = DDPG_FF(...)

if args.pretrained:
    pretrained = torch.load(args.pretrained)
    model.actor.load_state_dict(pretrained['actor'], strict=False)
    print(f"Loaded pretrained from {args.pretrained}")

# 開始訓練
training_loop(env, model, action_function, Preprocessor, timesteps=args.timesteps)
```

### 執行微調

```bash
python training_scripts/main.py \
  --pretrained ./model_pretrained.pt \
  --timesteps 200000
```

**微調策略：**
- 使用較小的 learning rate（原本的 1/10）
- 減少探索噪聲
- 監控 reward 曲線確保不退化

---

## Feature Freeze 策略（Opus + Gemini 交互審查）

### 問題 1：災難性遺忘

直接全網路微調可能破壞預訓練學到的「特徵提取」能力（如何理解關節狀態）。

### 問題 2：Critic 錯配（Gemini 審查補充）

> 天才 Actor 遇上白癡 Critic

- 預訓練的 Actor 已經是「足球高手」
- 但 DDPG 的 Critic 是隨機初始化的（不懂球的教練）
- 初期 Critic 會給出錯誤的 Q 值評估，導致 Actor 在前幾輪更新中被「洗腦」
- 風險：Policy Collapse（遺忘預訓練技能）

### 解決方案：四階段 Warmup

```
Stage 0 (0-10k):    完全凍結 Actor，只訓練 Critic  ← 新增！
Stage 1 (10k-30k):  只訓練 Actor 最後層
Stage 2 (30k-60k):  解凍 Actor 倒數第二層
Stage 3 (60k+):     全網路微調
```

```python
class FeatureFreezeScheduler:
    """
    四階段漸進式解凍策略（Opus + Gemini 共識）

    Stage 0 (0-10k):    完全凍結 Actor，只訓練 Critic
    Stage 1 (10k-30k):  只訓練 Actor 最後一層
    Stage 2 (30k-60k):  解凍 Actor 倒數第二層
    Stage 3 (60k+):     全網路訓練（降低 learning rate）
    """

    def __init__(self, model, actor_optimizer, critic_optimizer,
                 stage0_steps=10000, stage1_steps=30000, stage2_steps=60000):
        self.model = model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.stage0_steps = stage0_steps
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.current_stage = 0

        # Stage 0: 完全凍結 Actor
        self._freeze_actor(freeze_all=True)
        print("Stage 0: Actor frozen, training Critic only")

    def _freeze_actor(self, freeze_all=False, except_last=0):
        """凍結 Actor 網路層"""
        layers = list(self.model.actor.children())
        for i, layer in enumerate(layers):
            if freeze_all:
                freeze = True
            else:
                freeze = i < len(layers) - except_last
            for param in layer.parameters():
                param.requires_grad = not freeze

    def get_actor_lr(self, step):
        """動態調整 Actor 學習率"""
        if step < self.stage0_steps:
            return 0.0  # Stage 0: 完全凍結
        elif step < self.stage1_steps:
            return 1e-5  # Stage 1: 低學習率
        elif step < self.stage2_steps:
            return 3e-5  # Stage 2: 中等學習率
        else:
            return 3e-4  # Stage 3: 正常學習率

    def step(self, current_step):
        """根據訓練步數更新凍結策略"""
        # Stage 0 → Stage 1
        if current_step >= self.stage0_steps and self.current_stage < 1:
            self._freeze_actor(except_last=1)  # 只解凍最後一層
            self.current_stage = 1
            print(f"Step {current_step}: Stage 1 - Training last layer only")

        # Stage 1 → Stage 2
        elif current_step >= self.stage1_steps and self.current_stage < 2:
            self._freeze_actor(except_last=2)  # 解凍倒數兩層
            self.current_stage = 2
            print(f"Step {current_step}: Stage 2 - Unfreezing penultimate layer")

        # Stage 2 → Stage 3
        elif current_step >= self.stage2_steps and self.current_stage < 3:
            self._freeze_actor(except_last=len(list(self.model.actor.children())))
            self.current_stage = 3
            print(f"Step {current_step}: Stage 3 - Full network training")

        # 更新 Actor 學習率
        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = self.get_actor_lr(current_step)
```

### 使用方式

```python
# training_scripts/main.py 修改
from feature_freeze import FeatureFreezeScheduler

# 建立模型
model = DDPG_FF(...)

# 分離 Actor 和 Critic 優化器
actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.0)  # Stage 0 從 0 開始
critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)

# 載入預訓練權重（只有 Actor）
if args.pretrained:
    pretrained = torch.load(args.pretrained)
    model.actor.load_state_dict(pretrained['actor'], strict=False)
    print("Loaded pretrained Actor weights")

# 建立 Feature Freeze Scheduler
scheduler = FeatureFreezeScheduler(
    model, actor_optimizer, critic_optimizer,
    stage0_steps=10_000,   # Critic 預熱
    stage1_steps=30_000,   # 只訓練最後層
    stage2_steps=60_000    # 漸進解凍
)

# 訓練循環
for step in range(total_steps):
    # 更新凍結策略和學習率
    scheduler.step(step)

    # ... 正常 DDPG 訓練 ...
```

### 為什麼這樣設計？

| 階段 | 目的 | 效果 |
|------|------|------|
| Stage 0 | 讓 Critic 學會評估預訓練 Actor 的行為 | 避免「白癡教練」給錯誤梯度 |
| Stage 1 | 讓輸出層適應 DDPG 的 Tanh 和新物理反饋 | 保護底層特徵提取 |
| Stage 2 | 逐步調整特徵空間 | 平滑過渡 |
| Stage 3 | 全網路微調以最大化性能 | 完全適應新環境 |

---

## Reward Annealing（Gemini 審查核心建議）

### 問題：Sparse Reward 陷阱

官方環境只在 episode 結束時給獎勵（Sparse），而 MJX 預訓練使用每步獎勵（Dense）。
直接切換會導致：
- Value Network 崩潰（回報信號突然消失）
- Agent 喪失方向感

### 解決方案：混合獎勵退火

```python
class RewardAnnealer:
    """
    混合獎勵退火 - 從 Dense 逐漸切換到 Official Sparse
    """
    def __init__(self, total_steps, warmup_ratio=0.3):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

    def get_reward(self, step, dense_reward, official_reward):
        if step < self.warmup_steps:
            # 早期：主要使用 dense reward
            alpha = 1.0 - (step / self.warmup_steps) * 0.9  # 1.0 → 0.1
            beta = 0.1 + (step / self.warmup_steps) * 0.9   # 0.1 → 1.0
        else:
            # 後期：完全使用 official reward
            alpha = 0.1  # 保留微量 dense 作為引導
            beta = 1.0

        return alpha * dense_reward + beta * official_reward
```

### 使用方式

```python
# 在 training_loop 中
annealer = RewardAnnealer(total_steps=200_000, warmup_ratio=0.3)

for step in range(total_steps):
    # 獲取環境獎勵
    obs, official_reward, terminated, truncated, info = env.step(action)

    # 計算 dense reward（使用 MJX 訓練時的獎勵函數）
    dense_reward = compute_dense_reward(obs, info, prev_action, action)

    # 混合獎勵
    reward = annealer.get_reward(step, dense_reward, official_reward)
```

### 行為約束（替代 Anchor Reward）

Gemini 原本建議保留部分 Dense Reward 作為 Anchor。
**Opus 反對**：可能導致任務衝突（MJX locomotion vs 官方任務）。
**最終方案**：使用行為約束而非獎勵錨點。

```python
class FineTuneReward:
    def compute(self, env, obs, action, prev_action=None):
        # 官方稀疏獎勵
        sparse = env.official_reward()

        # 行為約束（負向懲罰，不是正向獎勵）
        standing_penalty = -1.0 if env.robot_fallen else 0.0

        # 防止完全靜止（Policy Collapse 預防）
        idle_penalty = -0.1 if np.allclose(action, 0, atol=0.01) else 0.0

        # Action Smoothness（防止高頻震盪）
        smoothness_penalty = 0.0
        if prev_action is not None:
            delta = action - prev_action
            smoothness_penalty = -0.01 * np.sum(delta ** 2)

        return sparse + standing_penalty + idle_penalty + smoothness_penalty
```

---

## 本地評估

### 測試所有環境

```bash
# 守門員環境
python training_scripts/test.py --env LowerT1GoaliePenaltyKick-v0

# 踢球到目標環境
python training_scripts/test.py --env LowerT1KickToTarget-v0

# 障礙物環境
python training_scripts/test.py --env LowerT1ObstaclePenaltyKick-v0
```

### 評估指標

- **Episode Reward:** 越高越好
- **Success Rate:** 達成目標的比例
- **穩定性:** 多次運行結果的方差

---

## SAI 提交

### 關鍵發現：SAI 接受自定義 Preprocessor

從 `imitation_learning/submission/submit_sai.py` 分析：
```python
sai.submit(name="baseline", model=model, preprocessor_class=Preprocessor)
```

**這意味著**：
- 我們可以提交使用 87 維 Preprocessor 的模型
- 不需要使用官方 IL 的 30 維 Preprocessor
- 只需在提交時指定我們自己的 Preprocessor 類

### 建立 SAC 提交腳本

建立文件 `training_scripts/submit_sac.py`：

```python
import os
import numpy as np
from sai_rl import SAIClient

# 初始化 SAI client
sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")
env = sai.make_env()

# === 使用 87 維 Preprocessor（與訓練一致）===
class Preprocessor87D:
    """87 維 Preprocessor - 與 training_scripts/main.py 完全一致"""

    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)
        return a - b + c

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)

        # ... (與 main.py Preprocessor 完全一致)
        # 輸出 87 維 observation

        return obs

# 載入訓練好的 SAC 模型
from model import SACModel
model = SACModel(model_path="./exp/final_sac_actor.pt")

# 本地評估
sai.benchmark(model, preprocessor_class=Preprocessor87D)

# 提交
sai.submit(name="sac-v1", model=model, preprocessor_class=Preprocessor87D)
```

### 執行提交

```bash
python training_scripts/submit_sac.py
```

### 提交後

1. 在 [SAI 平台](https://competesai.com/competitions/cmp_xnSCxcJXQclQ) 查看分數
2. 分析排行榜位置
3. 根據結果調整策略

---

## 迭代優化

### 根據提交結果調整

| 問題 | 可能原因 | 解決方案 |
|------|----------|----------|
| 分數低於 baseline | Preprocessor 不一致 | 驗證 JAX/NumPy 輸出一致性 |
| 特定環境差 | 泛化不足 | 針對該環境額外微調 |
| 行為不穩定 | Sim-to-Sim Gap | 增加官方環境微調步數 |
| 機器人倒下 | 站立獎勵不足 | 調整 MJX 獎勵權重 |

### 進階優化

1. **多環境訓練：** 在微調時混合多個環境
2. **Domain Randomization：** 在 MJX 中隨機化物理參數
3. **Curriculum Learning：** 從簡單任務逐步增加難度

---

## 下一步

如需整合更多工具（W&B, Optuna），請參考 [04-tooling-integration.md](./04-tooling-integration.md)。
