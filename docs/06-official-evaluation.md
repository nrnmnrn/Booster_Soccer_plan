# 官方評估函數參考

## 概述

本文件記錄官方 SAI 平台使用的評估函數。這些函數定義了最終排名的計分方式。

**重要：** 獎勵只在 episode 結束（`terminated` 或 `truncated`）時計算一次，中途回傳 0.0。

---

## LowerT1GoaliePenaltyKick-v0

守門員罰球環境。

```python
import numpy as np
_FLOAT_EPS = np.finfo(np.float64).eps

reward_config = {
    "robot_distance_ball": 0.25,
    "ball_vel_twd_goal": 1.5,
    "goal_scored": 2.50,
    "offside": -3.0,
    "ball_hits": -0.2,
    "robot_fallen": -1.5,
    "ball_blocked": -0.5,
    "steps": -1.0,
}

def evaluation_fn(env, eval_state):
    if not eval_state.get("timestep", False):
        eval_state["timestep"] = 0

    raw_reward = env.compute_reward()
    raw_reward.update({"steps": np.float64(1.0)})

    eval_state["timestep"] += 1

    if eval_state.get("terminated", False) or eval_state.get("truncated", False):
        reward = 0.0
        for key, value in raw_reward.items():
            if key in reward_config:
                val = float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
                reward += val * reward_config[key]
        return (reward, eval_state)

    return (0.0, eval_state)
```

---

## LowerT1ObstaclePenaltyKick-v0

障礙物罰球環境。**使用與 GoaliePenaltyKick 相同的評估函數。**

```python
# 與 LowerT1GoaliePenaltyKick-v0 完全相同
reward_config = {
    "robot_distance_ball": 0.25,
    "ball_vel_twd_goal": 1.5,
    "goal_scored": 2.50,
    "offside": -3.0,
    "ball_hits": -0.2,
    "robot_fallen": -1.5,
    "ball_blocked": -0.5,
    "steps": -1.0,
}
```

---

## LowerT1KickToTarget-v0

踢球到目標環境。**獎勵結構不同。**

```python
import numpy as np
_FLOAT_EPS = np.finfo(np.float64).eps

reward_config = {
    "offside": -1.0,
    "success": 2.0,
    "distance": 0.5,
    "steps": -0.3,
}

def evaluation_fn(env, eval_state):
    if not eval_state.get("timestep", False):
        eval_state["timestep"] = 0

    raw_reward = env.compute_reward()
    raw_reward.update({"steps": np.float64(1.0)})

    eval_state["timestep"] += 1

    if eval_state.get("terminated", False) or eval_state.get("truncated", False):
        reward = 0.0
        for key, value in raw_reward.items():
            if key in reward_config:
                val = float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
                reward += val * reward_config[key]
        return (reward, eval_state)

    return (0.0, eval_state)
```

---

## 獎勵指標對比

### Penalty Kick 環境（GoaliePenaltyKick & ObstaclePenaltyKick）

| 指標 | 權重 | 類型 | 說明 |
|------|------|------|------|
| `goal_scored` | +2.50 | boolean | 進球成功 |
| `ball_vel_twd_goal` | +1.50 | float | 球朝目標的速度 |
| `robot_distance_ball` | +0.25 | float | 機器人接近球的距離 |
| `offside` | -3.00 | boolean | 越位判定 |
| `robot_fallen` | -1.50 | boolean | 機器人倒下 |
| `steps` | -1.00 | count | 每步懲罰（步數 × -1.0） |
| `ball_blocked` | -0.50 | boolean | 球被阻擋 |
| `ball_hits` | -0.20 | float | 球碰撞次數 |

### KickToTarget 環境

| 指標 | 權重 | 類型 | 說明 |
|------|------|------|------|
| `success` | +2.00 | boolean | 踢到目標 |
| `distance` | +0.50 | float | 距離目標的距離（越近越高） |
| `offside` | -1.00 | boolean | 越位判定 |
| `steps` | -0.30 | count | 每步懲罰（較輕） |

---

## 策略建議

### 最大化分數的關鍵

1. **不要倒下**
   - `robot_fallen` = -1.5 是很重的懲罰
   - 站立穩定性是基礎

2. **快速完成**
   - `steps` 懲罰存在，拖越久分數越低
   - Penalty Kick: -1.0 × 步數
   - KickToTarget: -0.3 × 步數

3. **避免越位**
   - `offside` = -3.0（Penalty Kick）或 -1.0（KickToTarget）
   - 最重的單一懲罰

4. **進球/成功是最高獎勵**
   - `goal_scored` = +2.5
   - `success` = +2.0

### 環境差異注意

| 環境 | 時間懲罰 | 主要目標 |
|------|----------|----------|
| GoaliePenaltyKick | -1.0/步 | 進球 + 避開守門員 |
| ObstaclePenaltyKick | -1.0/步 | 進球 + 繞過障礙 |
| KickToTarget | -0.3/步 | 精準踢到目標位置 |

---

## 與 MJX 預訓練的關係

```
官方評估（稀疏）          MJX 預訓練（密集）
────────────────          ────────────────
episode 結束計算    →     每步計算
robot_fallen: -1.5  →     r_stand: 每步檢查
steps: -1.0/步      →     r_time: -0.01/步
goal_scored: +2.5   →     r_kick: 接觸球時獎勵
```

MJX 預訓練的目標是讓機器人學會基礎技能（站立、接近球、踢球），官方環境微調再學習環境特定策略。

---

## 資源連結

- [SAI 競賽頁面](https://competesai.com/competitions/cmp_xnSCxcJXQclQ)
- [SAI 文檔](https://docs.competesai.com/getting-started/quick-start)
- [競賽 GitHub](https://github.com/ArenaX-Labs/booster_soccer_showdown)
