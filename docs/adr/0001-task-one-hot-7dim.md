# ADR-0001: MJX 環境 task_one_hot 使用 7 維編碼

## 狀態

已接受

## 日期

2026-01-08

## 背景

在 Databricks 上執行 SAC 訓練時，遇到維度不匹配錯誤：

```
Incompatible shapes for broadcasting: (2048, 83) and requested shape (2048, 87)
```

**問題分析**：
- MJX 環境產生 **83 維** observation
- SAC 訓練期望 **87 維**（官方 `main.py:94` 使用 `n_features=87`）
- 差距：**4 維**

**維度組成**：
```
固定部分（包含 2 維 player_team）：80 維
task_one_hot：原 3 維 → 需要 7 維
總計：80 + 7 = 87 維 ✓
```

**考慮過的替代方案**：
1. 添加額外 padding（不語義化）
2. 修改 player_team 維度（會破壞官方格式）
3. 擴展 task_one_hot 至 7 維（保持語義，易於微調）

## 決策

我們決定將 `task_one_hot` 從 3 維擴展至 7 維：
- 前 3 維保持原有任務 one-hot 語義
- 後 4 維填充 0，作為預留空間

```python
# 任務編碼（7 維）
task_index=0 (GoaliePK)     → [1,0,0,0,0,0,0]
task_index=1 (ObstaclePK)   → [0,1,0,0,0,0,0]
task_index=2 (KickToTarget) → [0,0,1,0,0,0,0]
```

## 理由

1. **符合官方維度**：80 + 7 = 87 維，與官方 `n_features=87` 一致
2. **保持語義一致**：前 3 維的任務 one-hot 語義與官方 `preprocessor.py` 一致
3. **向後兼容**：額外的 0 填充不影響模型學習，微調時可調整
4. **最小改動**：只需修改 task_one_hot 初始化，不影響其他組件

## 後果

### 正面影響

- 解決維度不匹配錯誤，SAC 訓練可正常執行
- 預訓練模型可順利轉換為 PyTorch 進行微調

### 負面影響

- task_one_hot 語義可能與官方實際格式不完全一致
- 如果官方 `info['task_index']` 維度不同，微調時需要調整

### 中性影響

- 後 4 維填 0 在預訓練階段不會被學習到有意義的表徵
- JAX → PyTorch 轉換不受影響（只轉換 actor weights）

## 遇到的問題與解決

| 問題 | 解決方案 | 日期 |
|------|----------|------|
| sai_mujoco body name bug 無法驗證官方維度 | 採用理論分析：80 + 7 = 87 | 2026-01-08 |

## 相關文件

- `src/mjx/preprocessor_jax.py` - JAX 版本 preprocessor
- `src/mjx/soccer_env.py` - MJX 足球環境
- `booster_soccer_showdown/training_scripts/main.py:94` - 官方 n_features=87
- `docs/troubleshooting.md` - Preprocessor 維度問題
