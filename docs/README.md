# Booster Soccer Showdown - 競賽開發計劃

## 競賽資訊

| 項目 | 內容 |
|------|------|
| **競賽名稱** | Booster Soccer Showdown |
| **獎金池** | $10,000 USD |
| **截止日期** | **2026 年 1 月 9 日（五）** |
| **平台** | [SAI](https://competesai.com/competitions/cmp_xnSCxcJXQclQ) |
| **核心挑戰** | 訓練**單一策略**在三個環境中泛化（不允許 per-task tuning） |

### 三個競賽環境

| 環境 | 任務 |
|------|------|
| `LowerT1GoaliePenaltyKick-v0` | 守門員罰球 |
| `LowerT1ObstaclePenaltyKick-v0` | 障礙物罰球 |
| `LowerT1KickToTarget-v0` | 踢球到目標 |

### 官方評估獎勵（終局計算）

**GoaliePenaltyKick & ObstaclePenaltyKick：**

| 指標 | 權重 | 說明 |
|------|------|------|
| `goal_scored` | +2.50 | 進球 |
| `ball_vel_twd_goal` | +1.50 | 球朝目標速度 |
| `robot_distance_ball` | +0.25 | 接近球 |
| `robot_fallen` | **-1.50** | 倒下（重懲罰） |
| `offside` | -3.00 | 越位 |
| `ball_blocked` | -0.50 | 球被阻擋 |
| `ball_hits` | -0.20 | 球碰撞 |
| `steps` | -1.00 | 時間懲罰 |

**KickToTarget：**

| 指標 | 權重 |
|------|------|
| `success` | +2.00 |
| `distance` | +0.50 |
| `offside` | -1.00 |
| `steps` | -0.30 |

> **重要：** 獎勵只在 episode 結束（`terminated` 或 `truncated`）時計算，中途回傳 0.0。

---

## 策略概述

**MJX 預訓練 + PyTorch 微調**（整合 Opus + Gemini 建議）

```
核心流程：
mimic/assets XML → MJX (GPU) → 簡化獎勵預訓練 → jax2torch → 官方環境微調 → 提交
```

**硬體：** Databricks L4 GPU (24GB VRAM)

---

## 文件結構

| 文件 | 內容 |
|------|------|
| [01-environment-setup.md](./01-environment-setup.md) | 環境設置（Databricks + 本地 + Unity Catalog） |
| [02-mjx-training.md](./02-mjx-training.md) | MJX 環境建立 + JAX SAC 訓練 + Domain Randomization |
| [03-finetuning-submission.md](./03-finetuning-submission.md) | PyTorch 微調 + SAI 提交 |
| [04-tooling-integration.md](./04-tooling-integration.md) | 工具整合（W&B + MLflow 分工, GPU 監控, Optuna） |
| [05-verification-gates.md](./05-verification-gates.md) | **驗證關卡（必讀）** |
| [06-official-evaluation.md](./06-official-evaluation.md) | **官方評估函數參考** |
| [07-databricks-mlops.md](./07-databricks-mlops.md) | **Databricks MLOps（Jobs, Unity Catalog, 監控）** |

---

## 時間軸

### Week 1: 基礎建設 + MJX 環境 + Databricks MLOps

| Day | 任務 | 產出 | 驗證關卡 |
|-----|------|------|----------|
| 1 | Databricks 環境 + **Unity Catalog 設置** | JAX/MJX/W&B/MLflow 可用, Volumes 建立 | - |
| 2 | Preprocessor JAX 翻譯 + **Job 1 (Setup)** | `preprocessor_jax.py`, Workflow 定義 | **Gate 1** |
| 3 | 獎勵函數 + Domain Randomization + **GPU 監控** | `rewards.py`, `gpu_monitor.py` | - |
| 4-5 | JAX SAC + Brax 整合 + **Job 2 (Pre-train)** | `mjx_env.py`, `train_mjx_sac.py` | **Gate 2** |

### Week 2: 預訓練 + 轉換 + Workflow 自動化

| Day | 任務 | 產出 | 驗證關卡 |
|-----|------|------|----------|
| 1-3 | MJX 大規模預訓練 (10M 步) + **W&B/MLflow 雙重記錄** | `checkpoint.pkl`, Unity Catalog 模型 | - |
| 4 | JAX → PyTorch 轉換 + **Job 3 (Conversion)** | `model_pretrained.pt`, 自動轉換流程 | - |
| 5 | 官方環境驗證 + **Job 4 (Fine-tune)** | 確認行為定性一致 | **Gate 3** |

### Week 3: 微調 + 提交 + 端到端 Pipeline

| Day | 任務 | 產出 |
|-----|------|------|
| 1-3 | 官方環境微調 + **完整 Workflow 測試** | 優化後模型, Pipeline 驗證 |
| 4 | SAI 提交 + **Job 5 (Submit)** | 競賽分數, 自動提交流程 |
| 5 | 迭代優化 + **模型版本比較** | 改進版模型, MLflow 追蹤 |

---

## 關鍵優化（整合 Opus + Gemini 審查）

### 1. 直接使用現有 XML（省 2 天）

```python
# 不需要從 sai_mujoco 提取！
xml_path = "mimic/assets/booster_t1/booster_lower_t1.xml"
```

### 2. 87 維 Preprocessor（統一標準）

- 使用完整 87 維 observation（與 DDPG 訓練一致）
- SAI 接受自定義 Preprocessor，無需使用 IL 的 30 維版本
- 詳見 `training_scripts/main.py` 的 `Preprocessor` 類

### 3. Task Index 隨機化（Gemini 最佳建議）

在 MJX 環境 `reset()` 時隨機注入 `task_onehot`：
- 即使物理場景相同，網路也能學會對 task_index 敏感
- 微調時，網路已準備好根據任務調整行為

```python
# mjx_env.py reset()
task_id = jax.random.randint(rng, (), 0, 3)
task_onehot = jax.nn.one_hot(task_id, 3)  # 注入到 observation
```

### 4. Reward Annealing（微調策略）

微調時使用混合獎勵退火：
```
R_total = α × R_dense + β × R_official
開始：α=1.0, β=0.1
結束：α=0.1, β=1.0
```

### 5. Action Smoothness（獎勵項）

```python
r_smoothness = -0.01 * jnp.sum((action - prev_action) ** 2)
```

### 6. Domain Randomization 分級

| Level | 適用情況 | 配置 |
|-------|----------|------|
| Level 1 | 基礎訓練 | friction ±10%, obs_noise 0.005 |
| Level 2 | Gate 3 失敗 | friction ±30%, mass ±10% |
| Level 3 | 最後手段 | friction ±50%, 球/腳特別強化 |

---

## 需要建立的新文件

```
training_scripts/
├── mjx_env.py              # MJX 環境封裝（Brax 介面層）
├── preprocessor_jax.py     # JAX 版 Preprocessor
├── rewards.py              # 簡化獎勵函數 + Domain Randomization
├── train_mjx_sac.py        # JAX SAC 訓練腳本
└── main.py                 # 修改：支援 pretrained 載入

scripts/
├── test_preprocessor_alignment.py  # Gate 1: Preprocessor 對齊測試
├── test_model_alignment.py         # Gate 2: 模型輸出對齊測試
├── test_behavior_consistency.py    # Gate 3: 行為定性一致測試
└── record_test_data.py             # 錄製測試數據

test_data/
├── preprocessor_test_cases.npz     # Preprocessor 測試數據
└── preprocessor_info.pkl           # Info dict 測試數據

databricks/                         # [NEW] Databricks MLOps
├── workflows/                      # Databricks Jobs/Workflows
│   ├── 01_environment_setup.py     # Job 1: 環境設置 + 驗證
│   ├── 02_mjx_pretraining.py       # Job 2: MJX SAC 預訓練
│   ├── 03_model_conversion.py      # Job 3: JAX → PyTorch
│   ├── 04_pytorch_finetuning.py    # Job 4: DDPG 微調
│   └── 05_sai_submission.py        # Job 5: SAI 提交
├── monitoring/
│   └── gpu_monitor.py              # GPU 監控 + 警報
├── integrations/
│   └── dual_logger.py              # W&B + MLflow 統一接口
└── config/
    ├── cluster_config.json         # Cluster 規格定義
    └── job_definitions.json        # Workflow 編排
```

---

## 風險與檢查點（更新版）

| 時間點 | 檢查項目 | 失敗回退 |
|--------|----------|----------|
| **Day 1** | MJX 物理對齊測試（誤差不指數發散） | 調整 timestep/solver 參數 |
| Day 2 | XML 能載入 MJX？ | 修改 XML 移除不支援功能 |
| Day 2 | Preprocessor JAX 翻譯正確？ | 手動逐行對比 |
| Week 1 結束 | MJX 環境能跑？ | 切換路線 A（純 PyTorch） |
| Week 2 Day 3 | SAC 收斂？ | 先用 DDPG 驗證環境 |
| **Day 11** | jax2torch log_std 驗證 | 檢查數值範圍 [-5, 2] |
| Week 2 Day 5 | 權重載入後行為正常？ | 檢查 key mapping |

### 新增風險（Gemini 審查識別）

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| log_std 轉換錯誤 | 中 | 高 | Gate 3 + 數值範圍檢查 |
| Policy Collapse（躺著不動） | 低 | 中 | 行為約束 + idle_penalty |
| Action 高頻震盪 | 中 | 中 | Action Smoothness 懲罰 |
| Sparse Reward 收斂困難 | 高 | 高 | Reward Annealing |

---

## 進度追蹤

### 已完成
- [x] Databricks 環境設置（JAX/MJX/W&B 可用）
- [x] Preprocessor JAX 翻譯（`preprocessor_jax.py`）
- [x] 獎勵函數實作（`soccer_env.py`）
- [x] JAX SAC 訓練代碼（`sac_agent.py`, `train_sac.py`）
- [x] 修復 `dist.mode()` NotImplementedError 錯誤
- [x] 60k 步測試驗證通過（log_alpha=-3.38）
- [x] 優化訓練配置（batch_size=512, updates_per_step=4）

### 進行中
- [ ] 180k 步正式訓練（~3 小時）
- [ ] W&B 監控 `eval/mean_reward` 趨勢

### 待辦
- [ ] jax2torch 轉換
- [ ] 官方環境驗證
- [ ] **Gate 3: 行為定性一致驗證通過**
- [ ] 官方環境微調
- [ ] SAI 提交

---

## 資源連結

**競賽相關：**
- [Booster Soccer Showdown GitHub](https://github.com/ArenaX-Labs/booster_soccer_showdown)
- [SAI 競賽頁面](https://competesai.com/competitions/cmp_xnSCxcJXQclQ)
- [Booster Dataset (HuggingFace)](https://huggingface.co/datasets/SaiResearch/booster_dataset)

**學習資源：**
- [MJX Tutorial Colab](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb)
- [Brax GitHub](https://github.com/google/brax) - MJX 環境 Wrapper
- [CleanRL JAX SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action_jax.py)
- [Flashbax Documentation](https://github.com/instadeepai/flashbax)
