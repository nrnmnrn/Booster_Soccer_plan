# Claude 專案上下文

## 專案目標

參加 **Booster Soccer Showdown** 機器人足球競賽（獎金池 $10,000），目標是訓練一個能在多個足球環境中泛化的機器人智能體。

---

## 用戶背景

- **ML 經驗：** 有機器學習基礎，但不熟悉強化學習（RL）
- **開發環境：** 本地開發 + 遠端 Databricks
  - **Node Type:** g2-standard-12 [L4]
  - **Memory:** 48 GB
  - **GPU:** 1x NVIDIA L4 (24GB VRAM)
  - **Runtime:** 16.4 LTS ML (Apache Spark 3.5.2, Scala 2.12, Python 3.12, CUDA 12.6)
- **開發工具：** Plandex（AI 輔助代碼規劃與實作）
- **雲端平台：** GCP Databricks
- **團隊規模：** 單人參賽
- **時間：** 2-4 週

---

## 技術策略

採用 **路線 B：MJX 預訓練 + PyTorch 微調**（整合 Opus + Gemini 審查）

```
核心流程：
mimic/assets XML → MJX (GPU) → 密集獎勵預訓練 → jax2torch → Reward Annealing 微調 → 提交
```

### 關鍵決策（2025-12-18 更新）

1. **使用 MJX GPU 加速**：在 GPU 上並行運行 2048 個模擬環境
2. **87 維 Preprocessor**：與 DDPG 訓練一致，SAI 接受自定義 Preprocessor
3. **Task Index 隨機化**：在 MJX 環境 reset 時隨機注入 task_onehot
4. **Action Smoothness**：納入獎勵函數，減少高頻震盪
5. **Domain Randomization 三級分層**：Level 1 → Level 2 → Level 3
6. **Reward Annealing**：微調時從 Dense 逐漸切換到 Official Sparse
7. **官方環境微調**：延長至 25-30% 訓練時間

---

## Claude 的主要任務

### 1. 代碼開發

- 建立 MJX 環境封裝 (`mjx_env.py`)
- 翻譯 Preprocessor 從 NumPy 到 JAX (`preprocessor_jax.py`)
- 實作簡化獎勵函數 (`rewards.py`)
- 建立 JAX SAC 訓練腳本 (`train_mjx_sac.py`)
- 修改 `main.py` 支援載入預訓練權重

### 2. 技術指導

- 解釋 RL 概念（DDPG, SAC, Actor-Critic）
- 協助調試 MJX/JAX 相關問題
- 提供代碼審查和優化建議

### 3. 競賽準備

- 協助準備 SAI 提交
- 分析競賽結果並建議改進方向

---

## 關鍵文件參考

### 競賽 Repo（booster_soccer_showdown）

| 文件 | 用途 |
|------|------|
| `mimic/assets/booster_t1/booster_lower_t1.xml` | 機器人 XML 定義 |
| `imitation_learning/utils/networks.py` | Flax MLP 網路（可重用） |
| `imitation_learning/train.py` | JAX IL 訓練參考 |
| `training_scripts/main.py` | PyTorch DDPG 入口 + Preprocessor |
| `training_scripts/ddpg.py` | Actor-Critic 定義 |
| `imitation_learning/scripts/jax2torch.py` | JAX → PyTorch 轉換 |

### 本專案

| 文件 | 用途 |
|------|------|
| `docs/README.md` | 計劃概述 |
| `docs/01-environment-setup.md` | 環境設置 + Unity Catalog |
| `docs/02-mjx-training.md` | MJX 訓練流程 |
| `docs/03-finetuning-submission.md` | 微調與提交 |
| `docs/04-tooling-integration.md` | 工具整合 (W&B + MLflow + GPU 監控) |
| `docs/05-verification-gates.md` | 驗證關卡 |
| `docs/06-official-evaluation.md` | 官方評估函數 |
| `docs/07-databricks-mlops.md` | **Databricks MLOps (Jobs, Pipeline)** |

---

## 技術棧

| 類別 | 工具 |
|------|------|
| **物理模擬** | MuJoCo, MJX |
| **深度學習** | JAX, Flax, PyTorch |
| **RL 演算法** | SAC (預訓練), DDPG (微調) |
| **實驗追蹤** | W&B (實時監控) + MLflow (模型版本) |
| **模型治理** | Unity Catalog (Model Registry) |
| **調參** | Optuna |
| **雲端** | GCP Databricks (L4 GPU) |
| **自動化** | Databricks Jobs/Workflows |
| **AI 開發輔助** | Plandex, Claude Code |

---

## Databricks MLOps 架構（2025-12-19 更新）

### W&B + MLflow 分工策略

| 功能 | W&B | MLflow |
|------|-----|--------|
| **實時訓練曲線** | ✅ 主要 | 備援 |
| **GPU 監控** | ✅ wandb.log() | log_metric() |
| **影片記錄** | ✅ wandb.Video() | - |
| **超參數記錄** | ✅ config | log_params() |
| **模型版本控制** | artifact | ✅ Unity Catalog |
| **模型部署** | - | ✅ Model Registry |
| **Lineage 追蹤** | - | ✅ Unity Catalog |

### Unity Catalog 結構

```
booster_soccer/                    # Catalog
├── rl_models/                     # Schema
│   ├── checkpoints (Volume)       # Checkpoint 儲存
│   ├── artifacts (Volume)         # 模型 Artifacts
│   ├── mjx_sac_pretrained         # 模型註冊
│   └── ddpg_finetuned             # 模型註冊
└── experiments/                   # Schema
    └── runs (Table)               # 實驗追蹤表
```

### Jobs/Workflows Pipeline

```
[Job 1: Setup] → [Job 2: MJX Pre-train] → [Job 3: Conversion]
     (CPU)            (L4 GPU)                 (CPU)
                          ↓
              [Job 4: Fine-tune] → [Job 5: Submit]
                  (L4 GPU)              (CPU)
```

### GCP 成本優化

| 配置 | 說明 |
|------|------|
| **Preemptible VM** | `availability: PREEMPTIBLE_GCP`，節省 56-70% |
| **Job Retry** | `max_retries: 2`，配合 Preemptible |
| **Checkpoint 間隔** | 200k 步（縮短以降低被搶佔損失） |

### JAX/XLA 記憶體設置

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # 只用 75% VRAM，預留給渲染
JAX_PREALLOCATE=false                 # 動態分配
MUJOCO_GL=egl                         # Headless 渲染
```

### GPU 監控設計

- **監控頻率**：每 1000 步記錄 metrics，每 5000 步完整狀態
- **警報閾值**：Memory > 95%，Temperature > 85°C
- **工具**：pynvml + W&B/MLflow 雙重記錄

---

## 風險與回退（整合 Gemini 審查）

| 風險 | 回退方案 |
|------|----------|
| MJX 環境建立失敗 | 切換路線 A（純 PyTorch DDPG） |
| Sim-to-Sim Gap 過大 | 增加官方環境微調步數 + 升級 DR Level |
| 時間不足 | 優先完成 baseline 提交 |
| **log_std 轉換錯誤** | Gate 3 驗證 + 數值範圍檢查 [-5, 2] |
| **Sparse Reward 收斂困難** | Reward Annealing |
| **Policy Collapse** | 行為約束 + idle_penalty |
| **Action 高頻震盪** | Action Smoothness 懲罰 |

---

## Opus + Gemini 最終審查共識（2025-12-18）

### Critical Blockers 已識別

| 問題 | 嚴重性 | 解決方案 |
|------|--------|----------|
| **XML 缺少場景/球** | 🔴 高 | 建立 `mjx_scene.xml` 包含 ground + ball |
| **Info Dict 不存在** | 🔴 高 | 實作 `_build_info_from_mjx_data()` |
| **step() 漏傳 task_onehot** | 🟡 中 | 修正 `_get_obs()` 調用 |
| **SAC→DDPG 架構不匹配** | 🔴 高 | 只取 mean 權重，捨棄 log_std |

### 技術決策共識

| 項目 | 決定 |
|------|------|
| Preprocessor 維度 | 使用 87 維 |
| Body ID 獲取 | **使用 `mj_name2id`**，禁止硬編碼 |
| Info Dict 重建 | **只建必要 keys**（Preprocessor 實際使用的） |
| Quaternion 順序 | MuJoCo 使用 `[w, x, y, z]`，Gate 1 需驗證 |
| SAC→DDPG 轉換 | 只取 Actor 的 mean 部分（前 12 維） |
| Feature Freeze | **三階段**：0-20k 只訓練最後層 → 20k-50k 解凍倒數第二層 → 50k+ 全網路 |
| Reward Annealing | 整個微調過程使用 α(1.0→0.1) + β(0.1→1.0) |
| 工作優先級 | **先跑通再對齊**（Day 1 目標是端到端能執行） |

### MJX Info Dict 必要 Keys

```
robot_quat, robot_gyro, robot_accelerometer, robot_velocimeter,
goal_team_0_rel_robot, goal_team_1_rel_robot,
ball_xpos_rel_robot, ball_velp_rel_robot, ball_velr_rel_robot,
player_team, task_index
```

**可簡化的 Keys（MJX 預訓練設為 zeros）：**
- `goalkeeper_team_*` → zeros
- `defender_xpos` → zeros
- `target_xpos_rel_robot` → zeros

---

## 溝通偏好

- **語言：** 繁體中文為主，技術術語可用英文
- **代碼風格：** 遵循現有 repo 的風格
- **回覆長度：** 簡潔為主，需要時可詳細解釋
