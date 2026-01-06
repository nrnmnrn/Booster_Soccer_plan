# Databricks MLOps

## 概述

本文件說明如何使用 Databricks Jobs/Workflows 建立端到端的 RL 訓練 Pipeline，包含 Unity Catalog 整合、GPU 監控和自動化流程。

**目標：** 建立可重現、可追蹤、可擴展的 RL 訓練基礎設施。

---

## 算力策略：開發求穩，訓練求省

為了在有限預算下極大化實驗次數，我們將開發與訓練階段的算力邏輯徹底分離。

### 開發階段：All-purpose Cluster (On-demand)

| 項目 | 配置 |
|------|------|
| **用途** | Gate 1/2 驗證、Debug、Notebook 實驗 |
| **類型** | All-purpose Compute (Single Node) |
| **硬體** | NVIDIA L4 GPU (On-demand) |
| **機制** | Auto-termination = 60 分鐘 |

**選擇原因：**
- **快啟動優化：** Databricks 在 All-purpose 模式下具備進程級重啟能力。在密集 Debug 代碼時，重啟只需 10-20 秒，無需像 Pool 一樣支付閒置 VM 租金即可獲得極佳的反應速度。
- **環境一致性：** 搭配 Docker Container Service (DCS)，環境已固化，減少冷啟動時的安裝等待。

### 訓練階段：Job Cluster + Instance Pool (Spot)

| 項目 | 配置 |
|------|------|
| **用途** | Job 2 (預訓練)、Job 4 (微調) |
| **類型** | Train-Pool + Spot Instances |
| **硬體** | NVIDIA L4 GPU (Preemptible) |

**成本優勢：**
| 項目 | 效益 |
|------|------|
| DBU 費率 | Jobs Workload 僅為 All-purpose 的 ~1/3 |
| 硬體單價 | Spot 比 On-demand 便宜 70-80% |
| 綜合預算 | 同樣的錢可多跑 **4-5 倍**實驗 |

**選擇原因：**
- **自動恢復能力：** 配合 Unity Catalog Volumes 存儲 Checkpoints。即便 Spot 機器被回收，Job 會自動重試，腳本會偵測最新權重並實現「無感續練」。
- **Warm Start 優勢：** 設置 `idle_instance_autotermination_minutes = 60`。當一個 4 小時的訓練 Job 完成後，機器會留在池子裡一小時。如果你立即調整參數啟動下一場實驗，將享受秒級開機。

---

## Train-Pool 配置

### Instance Pool 定義

```json
{
  "instance_pool_name": "booster-train-pool",
  "node_type_id": "g2-standard-12",
  "min_idle_instances": 0,
  "max_capacity": 2,
  "idle_instance_autotermination_minutes": 60,
  "preloaded_spark_versions": ["16.4-gpu-ml-scala2.12"],
  "gcp_attributes": {
    "availability": "PREEMPTIBLE_GCP"
  }
}
```

### 設計理由

| 參數 | 值 | 原因 |
|------|-----|------|
| `min_idle_instances` | 0 | 不產生閒置費用 |
| `idle_instance_autotermination_minutes` | 60 | Warm Start：連續實驗秒級開機 |
| `max_capacity` | 2 | 允許同時跑 2 個實驗 |
| `preloaded_spark_versions` | 16.4-gpu-ml | 預載 Runtime 加速啟動 |

### 建立 Instance Pool

```bash
# 使用 Databricks CLI
databricks instance-pools create --json '{
  "instance_pool_name": "booster-train-pool",
  "node_type_id": "g2-standard-12",
  "min_idle_instances": 0,
  "max_capacity": 2,
  "idle_instance_autotermination_minutes": 60,
  "preloaded_spark_versions": ["16.4-gpu-ml-scala2.12"],
  "gcp_attributes": {
    "availability": "PREEMPTIBLE_GCP"
  }
}'
```

---

## Jobs/Workflows Pipeline 架構

### 整體流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATABRICKS WORKFLOWS PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [Job 1: Setup]  →  [Job 2: MJX Pre-train]  →  [Job 3: Conversion]  │
│       (CPU)             (L4 GPU)                    (CPU)             │
│     ~30 min              ~4 hrs                    ~10 min            │
│                              │                                        │
│                              ↓                                        │
│                    [Job 4: Fine-tune]  →  [Job 5: Submit]            │
│                         (L4 GPU)              (CPU)                   │
│                          ~3 hrs               ~5 min                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Job 定義

| Job | 目的 | Cluster 類型 | 預估時間 | 輸出 |
|-----|------|-------------|---------|------|
| **Job 1: Setup** | 環境驗證、Unity Catalog 設置 | CPU | 30 min | Volumes 建立 |
| **Job 2: MJX Pre-train** | JAX SAC 預訓練 (10M 步) | L4 GPU | 2-4 hrs | `final_checkpoint.pkl` |
| **Job 3: Conversion** | JAX → PyTorch 轉換 | CPU | 10 min | `model_pretrained.pt` |
| **Job 4: Fine-tune** | DDPG 官方環境微調 | L4 GPU | 2-3 hrs | `model_finetuned.pt` |
| **Job 5: Submit** | Benchmark + SAI 提交 | CPU | 5 min | 競賽分數 |

---

## Job 詳細規劃

### Job 1: Environment Setup

**目的：** 驗證環境、建立 Unity Catalog 結構、確認 GPU 可用

**步驟：**
1. 安裝依賴 (`mujoco`, `mujoco-mjx`, `jax[cuda12]`, `wandb`, `mlflow`)
2. 驗證 JAX 可見 GPU
3. 建立 Unity Catalog Schema 和 Volumes
4. 驗證 XML 場景可載入
5. 測試 W&B 連線
6. 記錄設置結果到 MLflow

**Cluster 配置：**
```json
{
  "spark_version": "16.4-cpu-ml-scala2.12",
  "node_type_id": "n2-standard-4",
  "num_workers": 0,
  "timeout_seconds": 1800
}
```

**產出：**
- Unity Catalog Volumes 建立完成
- 環境驗證報告 (MLflow logged)

---

### Job 2: MJX Pre-training

**目的：** 使用 MJX GPU 加速進行 SAC 預訓練

**步驟：**
1. 初始化 W&B + MLflow 雙重記錄
2. 載入 MJX 環境 (2048 並行)
3. 執行 SAC 訓練循環 (10M 步)
4. 每 1000 步記錄 metrics
5. 每 5000 步記錄 GPU 狀態
6. 每 200k 步保存 checkpoint（配合 Preemptible 縮短間隔）
7. 最終模型註冊到 Unity Catalog

**Cluster 配置（使用 Train-Pool + Docker）：**
```json
{
  "instance_pool_id": "<TRAIN_POOL_ID>",
  "num_workers": 0,
  "spark_version": "16.4-gpu-ml-scala2.12",
  "docker_image": {
    "url": "your-registry/booster-rl:v1"
  },
  "timeout_seconds": 28800
}
```

> **注意**：使用 Instance Pool 時，`node_type_id` 和 `gcp_attributes` 由 Pool 定義，無需在 Job 配置中指定。

**Job 重試配置（配合 Preemptible）：**
```json
{
  "max_retries": 2,
  "retry_on_timeout": true
}
```

**備用方案：Init Script (install_mjx.sh)**

> ⚠️ **建議優先使用 Docker**：詳見 [01-environment-setup.md](./01-environment-setup.md)。

```bash
#!/bin/bash
# 核心套件安裝（版本與 Dockerfile 保持一致）
pip install --no-cache-dir \
  "mujoco==3.2.6" \
  "mujoco-mjx==3.2.6" \
  "brax==0.12.1" \
  "optax==0.2.4" \
  "wandb==0.19.1" \
  "mlflow==2.19.0" \
  "pynvml>=12.0.0"

# PyTorch（用於最終微調）
pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# JAX/XLA 記憶體和效能設置
echo "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75" >> /etc/profile
echo "export JAX_PREALLOCATE=false" >> /etc/profile
echo "export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda'" >> /etc/profile
echo "export MUJOCO_GL=egl" >> /etc/profile
```

**監控項目：**
- `train/reward_mean`
- `train/critic_loss`
- `train/actor_loss`
- `gpu/memory_percent`
- `gpu/utilization_percent`

**產出：**
- `/Volumes/booster_soccer/rl_models/checkpoints/mjx_pretraining/final_checkpoint.pkl`
- `booster_soccer.rl_models.mjx_sac_pretrained` (Model Registry)

---

### Job 3: Model Conversion

**目的：** 將 JAX SAC Actor 轉換為 PyTorch DDPG 格式

**步驟：**
1. 載入 JAX checkpoint
2. 提取 Actor 權重 (只取 mean 部分，捨棄 log_std)
3. 轉換為 PyTorch state_dict
4. 驗證維度正確 (最後一層 12 維，不是 24 維)
5. 保存轉換後模型
6. 記錄轉換結果到 MLflow

**關鍵轉換邏輯：**
```
SAC Actor 輸出: 24 (mean:12 + log_std:12)
                    ↓
                只取前 12 維
                    ↓
DDPG Actor 輸出: 12
```

**驗證檢查：**
- `layers.2.weight.shape == (12, 256)` ← Critical!
- `layers.2.bias.shape == (12,)`

**產出：**
- `/Volumes/booster_soccer/rl_models/checkpoints/pytorch_finetuning/model_pretrained.pt`
- `booster_soccer.rl_models.ddpg_pretrained` (Model Registry)

---

### Job 4: PyTorch Fine-tuning

**目的：** 在官方環境中使用 DDPG 微調預訓練模型

**步驟：**
1. 載入預訓練權重
2. 設置 Feature Freeze Scheduler (三階段)
3. 設置 Reward Annealer (Dense → Sparse)
4. 執行 DDPG 訓練循環 (200k 步)
5. 記錄 metrics 到 W&B + MLflow
6. 保存最終模型

**Feature Freeze 三階段：**

| 階段 | Steps | 可訓練層 | Learning Rate |
|------|-------|---------|---------------|
| Phase 1 | 0 - 20k | 最後一層 | 3e-5 |
| Phase 2 | 20k - 50k | 最後兩層 | 3e-5 |
| Phase 3 | 50k+ | 全網路 | 1e-5 (降低) |

**Reward Annealing：**
```
R_total = α × R_dense + β × R_official

開始: α = 1.0, β = 0.1
結束: α = 0.1, β = 1.0
```

**產出：**
- `/Volumes/booster_soccer/rl_models/checkpoints/pytorch_finetuning/model_finetuned.pt`
- `booster_soccer.rl_models.ddpg_finetuned` (Model Registry)

---

### Job 5: SAI Submission

**目的：** 本地 benchmark 並提交到 SAI 競賽

**步驟：**
1. 載入微調後模型
2. 在三個環境執行 benchmark
3. 記錄 benchmark 結果到 MLflow
4. 提交到 SAI 平台
5. 記錄提交 ID

**Benchmark 環境：**
- `LowerT1GoaliePenaltyKick-v0`
- `LowerT1ObstaclePenaltyKick-v0`
- `LowerT1KickToTarget-v0`

**產出：**
- Benchmark 結果 (MLflow metrics)
- SAI 提交記錄

---

## Workflow 編排

### Multi-Task Job 定義

```json
{
  "name": "booster_soccer_pipeline",
  "tasks": [
    {
      "task_key": "setup",
      "notebook_task": {
        "notebook_path": "/databricks/workflows/01_environment_setup"
      }
    },
    {
      "task_key": "pretrain",
      "depends_on": [{"task_key": "setup"}],
      "notebook_task": {
        "notebook_path": "/databricks/workflows/02_mjx_pretraining"
      }
    },
    {
      "task_key": "convert",
      "depends_on": [{"task_key": "pretrain"}],
      "notebook_task": {
        "notebook_path": "/databricks/workflows/03_model_conversion"
      }
    },
    {
      "task_key": "finetune",
      "depends_on": [{"task_key": "convert"}],
      "notebook_task": {
        "notebook_path": "/databricks/workflows/04_pytorch_finetuning"
      }
    },
    {
      "task_key": "submit",
      "depends_on": [{"task_key": "finetune"}],
      "notebook_task": {
        "notebook_path": "/databricks/workflows/05_sai_submission"
      }
    }
  ],
  "email_notifications": {
    "on_failure": ["your-email@example.com"]
  }
}
```

### 手動觸發 vs 自動排程

| 模式 | 使用場景 | 配置 |
|------|----------|------|
| **手動觸發** | 開發迭代、調試 | `databricks jobs run-now --job-id <ID>` |
| **自動排程** | 每日訓練、定期更新 | `schedule: { "quartz_cron_expression": "0 0 2 * * ?" }` |

---

## 成本優化策略

### GCP Preemptible Instances（推薦）

使用 Preemptible VM 可節省約 **56-70%** 的運算成本：

| 類型 | L4 GPU 價格 (每小時) | 10 小時訓練成本 |
|------|---------------------|----------------|
| **On-Demand** | ~$0.80 | ~$8.00 |
| **Preemptible** | ~$0.35 | ~$3.50 |

**配置方式（單節點 GPU Cluster）：**
```json
{
  "gcp_attributes": {
    "availability": "PREEMPTIBLE_GCP"
  }
}
```

> ⚠️ **注意**：對於單節點（無 Worker）Cluster，必須使用 `availability` 而非 `use_preemptible_executors`。後者只影響 Worker 節點。

**風險與緩解：**
| 風險 | 緩解措施 |
|------|----------|
| 被搶佔中斷訓練 | 縮短 Checkpoint 間隔至 200k 步 |
| 進度丟失 | 設置 `max_retries: 2` 自動重試 |
| 頻繁被搶佔 | 嘗試不同時段啟動（非尖峰時段） |

---

### Cluster 自動終止

```python
# 在 Notebook 結尾
dbutils.notebook.exit(json.dumps({
    "status": "success",
    "terminate_cluster": True
}))
```

### GPU Job 優化

| 策略 | 效果 | 實作 |
|------|------|------|
| **Checkpoint Recovery** | 避免重複訓練 | 自動偵測最新 checkpoint（每 200k 步） |
| **Early Stopping** | 節省無效訓練時間 | W&B alert + 手動停止 |
| **Batch Size 最大化** | 提高 GPU 利用率 | 2048 並行環境 |
| **Preemptible + Retry** | 節省 56-70% 成本 | `availability: PREEMPTIBLE_GCP` |

### Checkpoint Recovery 邏輯

#### 原子性寫入（防止 Spot 搶佔損壞）

```python
import os
import pickle
import tempfile
import shutil
import re

def save_checkpoint_atomic(params, checkpoint_dir, step, keep_last=2):
    """
    原子性寫入 checkpoint，防止 Spot 搶佔導致檔案損壞

    策略：
    1. 先寫入 temp 檔案
    2. 成功後 rename 為正式檔案（原子操作）
    3. 清理舊 checkpoint，只保留最近 N 個

    Args:
        params: 要保存的模型參數
        checkpoint_dir: checkpoint 目錄
        step: 當前訓練步數
        keep_last: 保留最近幾個 checkpoint（預設 2）

    Returns:
        保存的 checkpoint 路徑
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/step_{step}.pkl"

    # 1. 寫入 temp 檔案（同一目錄確保在同一檔案系統）
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=checkpoint_dir,
        delete=False,
        suffix='.tmp'
    ) as tmp:
        pickle.dump(params, tmp)
        tmp_path = tmp.name

    # 2. 原子 rename（POSIX 系統上 rename 是原子操作）
    shutil.move(tmp_path, checkpoint_path)

    # 3. 清理舊 checkpoint
    _cleanup_old_checkpoints(checkpoint_dir, keep_last)

    return checkpoint_path


def _cleanup_old_checkpoints(checkpoint_dir, keep_last):
    """保留最近 N 個 checkpoint，刪除其他"""
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pkl")
    ], key=lambda x: int(re.search(r'step_(\d+)', x).group(1)), reverse=True)

    for ckpt in checkpoints[keep_last:]:
        try:
            os.remove(os.path.join(checkpoint_dir, ckpt))
        except OSError:
            pass  # 忽略刪除失敗
```

#### 載入 Checkpoint（支援 Fallback）

```python
def load_latest_checkpoint(checkpoint_dir, fallback_count=2):
    """
    載入最新 checkpoint，支援 fallback 到較舊版本

    當最新 checkpoint 損壞時（Spot 搶佔導致），自動嘗試載入較舊的版本。

    Args:
        checkpoint_dir: checkpoint 目錄
        fallback_count: 最多嘗試的 checkpoint 數量

    Returns:
        (params, step) 或 (None, 0)
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0

    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pkl")
    ], key=lambda x: int(re.search(r'step_(\d+)', x).group(1)), reverse=True)

    if not checkpoints:
        return None, 0

    for i, ckpt in enumerate(checkpoints[:fallback_count]):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        try:
            with open(ckpt_path, 'rb') as f:
                params = pickle.load(f)
            step = int(re.search(r'step_(\d+)', ckpt).group(1))
            if i > 0:
                print(f"⚠️ 最新 checkpoint 損壞，已載入 {ckpt}")
            return params, step
        except (EOFError, pickle.UnpicklingError, OSError) as e:
            print(f"⚠️ Checkpoint {ckpt} 損壞: {e}，嘗試上一個...")
            continue

    print("❌ 所有 checkpoint 都無法載入，從頭開始訓練")
    return None, 0
```

#### 使用範例

```python
# 在訓練腳本中
checkpoint_dir = "/Volumes/booster_soccer/rl_models/checkpoints/mjx_pretraining"

# 嘗試載入之前的進度
params, start_step = load_latest_checkpoint(checkpoint_dir)
if params is not None:
    print(f"✅ 從 step {start_step} 繼續訓練")
else:
    print("🆕 從頭開始訓練")
    params = init_params()
    start_step = 0

# 訓練循環
for step in range(start_step, total_steps):
    # ... 訓練邏輯 ...

    # 每 200k 步保存 checkpoint（原子寫入）
    if step % 200_000 == 0 and step > 0:
        save_checkpoint_atomic(params, checkpoint_dir, step, keep_last=2)
```

---

## 錯誤處理與警報

### 失敗通知

```json
{
  "email_notifications": {
    "on_failure": ["team@example.com"],
    "on_success": ["team@example.com"]
  }
}
```

### W&B 警報整合

```python
# GPU 記憶體警報
if gpu_memory_percent > 95:
    wandb.alert(
        title="GPU Memory Critical",
        text=f"Memory at {gpu_memory_percent}%",
        level=wandb.AlertLevel.ERROR
    )

# 訓練停滯警報
if reward_moving_avg < threshold:
    wandb.alert(
        title="Training Stalled",
        text="Reward not improving for 100k steps"
    )
```

---

## 模型 Lineage 追蹤

### 雙軌追蹤策略

**設計原則：**
- **Run (歷史追蹤)**：每個 Job 都記錄，確保完整 Lineage
- **Registry (里程碑)**：只在關鍵節點註冊，保持 Catalog 整潔

> 「歷史要紀錄在 Run，里程碑要註冊在 Catalog。」

### 各 Job 的 MLflow 動作

| 階段 | MLflow 動作 | Registry Alias | 追溯意義 |
|------|------------|----------------|---------|
| **Job 2 (Pre-train)** | Log Model + Register | `Candidate-Pretrain` | 紀錄機器人學會「走路」的過程 |
| **Job 3 (Conversion)** | Log Artifact | N/A | 紀錄 JAX→PyTorch 轉換的誤差數據 |
| **Job 4 (Fine-tune)** | Log Model + Register | `Candidate-Finetuned` | 紀錄官方環境適應後的表現 |
| **Gate 3 (Verification)** | 更新 Alias | `Champion` / `Finalist` | 決賽候選模型 |

### 為什麼「每個 Job 都記錄」是必要的？

1. **問題追溯：** 發現微調行為崩潰時，能一鍵找回源頭
2. **避免硬編碼路徑錯誤：** 用 Alias 載入模型，而非手動輸入 Volume 路徑
3. **血緣追蹤：** 知道模型是由哪個 Notebook 版本產生

### Unity Catalog 模型關係

```
sac_actor (v1)  @Candidate-Pretrain
         ↓
    [jax2torch]  (logged as artifact)
         ↓
ddpg_pretrained (v1)  @Candidate-Finetuned
         ↓
    [fine-tune + Gate 3]
         ↓
ddpg_finetuned (v1)  @Champion → SAI Submission
```

### 跨 Job 模型共享（避免硬編碼路徑）

```python
# Job 3 載入 Job 2 產出的模型（使用 Alias 而非 Volume 路徑）
model = mlflow.models.load_model(
    "models:/booster_soccer.rl_models.sac_actor@Candidate-Pretrain"
)

# 比較：避免這種硬編碼方式
# model = load("/Volumes/.../final_v2_new_fixed.pkl")  # 容易出錯
```

### DualLogger 建議實現

```python
class DualLogger:
    def __init__(self, wandb_project, mlflow_experiment):
        wandb.init(project=wandb_project)
        mlflow.set_experiment(mlflow_experiment)
        self.mlflow_run = mlflow.start_run()

    def log_model(self, model, model_name, register=False):
        """
        永遠 Log 到 Run 裡（確保有歷史可以追溯）
        只有在 register=True 時才註冊到 Unity Catalog
        """
        mlflow.pytorch.log_model(model, artifact_path="model")

        if register:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name=f"booster_soccer.rl_models.{model_name}"
            )

    def set_alias(self, model_name, version, alias):
        """為通過驗證的模型設置 Alias"""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"booster_soccer.rl_models.{model_name}",
            alias=alias,
            version=version
        )
```

### MLflow 記錄 Lineage（Job 4 範例）

```python
# 在 Job 4 (Fine-tuning) 中
with mlflow.start_run(run_name="ddpg_finetune_v1") as run:
    # 記錄父模型資訊
    mlflow.set_tag("parent_model", "booster_soccer.rl_models.ddpg_pretrained")
    mlflow.set_tag("parent_alias", "Candidate-Pretrain")
    mlflow.set_tag("training_type", "finetuning")

    # 訓練完成後
    logger.log_model(model, "ddpg_finetuned", register=True)

    # 如果通過 Gate 3 驗證
    if gate3_passed:
        logger.set_alias("ddpg_finetuned", run.info.run_id, "Champion")
```

### 儲存空間管理

| 建議 | 說明 |
|------|------|
| **每個 Job 只在結束時註冊一次** | 中間過程用 `mlflow.log_artifact()` 存成普通文件 |
| **定期清理失敗 Run** | 設置腳本或利用 MLflow 的 `deleted` 標籤管理 |
| **Checkpoint 存 Volume** | 頻繁存檔用 Volume，只有里程碑才註冊到 Registry |

---

## 快速開始

### 1. 設置 Unity Catalog

```sql
CREATE CATALOG IF NOT EXISTS booster_soccer;
CREATE SCHEMA IF NOT EXISTS booster_soccer.rl_models;
CREATE VOLUME IF NOT EXISTS booster_soccer.rl_models.checkpoints;
```

### 2. 建立 Instance Pool (Train-Pool)

```bash
# 使用 Databricks CLI 建立 Instance Pool
databricks instance-pools create --json '{
  "instance_pool_name": "booster-train-pool",
  "node_type_id": "g2-standard-12",
  "min_idle_instances": 0,
  "max_capacity": 2,
  "idle_instance_autotermination_minutes": 60,
  "preloaded_spark_versions": ["16.4-gpu-ml-scala2.12"],
  "gcp_attributes": {
    "availability": "PREEMPTIBLE_GCP"
  }
}'

# 記下返回的 instance_pool_id，用於 Job 配置
```

### 3. 準備 Docker Image

參考 [01-environment-setup.md](./01-environment-setup.md) 中的 Dockerfile，Build 並 Push 到你的 Registry：

```bash
# Build
docker build -t your-registry/booster-rl:v1 .

# Push
docker push your-registry/booster-rl:v1
```

### 4. 上傳 Init Script（備用方案）

如果 Docker Image 尚未準備好，可暫時使用 Init Script：

```bash
# 先建立 scripts 目錄（在 Unity Catalog Volume 中）
databricks fs mkdir /Volumes/booster_soccer/rl_models/scripts

# 上傳 Init Script 到 Unity Catalog Volume（避免 dbfs:/ 在 UC Cluster 被禁用）
databricks fs cp scripts/install_mjx.sh /Volumes/booster_soccer/rl_models/scripts/install_mjx.sh
```

### 5. 建立 Workflow

```bash
databricks jobs create --json-file config/job_definitions.json
```

### 6. 執行 Pipeline

```bash
databricks jobs run-now --job-id <JOB_ID>
```

### 7. 監控訓練

- **W&B Dashboard:** 實時曲線和影片
- **MLflow UI:** 模型版本和比較
- **Databricks Job UI:** Pipeline 狀態

---

## 常見問題

### Q: GPU Job 啟動很慢？

**原因：** Init script 每次都重新安裝套件

**解決：** 使用預裝好套件的 Docker Container 或 Cluster Policy

### Q: W&B 無法上傳？

**原因：** Databricks 網路限制

**解決：** 使用 offline mode，訓練後同步
```python
wandb.init(mode="offline")
# 訓練結束後
# wandb sync ./wandb/offline-run-*
```

### Q: Unity Catalog 權限不足？

**原因：** 缺少必要權限

**解決：** 確認有 `CREATE MODEL`, `USE SCHEMA`, `USE CATALOG` 權限

---

## 資源連結

- [Databricks Jobs Documentation](https://docs.databricks.com/workflows/jobs/jobs.html)
- [Unity Catalog Model Registry](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
- [GPU Cluster Configuration](https://docs.databricks.com/compute/gpu.html)

---

## 下一步

1. 完成 [01-environment-setup.md](./01-environment-setup.md) 的環境設置
2. 建立 `databricks/workflows/` 目錄結構
3. 依序實作 5 個 Job Notebooks
4. 測試端到端 Pipeline
5. 迭代優化並提交競賽
