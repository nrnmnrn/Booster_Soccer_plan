# 工具整合

## 概述

本文件說明如何整合 W&B、MLflow、GPU 監控和 Optuna 進行完整的 MLOps 工作流程。

**核心策略：W&B 負責實驗追蹤，MLflow 負責模型管理**

---

## W&B + MLflow 雙軌追蹤策略

### 核心原則

> 「歷史要紀錄在 Run，里程碑要註冊在 Catalog。」

- **W&B**：訓練過程中的「眼睛」- 看曲線、看影片、快速迭代
- **MLflow Run**：每個 Job 都記錄，確保完整血緣追蹤
- **MLflow Registry**：只在關鍵節點註冊，保持 Catalog 整潔

### 功能分工表

| 功能 | W&B | MLflow |
|------|-----|--------|
| **實時訓練曲線** | ✅ 主要 | 備援 |
| **GPU 監控** | ✅ wandb.log() | log_metric() |
| **影片記錄** | ✅ wandb.Video() | - |
| **超參數記錄** | ✅ config | log_params() |
| **每個 Job 的 Run** | - | ✅ 必要（血緣追蹤） |
| **模型 Registry** | artifact | ✅ 只註冊里程碑 |
| **跨 Job 模型共享** | - | ✅ load_model(alias) |
| **Lineage 追蹤** | - | ✅ Unity Catalog |

### 各 Job 的記錄策略

| 階段 | W&B | MLflow Run | MLflow Registry |
|------|-----|-----------|-----------------|
| **Job 2 (Pre-train)** | 實時曲線 + 影片 | ✅ Log Model | Register @Candidate-Pretrain |
| **Job 3 (Conversion)** | - | ✅ Log Artifact | N/A |
| **Job 4 (Fine-tune)** | 實時曲線 | ✅ Log Model | Register @Candidate-Finetuned |
| **Gate 3 通過** | - | 更新 Tag | 設置 @Champion Alias |

詳細的 MLflow 雙軌追蹤策略請見 [07-databricks-mlops.md](./07-databricks-mlops.md#模型-lineage-追蹤)。

---

## Weights & Biases (W&B) - 實驗追蹤

### 為什麼使用 W&B？

- **RL 訓練標配：** 追蹤不穩定的訓練過程
- **影片記錄：** 直接在網頁上觀看機器人行為
- **原生支援：** `imitation_learning/train.py` 已有 W&B 整合

### 設置

```python
import wandb

# 初始化
wandb.init(
    project="booster_soccer",
    config={
        "algorithm": "SAC",
        "batch_size": 2048,
        "learning_rate": 3e-4,
        "total_timesteps": 10_000_000
    }
)
```

### 記錄指標

```python
# 在訓練循環中
for step in range(total_timesteps):
    # ... 訓練邏輯 ...

    if step % 1000 == 0:
        wandb.log({
            "reward": episode_reward,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "entropy": entropy,
            "step": step
        })
```

### 影片記錄

```python
import time
from imitation_learning.utils.logging import get_wandb_video

class VideoLogger:
    """按時間間隔記錄影片，避免 MJX 高吞吐量下記錄過於頻繁"""
    def __init__(self, interval_seconds=300):  # 每 5 分鐘
        self.interval = interval_seconds
        self.last_log_time = 0

    def should_log(self):
        current_time = time.time()
        if current_time - self.last_log_time >= self.interval:
            self.last_log_time = current_time
            return True
        return False

# 初始化
video_logger = VideoLogger(interval_seconds=300)

# 在訓練循環中使用
if video_logger.should_log():
    # 收集 render frames
    renders = collect_episode_renders(env, model)

    # 上傳到 W&B
    wandb.log({
        "video": get_wandb_video(renders, fps=30),
        "step": step
    })
```

> **設計說明：** 使用時間間隔（而非固定步數）來記錄影片，因為 MJX (2048 並行環境) 的吞吐量可能達到 1M+ steps/min。固定 50,000 步可能每分鐘記錄多次影片，導致 W&B 儲存空間爆炸。每 5 分鐘記錄一次更加穩定。

### 整合到 RL 訓練腳本

```python
# training_scripts/training.py 修改

import wandb

def training_loop(env, model, action_function, preprocess_class, timesteps=1000):
    # 初始化 W&B
    wandb.init(project="booster_soccer_rl")

    replay_buffer = ReplayBuffer(max_size=100000)
    preprocessor = preprocess_class()

    for total_steps in range(timesteps):
        # ... 訓練邏輯 ...

        # 記錄到 W&B
        if total_steps % 1000 == 0:
            wandb.log({
                "episode_reward": episode_reward,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "buffer_size": len(replay_buffer)
            })

    wandb.finish()
```

---

## MLflow - 模型管理（Unity Catalog）

### 基本設置

```python
import mlflow

# 設置 Unity Catalog 作為 Model Registry
mlflow.set_registry_uri("databricks-uc")

# 設置實驗
mlflow.set_experiment("/Users/<username>/booster_soccer_mjx")
```

### 與 W&B 並行使用

```python
import wandb
import mlflow

# 同時記錄到兩個系統
with mlflow.start_run(run_name="mjx_sac_v1") as mlflow_run:
    wandb.init(project="booster_soccer", name="mjx_sac_v1")

    mlflow.log_params(config)  # MLflow
    wandb.config.update(config)  # W&B

    for step in range(total_steps):
        metrics = {"reward": reward, "loss": loss}

        wandb.log(metrics, step=step)  # W&B - 實時
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)  # MLflow - 備援

    # 模型註冊到 Unity Catalog
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="booster_soccer.rl_models.mjx_sac"
    )

    wandb.finish()
```

### 模型版本比較

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "booster_soccer.rl_models.ddpg_finetuned"

# 列出所有版本
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")
```

---

## GPU 監控

### 為什麼需要 GPU 監控？

- **L4 GPU 成本控制：** 即時發現效能瓶頸
- **OOM 預警：** 在崩潰前發現記憶體問題
- **訓練效率分析：** 確認 GPU 利用率達到預期

### GPU 監控必要指標（必須包含）

訓練腳本 **必須** 包含以下監控指標：

| 指標 | 變數名 | 說明 |
|------|--------|------|
| GPU 利用率 | `gpu/utilization_percent` | 確認 GPU 被充分使用 |
| VRAM 使用量 | `gpu/memory_percent` | 預警 OOM 風險 |

```python
# 每 1000 步必須記錄
wandb.log({
    "gpu/utilization_percent": gpu_util,
    "gpu/memory_percent": vram_usage,
    "step": step
})
```

> 如果 GPU 利用率長期低於 80%，應檢查是否有 I/O 瓶頸或 batch size 過小。

### JAX/XLA 記憶體管理

JAX 預設會佔用 **所有可用 GPU 記憶體**，這會導致：
- PyTorch 轉換階段 OOM
- MuJoCo EGL 渲染失敗
- 無法在同一 GPU 上運行其他程序

**必須設置的環境變數：**

```python
import os
# 在 import jax 之前設置！
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # 只使用 75% VRAM
os.environ['JAX_PREALLOCATE'] = 'false'                 # 動態分配，避免碎片化
```

| 環境變數 | 建議值 | 說明 |
|---------|-------|------|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | L4 24GB × 0.75 = 18GB 給 JAX |
| `JAX_PREALLOCATE` | `false` | 動態分配，避免啟動時佔滿 VRAM |
| `XLA_FLAGS` | `--xla_gpu_cuda_data_dir=/usr/local/cuda` | XLA 編譯緩存路徑 |

> 💡 在 Init Script 中已預設這些值。詳見 [07-databricks-mlops.md](./07-databricks-mlops.md)。

### 監控設計

```python
import pynvml

class GPUMonitor:
    def __init__(self, alert_memory_threshold=0.95, alert_temp=85):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.threshold = alert_memory_threshold
        self.alert_temp = alert_temp

    def get_metrics(self):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            self.handle, pynvml.NVML_TEMPERATURE_GPU
        )

        return {
            "gpu/memory_used_gb": mem.used / 1e9,
            "gpu/memory_percent": 100 * mem.used / mem.total,
            "gpu/utilization_percent": util.gpu,
            "gpu/temperature": temp,
        }

    def log_and_alert(self, step):
        metrics = self.get_metrics()

        # 記錄到 W&B
        wandb.log(metrics, step=step)

        # 警報檢查
        if metrics["gpu/memory_percent"] > self.threshold * 100:
            wandb.alert(
                title="High GPU Memory",
                text=f"{metrics['gpu/memory_percent']:.1f}%"
            )

        return metrics
```

### 監控頻率建議

| 間隔 | 監控內容 |
|------|----------|
| 每 1000 步 | 基本 metrics (memory, utilization) |
| 每 5000 步 | 完整狀態 (+ temperature, power) |
| 每 200k 步 | Checkpoint + 系統快照（配合 Preemptible） |

### 整合到訓練循環

```python
monitor = GPUMonitor()

for step in range(total_steps):
    # 訓練邏輯...

    # GPU 監控
    if step % 1000 == 0:
        gpu_metrics = monitor.log_and_alert(step)

    # Checkpoint（每 200k 步，配合 Preemptible 縮短間隔）
    if step % 200000 == 0:
        save_checkpoint(model, step)
        mlflow.log_artifact(checkpoint_path)
```

---

## Optuna 超參數調優

### 為什麼使用 Optuna？

- **輕量級：** 比 Ray Tune 更適合單 GPU
- **剪枝功能：** 自動停止差的實驗，節省 GPU 費用
- **RL 支援：** CleanRL 有官方整合範例

### 設置

```python
import optuna

def objective(trial):
    # 定義超參數搜索空間
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.01)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)

    # 建立模型和環境
    model = DDPG_FF(
        n_features=87,
        action_space=env.action_space,
        neurons=[256, 256],
        learning_rate=lr
    )

    # 訓練
    total_reward = 0
    for step in range(100000):
        # ... 訓練邏輯 ...

        # 定期報告中間結果（用於剪枝）
        if step % 10000 == 0:
            trial.report(total_reward / (step + 1), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return total_reward

# 建立 study 並優化
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)
study.optimize(objective, n_trials=50)

# 獲取最佳超參數
print("Best params:", study.best_params)
print("Best value:", study.best_value)
```

### 與 W&B 整合

```python
import optuna
import wandb

def objective(trial):
    # 初始化 W&B run
    wandb.init(
        project="booster_soccer_tuning",
        config={
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        },
        reinit=True
    )

    # 訓練
    for step in range(100000):
        # ... 訓練邏輯 ...
        wandb.log({"reward": reward, "step": step})

    wandb.finish()
    return final_reward
```

### Optuna Dashboard

```bash
# 安裝
pip install optuna-dashboard

# 啟動 dashboard
optuna-dashboard sqlite:///optuna_study.db
```

---

## 工具優先級

| 工具 | 優先級 | 何時使用 |
|------|--------|----------|
| **W&B** | P0 (必要) | 從 Day 1 開始 |
| **Optuna** | P1 (推薦) | 基礎訓練跑通後 |
| **TensorBoard** | P2 (可選) | 離線環境備用 |

---

## 常見問題

### Q: W&B 在 Databricks 無法上傳？

**常見原因：** Databricks Worker 節點防火牆阻擋 W&B 上傳端口。

**症狀：**
- 訓練速度異常慢（每秒步數驟降）
- `wandb.log()` 執行時間超過數秒
- 訓練卡住不動

**解決方案 1：手動 Offline 模式**

```python
# 設置 offline 模式
wandb.init(mode="offline")

# 訓練結束後同步
wandb.finish()
# 然後手動上傳: wandb sync ./wandb/offline-run-*
```

**解決方案 2：自動連線偵測（推薦）**

```python
import wandb
import socket
import time

def init_wandb_with_fallback(project, config, timeout=10):
    """
    初始化 W&B，自動偵測連線問題並切換 offline 模式

    Args:
        project: W&B 專案名稱
        config: 訓練配置
        timeout: 連線超時秒數

    Returns:
        wandb run object
    """
    def check_wandb_connection():
        try:
            socket.create_connection(("api.wandb.ai", 443), timeout=timeout)
            return True
        except (socket.timeout, OSError):
            return False

    if check_wandb_connection():
        try:
            run = wandb.init(project=project, config=config)
            print("✅ W&B 連線成功")
            return run
        except Exception as e:
            print(f"⚠️ W&B 初始化失敗: {e}，切換 offline 模式")
    else:
        print("⚠️ 無法連線 W&B，使用 offline 模式")

    # Fallback to offline
    run = wandb.init(project=project, config=config, mode="offline")
    print("📦 W&B offline 模式啟用")
    print("   訓練後請手動同步: wandb sync ./wandb/offline-run-*")
    return run
```

**解決方案 3：帶延遲偵測的 Logger**

```python
class WandbLogger:
    """
    帶有自動 offline fallback 和延遲偵測的 W&B Logger
    """
    def __init__(self, project, config):
        self.run = init_wandb_with_fallback(project, config)
        self.is_offline = self.run.mode == "offline"
        self.log_warning_shown = False

    def log(self, metrics, step=None):
        """
        記錄 metrics，偵測上傳延遲
        """
        start = time.time()
        wandb.log(metrics, step=step)
        elapsed = time.time() - start

        # 如果單次 log 超過 2 秒，發出警告
        if elapsed > 2.0 and not self.log_warning_shown:
            print(f"⚠️ W&B log 耗時 {elapsed:.1f}s，可能有網路瓶頸")
            print("   考慮切換 offline 模式或減少 log 頻率")
            self.log_warning_shown = True

    def finish(self):
        """結束記錄"""
        wandb.finish()
        if self.is_offline:
            print("📦 請執行: wandb sync ./wandb/offline-run-*")
```

**使用範例：**

```python
# 取代原本的 wandb.init()
logger = WandbLogger(project="booster_soccer_mjx", config=config)

# 訓練循環中
for step in range(total_steps):
    # ... 訓練邏輯 ...
    if step % 1000 == 0:
        logger.log({"reward": reward, "loss": loss}, step=step)

# 結束
logger.finish()
```

### Q: Optuna 試驗太慢？

```python
# 使用更激進的剪枝
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=3,      # 減少啟動試驗數
    n_warmup_steps=5000,     # 減少預熱步數
    interval_steps=5000      # 更頻繁檢查
)
```

### Q: 如何恢復中斷的 Optuna study？

```python
# 使用持久化 storage
study = optuna.create_study(
    study_name="booster_soccer",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True  # 恢復已有 study
)
```

---

## 資源連結

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Video Logging](https://docs.wandb.ai/ref/python/data-types/video/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [CleanRL Hyperparameter Tuning](https://docs.cleanrl.dev/advanced/hyperparameter-tuning/)
- [Optuna Examples for RL](https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py)
