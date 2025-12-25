# 環境設置

## 概述

本文件說明如何設置 Databricks 和本地開發環境，包含 Unity Catalog 和 MLflow 整合。

---

## Databricks 環境設置

### Cluster 配置

- **Node Type:** g2-standard-12 [L4]
- **Memory:** 48 GB
- **GPU:** 1x NVIDIA L4 (24GB VRAM)
- **Runtime:** Databricks ML Runtime with GPU (**16.4-gpu-ml-scala2.12**)
- **Access Mode:** Dedicated (formerly single user) - **必須用於 Unity Catalog**
- **GCP Preemptible:** 建議啟用以節省 56-70% 成本（詳見 [07-databricks-mlops.md](./07-databricks-mlops.md#gcp-preemptible-instances推薦)）

### 安裝依賴

在 Databricks Notebook 開頭執行：

```python
# 安裝核心套件
%pip install mujoco mujoco-mjx jax[cuda12] flax optax wandb flashbax

# 驗證 JAX GPU
import jax
print(f"JAX devices: {jax.devices()}")
# 預期輸出: [CudaDevice(id=0)]

# 驗證 MuJoCo
import mujoco
print(f"MuJoCo version: {mujoco.__version__}")

# 驗證 MJX
from mujoco import mjx
print("MJX available:", hasattr(mjx, 'put_model'))
```

### MuJoCo Headless 渲染設置

Databricks 是無頭環境，需要設置 EGL：

```python
import os
os.environ['MUJOCO_GL'] = 'egl'  # 或 'osmesa'
```

### JAX/XLA 記憶體設置（重要）

為避免 JAX 佔用過多 GPU 記憶體導致 OOM，**必須在 `import jax` 之前**設置：

```python
import os

# JAX 記憶體管理
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # 只使用 75% VRAM
os.environ['JAX_PREALLOCATE'] = 'false'                 # 避免預分配導致碎片化

# 然後才 import jax
import jax
print(f"JAX devices: {jax.devices()}")
```

**為什麼用 0.75？**
- L4 有 24GB VRAM
- MJX 2048 並行環境約需 15-18GB
- 預留空間給：MuJoCo 渲染 (EGL)、PyTorch 轉換階段、意外峰值

> 💡 在 Init Script 中，這些設置已預先配置。詳見 [07-databricks-mlops.md](./07-databricks-mlops.md)。

---

## 本地開發環境

### 系統需求

- Python 3.10+
- macOS / Linux / Windows

### 安裝步驟

```bash
# 1. Clone 競賽 repo
git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git
cd booster_soccer_showdown

# 2. 創建虛擬環境
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 安裝額外開發工具
pip install wandb optuna
```

### 驗證安裝

```bash
# 測試環境是否正常
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --renderer mujoco
```

---

## GitHub → Databricks 連結

### 方式 1: Databricks Repos（推薦）

1. Fork `booster_soccer_showdown` 到你的 GitHub
2. 在 Databricks Workspace → **Repos** → **Add Repo**
3. 輸入你的 GitHub repo URL
4. 設定 GitHub Personal Access Token (PAT)

**優點：**
- 自動版本控制
- 支援 branch 切換
- 直接在 Databricks 編輯

### 方式 2: DBFS 上傳

```python
# 在 Databricks Notebook
dbutils.fs.cp("local:/path/to/code", "dbfs:/booster/code", recurse=True)
```

---

## 環境變數設置

### W&B API Key

```python
import wandb
wandb.login(key="YOUR_API_KEY")
# 或設置環境變數
os.environ['WANDB_API_KEY'] = "YOUR_API_KEY"
```

### SAI API Key（提交用）

```python
# 在 submit_sai.py 中設置
os.environ['SAI_API_KEY'] = "YOUR_SAI_KEY"
```

---

## Unity Catalog 設置（MLOps 基礎）

### 建立 Catalog 和 Schema

在 Databricks SQL Editor 或 Notebook 中執行：

```sql
-- 建立 Catalog
CREATE CATALOG IF NOT EXISTS booster_soccer;

-- 建立 Schema
CREATE SCHEMA IF NOT EXISTS booster_soccer.rl_models;
CREATE SCHEMA IF NOT EXISTS booster_soccer.experiments;

-- 建立 Volumes（用於 Checkpoint 儲存）
CREATE VOLUME IF NOT EXISTS booster_soccer.rl_models.checkpoints;
CREATE VOLUME IF NOT EXISTS booster_soccer.rl_models.artifacts;
CREATE VOLUME IF NOT EXISTS booster_soccer.rl_models.logs;
```

### Volume 目錄結構

```
/Volumes/booster_soccer/rl_models/checkpoints/
├── mjx_pretraining/
│   ├── step_500000.pkl
│   └── final_checkpoint.pkl
└── pytorch_finetuning/
    ├── model_pretrained.pt
    └── model_finetuned.pt

/Volumes/booster_soccer/rl_models/artifacts/
├── videos/
├── plots/
└── reports/
```

### MLflow 配置

```python
import mlflow

# 設置 Unity Catalog 作為 Model Registry
mlflow.set_registry_uri("databricks-uc")

# 設置實驗（使用 Workspace 路徑）
mlflow.set_experiment("/Users/<your-username>/booster_soccer_experiments")

# 驗證設置
print(f"Registry URI: {mlflow.get_registry_uri()}")
```

### 權限需求

| 操作 | 需要的權限 |
|------|-----------|
| 建立 Registered Model | `CREATE MODEL` + `USE SCHEMA` + `USE CATALOG` |
| 建立 Model Version | 必須是 Registered Model 的 Owner |
| 讀取模型 | `EXECUTE` on model |

### Secrets 設置（推薦）

使用 Databricks Secrets 儲存 API Keys：

```bash
# 1. 建立 Secret Scope
databricks secrets create-scope --scope booster_soccer

# 2. 設置 Secrets
databricks secrets put --scope booster_soccer --key wandb_api_key
databricks secrets put --scope booster_soccer --key sai_api_key
```

```python
# 3. 在 Notebook 中使用
wandb_key = dbutils.secrets.get(scope="booster_soccer", key="wandb_api_key")
sai_key = dbutils.secrets.get(scope="booster_soccer", key="sai_api_key")
```

---

## 常見問題

### Q: JAX 無法偵測到 GPU？

```python
# 檢查 CUDA 版本
!nvidia-smi

# 確保安裝正確版本的 JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Q: MuJoCo 渲染錯誤？

```python
# 設置 headless 渲染
import os
os.environ['MUJOCO_GL'] = 'egl'

# 如果 EGL 不可用，試試 OSMesa
os.environ['MUJOCO_GL'] = 'osmesa'
```

### Q: sai_mujoco 套件找不到？

```bash
pip install sai-mujoco
```

---

## Docker 環境（Day 1 推薦）

**Day 1 就使用 Docker 環境**，以獲得最佳的穩定性和啟動速度。

### 為什麼 Day 1 就用 Docker？

| 原因 | 說明 |
|------|------|
| **環境一致性** | 避免「昨天能跑，今天套件升級就掛了」 |
| **啟動速度** | 省去每次 `pip install jax[cuda12]` 的 5-10 分鐘 |
| **系統依賴** | MuJoCo EGL 渲染需要系統級庫，Docker 內最穩定 |
| **搭配 Instance Pool** | 預載 Docker Image 可進一步加速 Warm Start |

### Dockerfile

```dockerfile
# 基於 Databricks 官方 GPU Image（確保與 Spark 相容）
FROM databricksruntime/gpu-ml:16.4-LTS

# 安裝 JAX 和 MuJoCo 依賴
RUN pip install --no-cache-dir \
    "jax[cuda12]==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install --no-cache-dir \
    mujoco mujoco-mjx brax flax optax wandb pynvml flashbax

# 安裝 EGL 渲染所需的系統庫
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# 設置環境變數
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
ENV JAX_PREALLOCATE=false
ENV MUJOCO_GL=egl
```

### Build & Push

```bash
# Build
docker build -t your-registry/booster-rl:v1 .

# Push 到 Container Registry（Docker Hub, GCR, etc.）
docker push your-registry/booster-rl:v1
```

### 在 Databricks 中使用

1. **Compute** → **Create Compute**
2. **Runtime Version:** 選擇 **16.4 LTS ML**
3. **Advanced Options** → **Docker** 分頁
4. 勾選 **Use your own Docker container**
5. **Docker Image URL:** `your-registry/booster-rl:v1`
6. 如果是 Private Registry，設定認證（建議用 Databricks Secrets）

### 最佳實踐

| 原則 | 說明 |
|------|------|
| **環境歸 Docker** | Python 套件、系統庫固定在 Image 中 |
| **邏輯歸 Workspace** | 訓練腳本、XML 放在 Databricks Repos 或 Workspace |
| **不要打包代碼** | 修改 Reward Function 不需要重 Build Docker |

### 快速開始流程

1. Build Docker Image 並 Push 到 Registry
2. 在 Databricks 建立 Cluster 並選擇此 Image
3. 驗證環境（JAX GPU + MuJoCo）
4. 開始開發

---

## 備用方案：Init Script

如果 Docker Image 建置遇到困難，可暫時使用 Init Script。

### Init Script (install_mjx.sh)

```bash
#!/bin/bash
# 核心套件安裝
pip install "jax[cuda12]==0.4.23" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install mujoco mujoco-mjx brax flax optax wandb pynvml flashbax

# JAX/XLA 記憶體和效能設置
echo "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75" >> /etc/profile
echo "export JAX_PREALLOCATE=false" >> /etc/profile
echo "export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda'" >> /etc/profile
echo "export MUJOCO_GL=egl" >> /etc/profile
```

### 上傳到 Unity Catalog Volume

```bash
# 建立 scripts 目錄
databricks fs mkdir /Volumes/booster_soccer/rl_models/scripts

# 上傳 Init Script
databricks fs cp scripts/install_mjx.sh /Volumes/booster_soccer/rl_models/scripts/install_mjx.sh
```

### 在 Cluster 配置中使用

```json
{
  "init_scripts": [
    "/Volumes/booster_soccer/rl_models/scripts/install_mjx.sh"
  ]
}
```

> **注意**：Init Script 每次啟動都會重新執行 pip install，會增加 5-10 分鐘的啟動時間。建議盡快切換到 Docker 環境。

---

## 下一步

環境設置完成後，前往 [02-mjx-training.md](./02-mjx-training.md) 開始 MJX 環境建立。
