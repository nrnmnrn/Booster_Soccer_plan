# Databricks notebook source
# MAGIC %md
# MAGIC # SAC MJX 訓練腳本
# MAGIC
# MAGIC 在 Databricks L4 GPU 上執行 SAC 訓練。
# MAGIC
# MAGIC **Cluster 要求**（見 docs/01-environment-setup.md）：
# MAGIC - Runtime: **17.3 LTS ML** 或 **16.4-gpu-ml-scala2.12**
# MAGIC - Node Type: **g2-standard-12** (L4 GPU, 24GB)
# MAGIC - Workers: 0 (Single Node)
# MAGIC - Cluster Library: 使用 `requirements.txt` 安裝（不需要 %pip）
# MAGIC
# MAGIC **前置條件**：
# MAGIC 1. 已建立 Unity Catalog Volume：`/Volumes/booster_soccer/rl_models/checkpoints`
# MAGIC 2. 已設置 W&B API Key（如使用 W&B）
# MAGIC 3. 專案已上傳到 Databricks Repos 或 Workspace

# COMMAND ----------

import os

# === W&B 登入（從 Databricks Secrets 讀取 API Key）===
try:
    import wandb
    wandb_key = dbutils.secrets.get(scope="booster_soccer", key="wandb_api_key")
    # 使用 wandb.login() 明確登入，relogin=True 強制重新認證
    wandb.login(key=wandb_key, relogin=True)
    print("✓ W&B 登入成功")
except Exception as e:
    print(f"⚠️ W&B 登入失敗: {e}")
    print("  如果不需要 W&B，請設置 use_wandb=False")

# JAX/XLA 記憶體設置（必須在 import jax 之前）
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["JAX_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "disabled"

# COMMAND ----------

# 驗證 JAX GPU
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
assert len(jax.devices('gpu')) > 0, "No GPU detected! 請確認使用 GPU Runtime"

# COMMAND ----------

import sys

# =============================================================================
# 設置專案路徑（請根據實際情況選擇一種方式）
# =============================================================================

# 方法 1: 專案在 Databricks Repos（推薦）
# project_root = "/Workspace/Repos/<your-username>/Booster_Soccer_plan"

# 方法 2: 專案在 Workspace Files
# project_root = "/Workspace/Users/<your-email>/Booster_Soccer_plan"

# === 請取消註解並修改為你的路徑 ===
project_root = "/Workspace/Users/adamlin@cheerstech.com.tw/.bundle/Booster_Soccer_plan/dev/files/"
# =============================================================================

sys.path.insert(0, project_root)
print(f"Project root: {project_root}")

# 驗證路徑
import os
if not os.path.exists(os.path.join(project_root, "src")):
    raise FileNotFoundError(
        f"專案路徑不正確: {project_root}\n"
        "請修改上方的 project_root 變數為正確的路徑"
    )
print("✓ 專案路徑驗證成功")

# COMMAND ----------

# 配置訓練參數
from src.training.config import SACConfig

config = SACConfig(
    # === 環境 ===
    num_envs=2048,              # L4 GPU 推薦值
    max_episode_steps=1000,

    # === 訓練 ===
    total_timesteps=10_000_000, # 10M steps
    learning_starts=10_000,     # 前 10k 步隨機探索
    batch_size=256,
    buffer_size=1_000_000,

    # === Domain Randomization ===
    dr_level=1,                 # Level 1: 基礎（±5%）
    random_task_index=True,     # 隨機任務（重要：確保泛化能力）

    # === Checkpoint（使用 Unity Catalog Volume）===
    save_frequency=200_000,     # 每 200k 步保存（配合 Preemptible 縮短間隔）
    checkpoint_dir="/Volumes/booster_soccer/rl_models/checkpoints/mjx_pretraining",

    # === 監控 ===
    use_mlflow=True,            # Databricks 原生整合
    use_wandb=True,             # 實時監控（需設置 WANDB_API_KEY）
    mlflow_experiment="/Users/adamlin@cheerstech.com.tw/booster_soccer_experiments",  # 請修改
    wandb_project="booster_soccer_mjx",

    # === 隨機種子 ===
    seed=42,
)

print("Training config:")
for k, v in config.to_dict().items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 開始訓練
# MAGIC
# MAGIC 訓練會自動：
# MAGIC - 記錄指標到 MLflow 和 W&B
# MAGIC - 每 200k 步保存 checkpoint 到 Unity Catalog Volume
# MAGIC - 支持 Preemptible 機器被搶佔後的自動恢復

# COMMAND ----------

# 開始訓練
from src.training.train_sac import train_sac

state, checkpoint_path = train_sac(config)

print(f"\n{'='*60}")
print(f"✅ 訓練完成！")
print(f"📁 Checkpoint: {checkpoint_path}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 下一步：JAX → PyTorch 轉換
# MAGIC
# MAGIC 訓練完成後，使用官方 `jax2torch.py` 腳本轉換模型：
# MAGIC
# MAGIC ```python
# MAGIC # 方法 1: 如果已安裝 booster_soccer_showdown
# MAGIC from booster_soccer_showdown.imitation_learning.scripts.jax2torch import convert
# MAGIC convert(pkl_path=checkpoint_path, output_path="/Volumes/booster_soccer/rl_models/checkpoints/pytorch_finetuning/model_pretrained.pt")
# MAGIC
# MAGIC # 方法 2: 直接運行腳本
# MAGIC # %sh python booster_soccer_showdown/imitation_learning/scripts/jax2torch.py \
# MAGIC #     --pkl {checkpoint_path} \
# MAGIC #     --out /Volumes/booster_soccer/rl_models/checkpoints/pytorch_finetuning/model_pretrained.pt
# MAGIC ```
# MAGIC
# MAGIC **重要**：轉換時只取 Actor 的 mean 部分（前 12 維），捨棄 log_std。
# MAGIC 詳見 `docs/07-databricks-mlops.md` Job 3: Model Conversion。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 從 Checkpoint 恢復訓練
# MAGIC
# MAGIC 如果訓練被中斷（例如 Preemptible 機器被搶佔），可以從最近的 checkpoint 恢復：
# MAGIC
# MAGIC ```python
# MAGIC from src.training.train_sac import train_sac
# MAGIC
# MAGIC # 找到最新的 checkpoint
# MAGIC checkpoint_dir = "/Volumes/booster_soccer/rl_models/checkpoints/mjx_pretraining"
# MAGIC import os
# MAGIC checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
# MAGIC latest = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
# MAGIC resume_path = os.path.join(checkpoint_dir, latest)
# MAGIC
# MAGIC # 恢復訓練
# MAGIC state, final_path = train_sac(config, resume_from=resume_path)
# MAGIC ```
