# Troubleshooting 避雷指南

本文檔記錄開發過程中遇到的常見問題與解決方案，幫助快速排除錯誤。

---

## 環境問題

### Python / Conda

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

### CUDA / GPU

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

### Databricks

| 問題 | 解決方案 |
|------|----------|
| `numpy.dtype size changed (Expected 96, got 88)` | NumPy 2.x ABI 不兼容。使用 `numpy<2` 約束，詳見 [套件安裝摘要](#databricks-套件安裝摘要) |
| Libraries UI 安裝後套件版本錯亂 | 不要用 UI 逐一安裝，改用 requirements.txt 作為 Cluster Library |

---

## 套件衝突

### JAX / MJX

| 問題 | 解決方案 |
|------|----------|
| `jax_cuda12_plugin not compatible with jaxlib` + Segfault | JAX 三件套版本必須一致：`jax==0.4.38 jaxlib==0.4.38 jax-cuda12-plugin==0.4.38` |
| `jax requires jaxlib >= 0.8.2` | 其他套件升級了 jax。使用 requirements.txt 一次性安裝所有套件 |
| `jax[cuda12]==0.4.38` 安裝後版本不對 | extras 語法不固定 jaxlib 版本，必須明確指定三個套件版本 |
| `vmap got inconsistent sizes for array axes` | 常數陣列（如 `player_team`）未廣播到 batch 維度。用 `jnp.tile(arr[None,:], (num_envs,1))` |
| `IndexError: f argument "3" with type "q"` | `mjx.step()` 不支援批量輸入。需用 `jax.vmap(mjx.step, in_axes=(None, 0))` 包裝 |

### PyTorch

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

### SAI 官方環境

| 問題 | 解決方案 |
|------|----------|
| `Environment 'LowerT1GoaliePenaltyKick' doesn't exist` | **SAI 套件版本太舊**！需要最新版 sai-mujoco/sai-rl + NumPy 2.x。見下方說明 |
| `TypeError: SAIClient.__init__() got an unexpected keyword argument 'scene_id'` | sai-rl 版本太舊。需升級到最新版（0.1.36+） |
| `No module named 'sai_rl'` | 在 `requirements.txt` 添加 `sai-rl`，然後重新安裝 Cluster Library |
| `AuthenticationError: No API key provided` | 設置 `SAI_API_KEY` 環境變數或傳入 `api_key` 參數 |

> **⚠️ SAI 套件版本要求（2026-01）**：
> - 官方環境需要 **NumPy 2.x**（numpy==2.1.3）
> - sai-mujoco 0.1.4 / sai-rl 0.1.5 是 2025-06 的舊版本，無法正常運作
> - 必須使用最新版並配合 NumPy 2.x，不能用 `numpy<2` 約束
> - 如果 pandas ABI 崩潰，在 notebook 開頭執行 `%pip install --upgrade pandas`

### MuJoCo

| 問題 | 解決方案 |
|------|----------|
| `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated` | MuJoCo API 返回 ndim=1 陣列。用 `.item()` 而非 `int()` 取值，如 `mj_model.joint(i).type.item()` |
| `'NoneType' has no attribute 'eglQueryString'` | Headless 環境。在 import mujoco **之前**設 `MUJOCO_GL=disabled` |
| `Cannot use OSMesa... PYOPENGL_PLATFORM is 'egl'` | **Databricks 無 OSMesa**。改用 `MUJOCO_GL=disabled`（MJX 訓練不需要渲染） |
| `'NoneType' has no attribute 'eglGetCurrentContext'` | OpenGL 已被其他套件 import。`restartPython()` 後**第一行**設環境變數 |
| `module 'mujoco' has no attribute '_enums'` | 安裝損壞。`pip uninstall mujoco mujoco-mjx -y` 後 `pip install --no-cache-dir` |
| `(mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented` | **MJX 不支援 cylinder**！將 XML 中所有 `type="cylinder"` 改為 `type="capsule"`，見下方說明 |

> **⚠️ MJX 碰撞限制**：MJX 只支援部分 geom 碰撞對。**Cylinder 與任何類型都不支援**。
> - ✅ 支援：sphere, capsule, box, plane 互相組合（大多數）
> - ❌ 不支援：cylinder, mesh, hfield
> - 解法：用 `capsule` 替代 `cylinder`（`size="r"` + `fromto="..."` 語法）

> **⚠️ Databricks 重要**：必須使用 `MUJOCO_GL=disabled`，OSMesa 在 Databricks Runtime 不可用。

---

## 邏輯錯誤

### Preprocessor

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

### 獎勵函數

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

### 模型轉換 (JAX → PyTorch)

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

---

## Databricks 套件安裝摘要

### 問題根源

| 問題 | 原因 |
|------|------|
| SAI 環境無法載入 | sai-mujoco/sai-rl 需要 NumPy 2.x，舊版本無法正常註冊環境 |
| NumPy ABI 不兼容 | Databricks 預裝 pandas 可能與 NumPy 2.x 不兼容（需升級 pandas） |
| Libraries UI 問題 | 逐一安裝套件，後安裝的會觸發依賴升級 |

### 解決方案：requirements.txt + Cluster Library

1. 建立 `requirements.txt`（見專案根目錄）
2. 上傳到 Workspace：`/Workspace/Users/<email>/booster-soccer/requirements.txt`
3. **Compute** → cluster → **Libraries** → **Install New** → **Workspace** → 選擇檔案

**關鍵原則**：
- **官方 SAI 套件優先**，使用最新版 + NumPy 2.x
- 其他套件**不指定版本**，讓 pip 自動解決依賴
- 所有套件在**同一個 pip 事務**安裝

**如果 pandas 崩潰**：
```python
%pip install --upgrade pandas
dbutils.library.restartPython()
```

### 驗證腳本

```python
import numpy as np
import sai_rl
print(f"numpy={np.__version__}, sai_rl={sai_rl.__version__}")

# 測試 SAI 環境
from sai_rl import SAIClient
sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie", api_key="YOUR_KEY")
env = sai.make_env(render_mode=None)
obs, info = env.reset()
print(f"task_index: {info.get('task_index')}")
```

---

## pip 快取問題

當遇到「安裝成功但 import 失敗」的詭異情況，可能是 pip 快取損壞：

| 指令 | 用途 |
|------|------|
| `pip install --no-cache-dir <pkg>` | 繞過快取，強制重新下載 |
| `pip cache purge` | 清空整個 pip 快取 |

**常見原因**：下載中斷導致不完整 wheel、不同 Python 版本 wheel 混用、多次安裝/解除安裝殘留。

---

## 如何新增條目

當遇到並解決問題後，請按以下格式新增：

```markdown
| `錯誤訊息或問題描述` | 解決方案說明。如有相關連結可附上。 |
```

**範例：**

```markdown
| `CUDA out of memory` | 降低 batch size 或設置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` |
```
