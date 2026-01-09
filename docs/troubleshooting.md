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
| JIT 編譯時 GPU OOM + Disk 暴增 | `num_envs=2048` 太大。降至 512，JIT 通過後可嘗試 1024。見 `config.py:57` |
| `DEBUG:ThreadMonitor` 無限重複 | JIT 編譯中，非錯誤。L4 GPU 首次編譯 MJX 需 5-15 分鐘 |

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
| `Shapes must be 1D sequences of concrete values` + `JitTracer` | JIT 函數中使用動態值確定 array shape。用 `@partial(jax.jit, static_argnames=('param_name',))` 將參數標記為靜態 |
| `TracerBoolConversionError` + `if param:` | JIT 函數中使用動態 bool 做條件判斷。用 `static_argnames=('param',)` 標記為靜態，或改用 `jax.lax.cond` |

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
| `sai.make_env("LowerT1KickToTarget-v0")` 失敗 | **官方 Bug（2026-01-08）**。使用 `comp_id` 參數初始化 SAIClient 後的 `make_env()` 無法直接指定環境。需聯繫官方或使用 workaround。 |
| `/soccer_ball` body name 找不到 | dm_control/PyMJCF attach 時自動加 `/` 前綴。需 patch sai_mujoco 源碼或使用帶前綴名稱。 |
| `mat1 and mat2 shapes cannot be multiplied (1x45 and 87x256)` | **Preprocessor 簽名錯誤**！SAI 框架調用 `modify_state(obs, info)` 但你的函數是 `modify_state(obs, info, task_one_hot)`。從 `info['task_index']` 內部獲取 task_one_hot。見下方說明 |

> **⚠️ Preprocessor API 契約（2026-01）**：
> - SAI 框架調用 `preprocessor.modify_state(obs, info)` — **只有 2 個參數**
> - 如果你的簽名是 `(obs, info, task_one_hot)`，會導致 TypeError 或 preprocessor 不被執行
> - **解法**：在函數內部從 `info['task_index']` 獲取 task_one_hot
> ```python
> def modify_state(self, obs, info):
>     task_one_hot = self.get_task_onehot(info)  # 內部獲取
>     # ... 其餘處理
> ```

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
| Quaternion 順序混淆 | **sai_mujoco 使用 [x,y,z,w]**，而非 MuJoCo 標準 [w,x,y,z]。見 `football.py` 的 `data[[1,2,3,0]]` 轉換 |
| 三個環境原始 obs 維度不同 | LowerT1KickToTarget-v0: **39維**，LowerT1GoaliePenaltyKick-v0: **45維**，LowerT1ObstaclePenaltyKick-v0: **54維**。Preprocessor 會統一為 87 維 |
| task_one_hot 維度（MJX 預訓練） | **7 維**。前 3 維為任務 one-hot（GoaliePK=[1,0,0,0,0,0,0]），後 4 維填 0。這使 80 + 7 = 87 維符合官方期望。見 ADR-0001 |
| `Incompatible shapes (2048, 83) vs (2048, 87)` | MJX 環境 task_one_hot 維度不足。擴展至 7 維即可解決。見 ADR-0001 |
| defender_xpos 維度 | 需驗證：可能是 **9 維**（3 defenders × 3 coords），不是 3 維 |

> **官方 Preprocessor 結構（87 維）**：
> 來源：`booster_soccer_showdown/imitation_learning/scripts/preprocessor.py`
> ```
> robot_qpos(12) + robot_qvel(12) + project_gravity(3) + robot_gyro(3) +
> accelerometer(3) + velocimeter(3) + goal_info(12) + ball_info(9) +
> player_team(2) + goalkeeper_info(12) + target_info(6) + defender_xpos(?) +
> task_one_hot(3) = 87
> ```

### 獎勵函數

| 問題 | 解決方案 |
|------|----------|
| *尚無記錄* | - |

### SAC 訓練

| 問題 | 解決方案 |
|------|----------|
| `Q: nan` 和 `α: nan` 在訓練輸出 | **Tanh 邊界問題**：action = ±1 時 `log(1-tanh²) = log(0) = NaN`。使用 `SafeTanh` bijector 修復 |
| `RuntimeWarning: overflow encountered in cast` | MJX 物理模擬產生極大值。通常是正常現象，不影響訓練 |
| obs 範圍過大（如 max=75） | 在 `soccer_env.py` 添加 `jnp.clip(obs, -10, 10)` |
| `NotImplementedError: mode is not implemented for this transformed distribution` | **distrax Tanh bijector 限制**：自定義 `SafeTanh` 無法使用 `.mode()`。改用 `jnp.tanh(dist.distribution.mean())` 手動計算。見 `sac_agent.py:168, 359` |
| 訓練崩潰但無 checkpoint（如 140k 步後） | **save_frequency 太大**：預設 500k 步太稀疏。建議設為 50k 步。見 `config.py:89` |

> **⚠️ SafeTanh 修復詳情**：
>
> **根本原因**：distrax 的 `Tanh()` 計算 `log(1 - tanh²(x))`。當 `tanh(x) = ±1` 時，`log(0) = -inf → NaN`。
>
> **修復**：`sac_agent.py` 定義 `SafeTanh` 類，添加 epsilon：
> ```python
> def forward_log_det_jacobian(self, x):
>     return jnp.log(1.0 - jnp.tanh(x)**2 + 1e-6)  # 添加 1e-6
> ```

### 模型轉換 (JAX → PyTorch)

| 問題 | 解決方案 |
|------|----------|
| `TypeError: the read only flag is not supported, should always be False` | JAX array 是 read-only。轉換時使用 `np.asarray(jax_array).copy()` 再傳給 `torch.tensor()` |

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
