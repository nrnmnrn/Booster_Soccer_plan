# 開發進度

> 此文件用於 Claude Code sessions 之間的交接。每次結束時更新。

---

## 當前狀態

**階段**：Week 1 Day 2 - MJX 環境開發
**日期**：2026-01-07

---

## 上次 Session 摘要

### 2026-01-07 (Session 3)

**完成項目**：
- ✅ Databricks GPU 驗證成功（JAX 0.4.38 + CudaDevice）
- ✅ MJXSoccerEnv reset/step 正常運作（1024 並行環境）
- ✅ 修復 MJX 碰撞限制（cylinder → capsule）
- ✅ 修復 vmap batch 維度錯誤（player_team/task_one_hot 廣播）
- ✅ 修復 mjx.step 批量執行（需 vmap 包裝）

**待解決**：
- ⚠️ obs shape 是 83 而非 87（缺少 4 維，待調查官方 preprocessor）

**關鍵發現**：
- MJX 不支援 `cylinder` geom 類型，必須用 `capsule` 替代
- `mjx.step()` 只接受單一環境，批量需用 `jax.vmap(mjx.step, in_axes=(None, 0))`
- 常數陣列必須用 `jnp.tile` 廣播到 batch 維度才能參與 vmap

---

### 2026-01-07 (Session 2)

**完成項目**：
- ✅ 確認 `MUJOCO_GL=disabled` 為 Databricks 唯一選項（OSMesa 不可用）
- ✅ 創建環境驗證 Notebook (`src/notebooks/01_environment_validation.ipynb`)
- ✅ 創建 XML 載入測試 Notebook (`src/notebooks/02_xml_loading.ipynb`)
- ✅ 創建 MJX 足球環境 XML (`src/mjx/assets/soccer_env.xml`)
  - 12 DOF 機器人（簡化幾何體，保持原始質量分佈）
  - 足球 + 2 個球門
  - 與官方 actuator 範圍一致
- ✅ 實現 JAX Preprocessor (`src/mjx/preprocessor_jax.py`)
  - 87 維輸出，與官方 NumPy 版本對齊
  - 支持批量處理 (vmap)
  - 包含四元數格式轉換 (wxyz → xyzw)
- ✅ 實現 MJX 環境類 (`src/mjx/soccer_env.py`)
  - 使用 `mj_name2id` 獲取 body ID（禁止硬編碼）
  - reset/step 接口
  - **待完成**：reward 函數設計

**關鍵發現**：
- 官方 Preprocessor 使用四元數 `[x,y,z,w]` 格式，但 MuJoCo 返回 `[w,x,y,z]`
- `booster_lower_t1.xml` 使用 `<include>` 引用外部文件，需要注意路徑
- MJX 訓練不需要渲染，`MUJOCO_GL=disabled` 是最乾淨的選項

### 2026-01-07 (Session 1)

**完成項目**：
- ✅ 解決 Databricks 套件安裝問題
  - NumPy ABI 不兼容 → `numpy<2` 約束
  - JAX 版本不匹配 → 明確指定三件套 `jax==jaxlib==plugin==0.4.38`
  - Libraries UI 問題 → 改用 requirements.txt + Cluster Library
- ✅ 創建 `requirements.txt`（固定所有依賴版本）
- ✅ 建立 `docs/troubleshooting.md`（避雷指南）

**關鍵發現**：
- Databricks 預裝 pandas 1.5.3（用 NumPy 1.x 編譯），NumPy 2.x 會導致 ABI 崩潰
- `jax[cuda12]` extras 語法不固定 jaxlib 版本，必須明確指定
- MuJoCo import 時自動初始化 OpenGL，必須在 import **之前**設環境變數

---

## 下一步行動

> 下一個 Claude session 應執行以下任務

### 優先級 1：調查 obs 維度差異（83 vs 87）
- 對比官方 `preprocessor.py` 找出缺少的 4 維
- 修正 `src/mjx/preprocessor_jax.py`

### 優先級 2：設計 Reward 函數
- 在 `soccer_env.py` 的 `_compute_reward` 方法中實現
- 建議從簡單獎勵開始：站立獎勵 + 朝向球獎勵
- 參考官方 benchmark 的獎勵設計

### 優先級 3：實現 PPO 訓練循環
- 創建 `src/training/ppo_mjx.py`
- 使用 JAX 實現 PPO（或使用 brax/mujoco_playground 的實現）

---

## 待解決問題

| 問題 | 狀態 | 備註 |
|------|------|------|
| ~~OSMesa 是否正常運作？~~ | ✅ 已解決 | 必須用 `MUJOCO_GL=disabled` |
| ~~MJX 能否載入 soccer_env.xml？~~ | ✅ 已解決 | 需改 cylinder → capsule |
| obs 維度 83 vs 87 | ⚠️ 待調查 | 缺少 4 維，需對比官方 preprocessor |
| Reward 函數設計 | 待設計 | 需要人工參與設計 |

---

## 環境狀態

| 項目 | 狀態 |
|------|------|
| Databricks Runtime | 16.4 LTS ML (Python 3.12, CUDA 12.6) |
| requirements.txt | 已創建，JAX 0.4.38 + numpy<2 |
| Cluster Library | 需用 requirements.txt 安裝（不要用 UI 逐一安裝） |
| Docker | 未使用（Databricks 限制） |

---

## 相關文件

- [troubleshooting.md](./troubleshooting.md) - 錯誤解決方案
- [01-environment-setup.md](./01-environment-setup.md) - 環境設置指南
- [README.md](./README.md) - 完整進度 checklist

### MJX 環境文件

| 文件 | 說明 |
|------|------|
| `src/mjx/soccer_env.py` | MJX 環境類（reset/step） |
| `src/mjx/preprocessor_jax.py` | JAX 版本 Preprocessor |
| `src/mjx/assets/soccer_env.xml` | 12 DOF 機器人 + 球 + 球門 |
| `src/notebooks/01_environment_validation.ipynb` | Databricks 環境驗證 |
