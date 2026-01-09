# 開發進度

> 此文件用於 Claude Code sessions 之間的交接。每次結束時更新。

---

## 當前狀態

**階段**：Week 2 - jax2torch 轉換完成，準備官方環境驗證
**日期**：2026-01-09

---

## 上次 Session 摘要

### 2026-01-09 (Session 8)

**完成項目**：
- ✅ **完成 jax2torch 轉換規劃與實作**
  - 分析官方 `jax2torch.py` 和 `model.py` 結構
  - 確認 SAC checkpoint 與官方格式完全兼容
  - 發現 `BoosterModel` 使用 `model(obs)[0]` 自動只取 mean
- ✅ **創建轉換 Notebook**
  - `src/notebooks/04_jax2torch_conversion.ipynb`
  - 支援在 Databricks 上直接執行
  - 包含結構驗證 + 權重轉換 + TorchScript 保存
- ✅ **準備提交文件結構**
  - `submission/model.py` - BoosterModel 包裝
  - `submission/preprocessor.py` - 87 維預處理器
  - `scripts/test_in_official_env.py` - 官方環境測試腳本
- ✅ **確認 Checkpoint 位置**
  - 路徑：`/Workspace/Users/adamlin@cheerstech.com.tw/.bundle/Booster_Soccer_plan/dev/files/exp/sac_mjx/checkpoints/u4oasfsj/final_checkpoint.pkl`
  - run_id: `u4oasfsj`

**關鍵發現**：
- **不需要手動截斷 log_std**：`TorchGCActor.forward()` 返回 `(mean, std)`，但 `BoosterModel` 只使用 `[0]` 即 mean
- **權重轉換需轉置**：JAX kernel shape `(in, out)` → PyTorch weight shape `(out, in)`
- **動態層探測**：`jax2torch.py` 自動探測 `Dense_0`, `Dense_1`... 順序，適應不同網路架構

**Checkpoint 結構確認**：
```
checkpoint["agent"]["network"]["params"]["modules_actor"]
├── actor_net/
│   ├── Dense_0, Dense_1, Dense_2  (權重)
│   └── LayerNorm_0, LayerNorm_1, LayerNorm_2
├── mean_net  (動作輸出)
└── log_std_net  (保留但不使用)
```

---

### 2026-01-09 (Session 7)

**完成項目**：
- ✅ **修復 `dist.mode()` NotImplementedError**
  - distrax Tanh-transformed distribution 不支援 `.mode()`
  - 改用 `jnp.tanh(dist.distribution.mean())` 手動計算
  - 修復位置：`sac_agent.py:168, 359`
- ✅ **優化訓練配置**
  - `batch_size`: 256 → 512（減少梯度噪聲）
  - `updates_per_step`: 1 → 4（提高樣本效率）
  - `save_frequency`: 500k → 50k（避免訓練損失）
- ✅ **完成 180k 步 SAC 訓練**（~20 分鐘）
  - log_alpha: -4.9（α ≈ 0.007，策略趨向確定性）
  - Q1_mean: 3 → 21（Critic 學習正常）
  - Episode Reward: 44.5-46.5（波動，無明顯上升）
  - GPU Memory: ~75% (18GB)，Temperature: 73-78°C
- ✅ **更新文檔**
  - troubleshooting.md：新增 2 個錯誤記錄
  - README.md：重構進度追蹤

**訓練結果評估**：
| 指標 | 值 | 狀態 |
|------|-----|------|
| 訓練完成 | 180k 步 | ✅ |
| Deterministic Action | Shape (12,), Range [-0.345, 0.287] | ✅ |
| Q 值學習 | 持續上升 | ✅ |
| Episode Reward | 無明顯上升 | ⚠️ 需觀察 |

**關鍵發現**：
- Episode reward 沒有明顯上升趨勢，可能是：
  1. 獎勵函數設計問題（需要 reward shaping）
  2. 訓練時間不足（180k 步可能不夠）
  3. MJX 環境與官方環境差異
- 建議：先完成 jax2torch 轉換，在官方環境驗證行為後再決定是否重新訓練

---

### 2026-01-08 (Session 6)

**完成項目**：
- ✅ **重大發現：Quaternion 順序**
  - sai_mujoco 使用 **[x, y, z, w]** 格式
  - 不是 MuJoCo 標準的 [w, x, y, z]
  - 見 `football.py` 的 `data[[1, 2, 3, 0]]` 轉換
- ✅ **確認 task_one_hot 維度**
  - **3 維**（不是之前假設的 7 維）
  - GoaliePenaltyKick: [1, 0, 0]
  - ObstaclePenaltyKick: [0, 1, 0]
  - KickToTarget: [0, 0, 1]
- ✅ **確認三個環境原始 obs 維度**
  - LowerT1KickToTarget-v0: **39 維**
  - LowerT1GoaliePenaltyKick-v0: **45 維**
  - LowerT1ObstaclePenaltyKick-v0: **54 維**
  - Preprocessor 統一為 87 維
- ✅ **發現官方 Bug**
  - `sai.make_env("LowerT1KickToTarget-v0")` 無法使用
  - `/soccer_ball` body name 找不到（dm_control/PyMJCF 前綴問題）
- ✅ **確認官方 Imitation Learning Pipeline**
  - collect_data.py → train.py → jax2torch.py → model.py → submit_sai.py
  - 完整流程可用，但需要 teleoperation 收集數據
- ✅ **更新文檔**
  - 修正 CLAUDE.md Quaternion 順序約束
  - 添加 troubleshooting.md SAI bug 記錄
  - 添加 Preprocessor 結構說明

**關鍵發現**：
- 官方 `training_scripts/main.py` 使用 `n_features=87`
- 提交模型 (`model.py`) 只用 obs 前 30 維做步態控制
- 專案已有完整 MJX 實現：`src/mjx/soccer_env.py` + `preprocessor_jax.py`

**決策**：
- 選擇 **MJX RL 方案**（不使用官方 IL 方案）
- 需要確保 MJX preprocessor 與官方 87 維格式兼容

---

### 2026-01-07 (Session 5)

**完成項目**：
- ✅ 修復 `02_xml_loading.ipynb` MJX 載入測試
  - 改用 `soccer_env.xml`（MJX 兼容版本）
  - 修復 NumPy scalar 提取警告（`int()` → `.item()`）
  - 添加 ball/goal body/site ID 獲取示範
- ✅ Databricks 驗證通過：MJX 編譯 + GPU step 成功
- ✅ 更新 troubleshooting.md（NumPy .item() 用法）

**Notebook 輸出摘要**：
- Actuators: 12 ✅
- qpos dim: 26 ✅ (robot 7 + joints 12 + ball 7)
- MJX 編譯成功
- First step (JIT): ~45s, 1000 steps: ~51s

**待完成**：
- ⚠️ 繼續 Session 4 的 task_index 維度驗證

---

### 2026-01-07 (Session 4)

**完成項目**：
- ✅ 分析 obs 維度差異原因（83 vs 87）
  - 發現：官方 preprocessor 實際輸出 83 維
  - `n_features=87` 可能是硬編碼錯誤，或 `info['task_index']` 是 7 維
- ✅ 創建維度驗證 Notebook (`src/notebooks/03_task_index_validation.ipynb`)
- ✅ 更新 `preprocessor_jax.py` 維度註解（修正計算錯誤）
- ✅ 實現 Reward 函數（6 個獎勵組件）
  - r_stand (0.4)：站立獎勵
  - r_approach (0.3)：靠近球獎勵
  - r_ball_vel (0.2)：球朝向球門速度
  - r_kick (0.05)：腳接觸球
  - r_energy (-0.01)：能量懲罰
  - r_time (-0.01)：時間懲罰

**待完成**：
- ⚠️ 在 Databricks 運行 `03_task_index_validation.ipynb` 確認真實維度
- ⚠️ 攻擊方向邏輯需確認（見 `soccer_env.py` 第 345-347 行 TODO(human)）

**關鍵發現**：
- 官方 `main.py` 的 `get_task_onehot()` 直接返回 `info['task_index']`
- imitation learning 使用手動創建的 3 維 one-hot
- 需要在官方環境驗證 `task_index` 真實維度才能確定最終答案

---

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

### 優先級 1：執行 jax2torch 轉換（在 Databricks）
轉換 Notebook 已準備好，需要在 Databricks 上執行：

**步驟**：
1. 上傳 `src/notebooks/04_jax2torch_conversion.ipynb` 到 Databricks
2. 執行 Notebook（會自動驗證結構 + 轉換權重）
3. 下載生成的 `model.pt` 到本地 `submission/` 目錄

**Checkpoint 路徑**：
```
/Workspace/Users/adamlin@cheerstech.com.tw/.bundle/Booster_Soccer_plan/dev/files/exp/sac_mjx/checkpoints/u4oasfsj/final_checkpoint.pkl
```

**輸出路徑**：
```
/Workspace/Users/adamlin@cheerstech.com.tw/.bundle/Booster_Soccer_plan/dev/files/submission/model.pt
```

### 優先級 2：官方環境驗證（Gate 3）
下載 `model.pt` 後，在本地執行測試：

```bash
python scripts/test_in_official_env.py --api-key YOUR_API_KEY --task GoaliePenaltyKick
```

**成功標準**：
- ✅ 機器人站立 > 50 步
- ✅ 有移動/踢球意圖
- ⚠️ 不要求高分數

### 優先級 3：官方環境微調（如 Gate 3 失敗）
如果機器人立即倒下或行為異常：
1. 檢查 obs 預處理是否正確（四元數順序 [x,y,z,w]）
2. 在官方環境進行 DDPG 微調
3. 調整獎勵函數重新訓練

### 優先級 4：SAI 提交
驗證通過後提交到競賽平台。

---

## 已解決問題（Session 8 更新）

| 問題 | 狀態 | 備註 |
|------|------|------|
| ~~task_one_hot 維度~~ | ✅ 已解決 | **3 維**，不是 7 維 |
| ~~Quaternion 順序~~ | ✅ 已解決 | sai_mujoco 用 **[x,y,z,w]** |
| ~~官方 Preprocessor 結構~~ | ✅ 已解決 | 87 維，見 troubleshooting.md |
| ~~jax2torch 結構兼容~~ | ✅ 已解決 | SAC checkpoint 與官方格式完全兼容 |
| ~~SAC→DDPG log_std 處理~~ | ✅ 已解決 | BoosterModel 自動只取 mean（`[0]`） |

---

## 待解決問題

| 問題 | 狀態 | 備註 |
|------|------|------|
| ~~OSMesa 是否正常運作？~~ | ✅ 已解決 | 必須用 `MUJOCO_GL=disabled` |
| ~~MJX 能否載入 soccer_env.xml？~~ | ✅ 已解決 | 需改 cylinder → capsule |
| ~~obs 維度 83 vs 87~~ | ✅ 已解決 | 87 維（task_one_hot 是 3 維不是 7 維） |
| ~~Reward 函數設計~~ | ✅ 已實現 | 6 組件獎勵，見 `soccer_env.py` |
| ~~task_one_hot 維度~~ | ✅ 已解決 | 3 維 |
| ~~Quaternion 順序~~ | ✅ 已解決 | [x,y,z,w]（sai_mujoco 格式） |
| defender_xpos 維度 | ⚠️ 待驗證 | 可能是 9 維（3×3），不是 3 維 |
| 攻擊方向邏輯 | ⚠️ 待確認 | TODO(human) 在 `soccer_env.py:345` |
| sai.make_env bug | ⚠️ 官方問題 | 無法直接指定環境，需等官方修復 |

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
| `src/mjx/soccer_env.py` | MJX 環境類（reset/step/reward） |
| `src/mjx/preprocessor_jax.py` | JAX 版本 Preprocessor |
| `src/mjx/assets/soccer_env.xml` | 12 DOF 機器人 + 球 + 球門 |
| `src/notebooks/01_environment_validation.ipynb` | Databricks 環境驗證 |
| `src/notebooks/02_xml_loading.ipynb` | MJX XML 載入測試（soccer_env.xml） |
| `src/notebooks/03_task_index_validation.ipynb` | task_index 維度驗證 |
| `src/notebooks/04_jax2torch_conversion.ipynb` | JAX → PyTorch 轉換（在 Databricks 執行） |

### 提交相關文件

| 文件 | 說明 |
|------|------|
| `submission/model.py` | BoosterModel 包裝（PD 控制器） |
| `submission/preprocessor.py` | 87 維預處理器 |
| `submission/model.pt` | 轉換後的 TorchScript 模型（待生成） |
| `scripts/test_in_official_env.py` | 官方環境測試腳本 |
