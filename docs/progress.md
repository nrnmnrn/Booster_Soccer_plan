# 開發進度

> 此文件用於 Claude Code sessions 之間的交接。每次結束時更新。

---

## 當前狀態

**階段**：Week 1 Day 1 - 環境設置
**日期**：2026-01-07

---

## 上次 Session 摘要

### 2026-01-07

**完成項目**：
- ✅ 解決 Databricks 套件安裝問題
  - NumPy ABI 不兼容 → `numpy<2` 約束
  - JAX 版本不匹配 → 明確指定三件套 `jax==jaxlib==plugin==0.4.38`
  - Libraries UI 問題 → 改用 requirements.txt + Cluster Library
- ✅ 創建 `requirements.txt`（固定所有依賴版本）
- ✅ 建立 `docs/troubleshooting.md`（避雷指南）
- ✅ 解決 MuJoCo 渲染錯誤
  - EGL 初始化失敗 → `MUJOCO_GL=osmesa` 或 `disabled`
  - 環境變數衝突 → 同時設 `MUJOCO_GL` + `PYOPENGL_PLATFORM`
  - `_enums` 缺失 → `pip install --no-cache-dir` 重裝

**關鍵發現**：
- Databricks 預裝 pandas 1.5.3（用 NumPy 1.x 編譯），NumPy 2.x 會導致 ABI 崩潰
- `jax[cuda12]` extras 語法不固定 jaxlib 版本，必須明確指定
- MuJoCo import 時自動初始化 OpenGL，必須在 import **之前**設環境變數

---

## 下一步行動

> 下一個 Claude session 應執行以下任務

### 優先級 1：驗證環境
```python
# 在 Databricks Notebook 執行
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import jax, jaxlib
print(f"JAX {jax.__version__}, jaxlib {jaxlib.__version__}")
print(f"Devices: {jax.devices()}")  # 預期: [CudaDevice(id=0)]

import mujoco
from mujoco import mjx
print(f"MuJoCo {mujoco.__version__}")
```

### 優先級 2：載入官方 XML
```python
xml_path = "mimic/assets/booster_t1/booster_lower_t1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
print(f"Bodies: {model.nbody}, Joints: {model.njnt}")
```

### 優先級 3：開始 Preprocessor JAX 翻譯
- 參考 `training_scripts/main.py` 的 `Preprocessor` 類
- 創建 `src/preprocessor_jax.py`
- 目標：87 維 observation 輸出

---

## 待解決問題

| 問題 | 狀態 | 備註 |
|------|------|------|
| OSMesa 是否正常運作？ | 待驗證 | 如果失敗，改用 `MUJOCO_GL=disabled` |
| MJX 能否載入官方 XML？ | 待驗證 | 可能需要移除不支援的功能 |

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
