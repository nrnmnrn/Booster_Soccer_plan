# Claude 專案上下文

## 專案目標

Booster Soccer Showdown 機器人足球競賽（獎金 $10,000），訓練單一策略在三個環境中泛化。

## 用戶背景

- **ML 經驗**：有基礎，但不熟悉 RL
- **硬體**：Databricks L4 GPU (24GB) | Python 3.12
- **時間**：2-4 週
- **團隊**：單人參賽

## 核心策略

```
mimic/assets XML → MJX (GPU) → 簡化獎勵預訓練 → jax2torch → 官方環境微調 → 提交
```

---

## 關鍵約束（必須遵守）

> Claude 在**每次代碼修改前**必須檢查這些規則

| 約束 | 說明 |
|------|------|
| Preprocessor 維度 | **必須是 87 維**（與 DDPG 訓練一致） |
| Quaternion 順序 | **[w, x, y, z]**（MuJoCo 標準） |
| SAC→DDPG 轉換 | **只取 mean**（前 12 維），捨棄 log_std |
| Body ID 獲取 | **使用 `mj_name2id`**，禁止硬編碼 |
| 禁止修改 | `.env`, `credentials`, 機密文件 |

---

## 當前進度

詳見 `docs/README.md`「進度追蹤」區塊。

---

## 常用命令

| 場景 | 命令 | 說明 |
|------|------|------|
| 功能開發 | `/feature-dev` | 導向式開發流程 |
| 文檔同步 | `/sync-docs` | 檢查 ADR/workflow/troubleshooting 更新 |
| RL 諮詢 | `databricks-rl-mentor` | 審查 RL 技術決策 |

---

## 文件索引

| 類別 | 路徑 | 說明 |
|------|------|------|
| 計劃總覽 | `docs/README.md` | 進度 checklist |
| 技術文檔 | `docs/01-07*.md` | 環境設置、訓練、微調、提交 |
| 架構決策 | `docs/adr/` | 重大技術決策記錄 |
| 流程圖表 | `docs/workflows/` | Mermaid 視覺化流程 |
| 避雷指南 | `docs/troubleshooting.md` | 報錯解決方案 |

---

## 溝通偏好

- **語言**：繁體中文為主，技術術語可用英文
- **風格**：簡潔直接，避免冗長解釋
- **代碼**：遵循現有 repo 風格
