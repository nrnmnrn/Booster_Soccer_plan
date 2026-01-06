---
name: sync-docs
description: 同步開發文檔。執行 /sync-docs 檢查 ADR、workflow、troubleshooting 是否需要更新，並比對進度與原始規劃的差異。
allowed-tools: Read, Grep, Glob, Edit, Write, AskUserQuestion
---

# 文檔同步 Skill

## 觸發時機
- 用戶說 `/sync-docs`、「sync docs」、「更新文檔」
- 完成 `/feature-dev` 後

---

## Step 1：分析本次 Session 變更

讀取最近修改的檔案，判斷需要同步的文檔類型：

| 變更類型 | 目標文檔 | 判斷標準 |
|----------|----------|----------|
| 架構/演算法決策 | `docs/adr/` | 涉及 preprocessor、rewards、網路架構、轉換邏輯 |
| 一般報錯/環境問題 | `docs/troubleshooting.md` | 錯誤訊息、套件衝突、設定問題 |
| 程式流程變更 | `docs/workflows/*.md` | 主要模塊邏輯改變 |
| 關鍵約束變更 | `CLAUDE.md` | 維度、順序、格式等硬性規則 |

---

## Step 2：進度比對（與原始規劃）

**必須執行**：讀取 `@docs/README.md` 中的「進度追蹤」區塊，與當前完成狀態比對。

### 執行步驟

1. 讀取 `docs/README.md`，找到所有 `- [ ]` 和 `- [x]` 項目
2. 計算 Checkbox 完成率：
   ```
   完成率 = 已完成項目數 / 總項目數 × 100%
   ```
3. 輸出進度比對報告

### 輸出格式

```
📊 進度比對報告：

原始規劃項目：12 項
已完成 (✓)：3 項
進行中 (→)：1 項
未開始 ( )：8 項

📈 完成率：25.0%

⚠️ 差異提醒：
- [落後] 原定 Day 2 完成的「MJX 環境建立」仍在進行中
- [新增] 額外實作了「GPU 監控模塊」（未在原始規劃中）
```

---

## Step 3：ADR 建立（自動編號）

**當需要建立新的 ADR 時：**

1. 讀取 `docs/adr/` 目錄下所有現有檔案
2. 找出最大編號 N（如 `0003-xxx.md` → N=3）
3. 新 ADR 使用 `N+1` 格式命名（如 `0004-new-decision.md`）
4. 使用 `.claude/templates/adr-template.md` 作為模板

### 命名規則

- 格式：`NNNN-kebab-case-title.md`
- 例如：`0004-reward-annealing-strategy.md`

---

## Step 4：Mermaid 語法檢查

**在更新任何 `docs/workflows/*.md` 前，必須執行語法驗證。**

### 檢查規則

1. 確認有 ` ```mermaid ` 開頭和 ` ``` ` 結尾
2. 確認流程圖類型有效：`flowchart TD`, `sequenceDiagram`, `stateDiagram-v2`
3. 確認節點 ID 不含特殊字符（只允許 `A-Za-z0-9_`）
4. 確認箭頭語法正確：`-->`, `-->>`, `-.->`, `==>`
5. 確認 subgraph 有對應的 end

### 最佳實踐

- **優先使用 `flowchart TD`**（Top-Down），這是最穩定且可讀的格式
- **避免複雜 subgraph**：如果需要分組，盡量使用簡單的兩層結構
- **節點文字用引號包裹**：`A["節點文字"]` 避免特殊字符問題

### 錯誤處理

如發現語法錯誤，**禁止寫入**，改為向用戶報告問題：

```
❌ Mermaid 語法錯誤：
- 第 5 行：subgraph 缺少對應的 end
- 第 8 行：節點 ID 包含非法字符 "-"

請修正後再試。
```

---

## Step 5：列出修改清單（強制確認）

**禁止直接修改文檔**。必須先列出清單，等待用戶確認：

```
📝 文檔同步建議：

1. [新增] docs/adr/0004-reward-annealing.md
   - 記錄 Reward Annealing 策略決策

2. [修改] docs/workflows/finetuning-flow.md
   - 更新微調流程圖（新增 Annealing 階段）
   - ✅ Mermaid 語法檢查通過

3. [修改] docs/troubleshooting.md
   - 新增：「CUDA out of memory」解決方案

---
📊 進度比對：當前完成率 25%，與原始規劃一致。

是否確定執行這些更新？(y/n)
```

---

## Step 6：執行更新

獲得用戶確認後：

1. 依序建立/修改文檔
2. 更新 `docs/adr/README.md` 索引（如有新 ADR）
3. 更新 `docs/workflows/README.md` 索引（如有新流程圖）
4. 報告完成狀態

### 完成訊息

```
✅ 文檔同步完成！

已建立：
- docs/adr/0004-reward-annealing.md

已修改：
- docs/workflows/finetuning-flow.md
- docs/troubleshooting.md

已更新索引：
- docs/adr/README.md
```

---

## 附錄：ADR vs Troubleshooting 分流指南

| 文件 | 用途 | 範例 |
|------|------|------|
| `docs/adr/` | **架構/演算法決策** | SAC→DDPG 轉換策略、87 維 Preprocessor 設計、Reward Annealing 方案 |
| `docs/troubleshooting.md` | **避雷指南** | 環境報錯、套件衝突、常見錯誤訊息解決方案 |

**判斷準則**：
- 如果這個問題涉及「為什麼這樣設計」→ ADR
- 如果這個問題是「遇到錯誤怎麼解決」→ Troubleshooting
