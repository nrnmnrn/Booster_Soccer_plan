# Architecture Decision Records (ADR)

本目錄記錄專案中的重大架構與演算法決策。

---

## ADR 索引

| 編號 | 標題 | 狀態 | 日期 |
|------|------|------|------|
| *尚無 ADR* | - | - | - |

---

## 什麼是 ADR？

ADR (Architecture Decision Record) 用於記錄：
- **架構決策**：系統結構、模組設計
- **演算法選擇**：SAC vs DDPG、Reward 設計
- **重大技術決策**：維度選擇、轉換策略

**不屬於 ADR 的內容**（應放在 `troubleshooting.md`）：
- 環境報錯解決
- 套件安裝問題
- 一般性 Bug 修復

---

## 如何建立新 ADR

1. 查看現有 ADR 的最大編號 N
2. 建立 `NNNN+1-kebab-case-title.md`
3. 使用模板：`.claude/templates/adr-template.md`
4. 完成後更新本索引

### 命名範例

```
0001-use-mjx-pretraining.md
0002-87dim-preprocessor-design.md
0003-sac-to-ddpg-conversion.md
```

---

## 模板位置

`.claude/templates/adr-template.md`
