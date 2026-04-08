# `experiments_v2_dog_only/` — 历史归档

第二轮 VisInject 实验的快照，**仅作历史保留，不要修改**。

## 概况

- **范围**：3 prompts × 3 model configs = 9 个实验
- **测试图**：仅 `ORIGIN_dog.png`
- **评估**：每个实验跑了 LLM-as-Judge（GPT-4o-mini），结果在 `judge_results.json`
- **时间**：2026-03-25 完成

## 与当前实验（v3）的区别

| 维度 | v2（本目录） | v3（`../experiments/`） |
|---|---|---|
| Prompt 数 | 3 (card / url / apple) | 7 (+email / news / ad / obey) |
| 测试图数 | 1 (dog only) | 7 |
| 实验数 | 9 | 21 |
| Response pairs | 9 | 147 |
| Judge 状态 | ✅ 已评估 | 待跑 |
| 目录结构 | 平坦：JSON 直接在 exp 根目录 | 分层：JSON 在 `results/` 子目录 |

v3 已完全取代 v2 的结论范围。v2 保留是为了：
- 复现性：早期实验结果可追溯
- 结构对照：v3 的 schema 与 v2 几乎一致，迁移历史教训

## 不要做的事

- 不要在这里加新文件
- 不要重新跑 judge 覆盖已有结果
- 不要重命名子目录

要做新实验，往 [`../experiments/`](../experiments/) 里加。
