# dualtrack v2 结果分析留档

日期：2026-04-12
最后验证日期：2026-04-12
任务：分析 baseline_v2 与 stage3_v2 的 dualtrack 评估结果

## 输入来源
- `results/eval_dualtrack_20260412_baseline_v2/baseline_responses.json`
- `results/eval_dualtrack_20260412_stage3_v2/stage3_responses.json`
- `src/evaluation/thinking_leak.py`

## 结论摘要
1. dualtrack 输出链路已生效，结果文件中已包含：
   - `raw_response`
   - `clean_response`
   - `leak_detected`
   - `leak_patterns`
   - `cleaning_applied`
   - `cleaning_skipped`
2. baseline_v2：
   - 总样本 90
   - raw 泄露 3
   - clean 泄露 3
   - `cleaning_applied=0`
   - 说明 baseline 本身泄露很少，当前保守规则几乎不触发有效清洗。
3. stage3_v2：
   - 总样本 90
   - raw 泄露 28
   - clean 泄露 24
   - `cleaning_applied=25`
   - `raw!=clean=25`
   - 说明保守规则已开始工作，但止损效果有限，仅将 28 条中的 4 条真正清理到不再命中泄露规则。
4. 当前问题不是 dualtrack 链路失效，而是**保守型前缀清理规则过弱**，清洗后仍残留大量推理痕迹。

## 判断
- 现阶段可以确认：注入模型的泄露问题确实显著高于 baseline。
- 也可以确认：仅靠当前第一版保守清洗，无法充分止损。
- 下一步应优先优化清洗规则，而不是急于进入训练数据预处理。

## 无迁移说明
无迁移，直接补充分析留档。
