# dualtrack 重跑尝试留档

日期：2026-04-12
最后验证日期：2026-04-12
任务：使用新双轨评估链路重跑 baseline 和 stage3_v2

## 执行情况
已尝试运行：
1. `python scripts/v4_eval_qwen3.py --device cuda:0 --judge_model GPT-5.2 --output_dir results/eval_dualtrack_20260412_baseline --stages baseline`
2. `python scripts/v4_eval_qwen3.py --device cuda:0 --judge_model GPT-5.2 --output_dir results/eval_dualtrack_20260412_stage3 --stages stage3 --stage3_checkpoint checkpoints/stage3_qwen3_v2/best.pt`

## 结果
两次运行均未完成有效评估，原因一致：
- `No CUDA GPUs are available`

## 产物
- `results/eval_dualtrack_20260412_baseline/`
- `results/eval_dualtrack_20260412_stage3/`

上述目录已生成，但其中仅包含失败情况下的空评估摘要，不可用于比较 raw / clean 差异。

## 结论
- 当前代码链路已具备 dualtrack 评估能力；
- 但当前执行环境没有可用 CUDA，因此无法在本轮会话内完成真实重跑；
- 后续需要在有 GPU 的环境中重新执行相同命令。

## 无迁移说明
无迁移，直接补充执行留档。
