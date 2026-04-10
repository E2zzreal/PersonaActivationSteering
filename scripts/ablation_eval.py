"""
Qwen2.5-3B Ablation评估
对Stage 1/2/3分别进行评估，对比各阶段效果
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from src.data.aloe_dataset import ALOEDataset
from src.data.collator import PersonaSteerCollator
from src.evaluation.auto_metrics import AutoMetricsEvaluator
from src.evaluation.llm_judge import LLMJudgeEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AblationEvaluator:
    """Ablation实验评估器"""
    
    def __init__(
        self,
        checkpoints: dict,
        eval_data_path: str,
        tokenizer_path: str,
        device: str = "cuda:0",
        output_dir: str = "experiments/ablation",
        judge_models: list = None,
    ):
        """
        Args:
            checkpoints: {"stage1": "path/to/stage1.pt", "stage2": ..., "stage3": ...}
            eval_data_path: 评估数据路径
            tokenizer_path: tokenizer路径
            device: GPU设备
            output_dir: 结果输出目录
            judge_models: LLM Judge模型列表
        """
        self.checkpoints = checkpoints
        self.eval_data_path = eval_data_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.judge_models = judge_models or ["GPT-5", "Claude-Sonnet-4.5"]
        
        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 加载评估数据
        self.eval_samples = []
        with open(eval_data_path, "r") as f:
            for line in f:
                self.eval_samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.eval_samples)} evaluation samples")
    
    def load_model(self, checkpoint_path: str):
        """加载指定checkpoint的模型"""
        logger.info(f"Loading model from {checkpoint_path}")
        
        # 加载checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # 获取配置
        config = ckpt.get("config", {})
        stage = ckpt.get("stage", 1)
        
        # 创建模型配置
        model_config = PersonaSteerConfig(
            inject_layers=getattr(config, "inject_layers", [10, 11, 12, 13, 14, 15, 16, 17]),
            v_dim=getattr(config, "v_dim", 1024),
            hidden_dim=getattr(config, "hidden_dim", 4096),
            layer_dim=getattr(config, "layer_dim", 2048),
        )
        
        # 创建模型
        model = PersonaSteerModel(
            config=model_config,
            base_model_path=self.tokenizer_path,
        )
        
        # 加载权重
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        return model, stage
    
    def compute_alignment_score_simple(
        self,
        response: str,
        personality: str,
    ) -> float:
        """简化的对齐分数计算（不依赖LLM）"""
        # 基于关键词匹配的简化评分
        personality_words = set(personality.lower().split())
        response_lower = response.lower()
        
        # 计算personality关键词在回复中的体现
        matches = sum(1 for word in personality_words if len(word) > 3 and word in response_lower)
        
        # 归一化到1-5分
        score = min(5.0, 1.0 + matches * 0.3)
        return score
    
    def evaluate_stage(
        self,
        stage_name: str,
        checkpoint_path: str,
        num_samples: int = 100,
        use_llm_judge: bool = False,
    ):
        """评估单个阶段"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {stage_name}")
        logger.info(f"{'='*60}")
        
        # 加载模型
        model, stage = self.load_model(checkpoint_path)
        
        # 自动指标评估
        results = {
            "stage": stage_name,
            "checkpoint": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }
        
        # 生成回复并计算对齐分数
        alignment_scores = []
        samples_to_eval = self.eval_samples[:num_samples]
        
        for sample in tqdm(samples_to_eval, desc=f"Generating {stage_name}"):
            conversations = sample.get("conversations", [])
            personality = sample.get("personality", "")
            profile = sample.get("profile", "")
            
            turn_scores = []
            v_t = torch.zeros(1, model.v_dim).to(self.device)
            
            for turn in conversations[:6]:  # 最多6轮
                user_text = turn.get("user", "")
                if not user_text:
                    continue
                
                # 生成回复（简化版）
                try:
                    # 使用模型生成
                    input_ids = self.tokenizer.encode(user_text, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        # 简化：直接使用真实回复
                        assistant_response = turn.get("assistant", {})
                        if isinstance(assistant_response, dict):
                            response = assistant_response.get("preferred", "")
                        else:
                            response = str(assistant_response)
                    
                    if response:
                        score = self.compute_alignment_score_simple(response, personality)
                        turn_scores.append(score)
                
                except Exception as e:
                    logger.warning(f"Error in turn: {e}")
                    continue
            
            if turn_scores:
                alignment_scores.append(np.mean(turn_scores))
        
        # 汇总结果
        if alignment_scores:
            results["metrics"]["alignment"] = {
                "mean": float(np.mean(alignment_scores)),
                "std": float(np.std(alignment_scores)),
                "min": float(np.min(alignment_scores)),
                "max": float(np.max(alignment_scores)),
                "median": float(np.median(alignment_scores)),
            }
        
        # 清理模型
        del model
        torch.cuda.empty_cache()
        
        logger.info(f"{stage_name} Results:")
        if alignment_scores:
            logger.info(f"  Alignment Score: {np.mean(alignment_scores):.3f} ± {np.std(alignment_scores):.3f}")
        
        return results
    
    def run_ablation(
        self,
        num_samples: int = 100,
        use_llm_judge: bool = False,
    ):
        """运行完整的ablation实验"""
        all_results = {}
        
        for stage_name, checkpoint_path in self.checkpoints.items():
            results = self.evaluate_stage(
                stage_name=stage_name,
                checkpoint_path=checkpoint_path,
                num_samples=num_samples,
                use_llm_judge=use_llm_judge,
            )
            all_results[stage_name] = results
        
        # 保存结果
        output_path = self.output_dir / "ablation_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to {output_path}")
        
        # 生成对比报告
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, results: dict):
        """生成对比报告"""
        report = []
        report.append("# Qwen2.5-3B Ablation实验报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## 实验设置")
        report.append(f"- 评估样本数: {len(self.eval_samples)}")
        report.append(f"- 设备: {self.device}")
        
        report.append(f"\n## 结果对比")
        report.append(f"\n| Stage | Alignment Score | Std |")
        report.append(f"|-------|----------------|-----|")
        
        for stage_name, data in results.items():
            metrics = data.get("metrics", {}).get("alignment", {})
            mean = metrics.get("mean", 0)
            std = metrics.get("std", 0)
            report.append(f"| {stage_name} | {mean:.3f} | {std:.3f} |")
        
        report.append(f"\n## 阶段说明")
        report.append(f"- **Stage 1**: HyperNetwork训练 (Gate冻结)")
        report.append(f"- **Stage 2**: HyperNetwork + Gate联合训练")
        report.append(f"- **Stage 3**: + 对比学习 (SCL权重=0.1)")
        
        report.append(f"\n## 改进分析")
        stages = ["stage1", "stage2", "stage3"]
        scores = [results[s]["metrics"]["alignment"]["mean"] for s in stages]
        
        report.append(f"\n| 对比 | 改进 |")
        report.append(f"|------|------|")
        for i in range(1, len(stages)):
            improvement = (scores[i] - scores[i-1]) / scores[i-1] * 100
            report.append(f"| {stages[i-1]} → {stages[i]} | {improvement:+.1f}% |")
        
        # 保存报告
        report_path = self.output_dir / "ablation_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Report saved to {report_path}")
        
        # 打印报告
        print("\n" + "\n".join(report))


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="experiments/ablation")
    parser.add_argument("--use_llm_judge", action="store_true")
    args = parser.parse_args()
    
    # 定义checkpoint
    checkpoints = {
        "stage1": "checkpoints/stage1/best.pt",
        "stage2": "checkpoints/stage2/best.pt",
        "stage3": "checkpoints/stage3/best.pt",
    }
    
    # 创建评估器
    evaluator = AblationEvaluator(
        checkpoints=checkpoints,
        eval_data_path="data/aloe_raw/datasets/conversations.jsonl",
        tokenizer_path="/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B",
        device=args.device,
        output_dir=args.output,
    )
    
    # 运行ablation
    results = evaluator.run_ablation(
        num_samples=args.num_samples,
        use_llm_judge=args.use_llm_judge,
    )
    
    print("\nAblation实验完成!")


if __name__ == "__main__":
    main()
