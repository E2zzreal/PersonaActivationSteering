"""
自动指标评估器
计算 Loss, PPL, v_t 稳定性, gate 分布等指标
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import compute_sft_loss

logger = logging.getLogger(__name__)


class AutoMetricsEvaluator:
    """
    自动指标评估器

    在评估集上计算以下指标:
    - Loss: SFT 损失值
    - PPL: 困惑度 (Perplexity)
    - v_variance: 干预向量的方差 (稳定性)
    - gate_distribution: 门控值的分布统计

    Args:
        device: 计算设备
    """

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def evaluate(
        self,
        model: nn.Module,
        eval_loader: DataLoader,
    ) -> dict[str, Any]:
        """
        在评估集上计算所有指标

        Args:
            model: PersonaSteer 模型
            eval_loader: 评估数据加载器

        Returns:
            dict: 包含各项指标的字典
        """
        model.eval()
        model.to(self.device)

        losses = []
        clean_losses = []
        ppls = []
        v_variances = []
        gate_values_list = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # 将数据移动到设备
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                user_ids = batch["user_ids"]
                personalities = batch["personalities"]  # 【方案A】新增

                batch_size = input_ids.size(1)

                # 多轮对话循环
                v_prev = torch.zeros(batch_size, model.v_dim).to(self.device)

                for turn_idx in range(input_ids.size(0)):
                    turn_input_ids = input_ids[turn_idx]
                    turn_labels = labels[turn_idx]
                    turn_mask = attention_mask[turn_idx]
                    turn_user_texts = [batch["user_texts"][b][turn_idx] for b in range(batch_size)]

                    # 过滤有效样本
                    valid_mask = torch.tensor(
                        [batch["num_turns"][b] > turn_idx for b in range(batch_size)],
                        device=self.device,
                    )

                    if not valid_mask.any():
                        continue

                    # 只对有效行做 forward
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    valid_input_ids = turn_input_ids[valid_idx]
                    valid_labels = turn_labels[valid_idx]
                    valid_user_texts = [batch["user_texts"][b][turn_idx] for b in valid_idx.tolist()]
                    valid_personalities = [personalities[i] for i in valid_idx.tolist()]  # 【方案A】新增
                    valid_v_prev = v_prev[valid_idx]

                    # Forward (injected)
                    logits, v_t_layers, v_norm = model(
                        input_ids=valid_input_ids,
                        v_prev=valid_v_prev,
                        personality_texts=valid_personalities,
                        user_query_texts=valid_user_texts,
                    )

                    # 计算注入后的损失
                    loss = compute_sft_loss(logits, valid_labels)
                    losses.append(loss.item())

                    # 【V4修复】计算clean loss（无注入）
                    if hasattr(model, 'baseline_mode') and not model.baseline_mode:
                        saved_gate = model.injection.current_gate_values
                        model.injection.current_gate_values = torch.zeros(
                            saved_gate.size(0), model.injection.num_inject_layers,
                            device=saved_gate.device
                        )
                        logits_clean, _, _ = model(
                            input_ids=valid_input_ids,
                            v_prev=valid_v_prev,
                            personality_texts=valid_personalities,
                            user_query_texts=valid_user_texts,
                        )
                        loss_clean = compute_sft_loss(logits_clean, valid_labels)
                        model.injection.current_gate_values = saved_gate
                    else:
                        loss_clean = loss
                    clean_losses.append(loss_clean.item())

                    # 计算 PPL
                    ppl = torch.exp(loss)
                    ppls.append(ppl.item())

                    # 计算干预向量方差：样本间差异（batch 维度）
                    if v_t_layers.dim() == 3:
                        v_mean = v_t_layers.mean(dim=1)  # (valid_batch, v_dim)
                    else:
                        v_mean = v_t_layers

                    # 修复：只有多个样本时才计算方差
                    if v_mean.size(0) > 1:
                        v_var = v_mean.var(dim=0).mean().item()  # 样本间方差
                        v_variances.append(v_var)

                    # 记录 gate 值
                    if hasattr(model, "injection") and hasattr(model.injection, "gate"):
                        if v_t_layers.dim() == 3:
                            gate_out = model.injection.gate(v_t_layers.mean(dim=1))
                        else:
                            gate_out = model.injection.gate(v_t_layers)
                        gate_values_list.append(gate_out.cpu().numpy())

                    # 只更新有效行的 v_prev，无效行保持不变
                    if v_t_layers.dim() == 3:
                        v_t_flat = v_t_layers.mean(dim=1).detach()
                    else:
                        v_t_flat = v_t_layers.detach()
                    v_prev = v_prev.clone()
                    v_prev[valid_idx] = v_t_flat

        # 汇总结果
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_clean_loss = float(np.mean(clean_losses)) if clean_losses else 0.0
        results = {
            "loss_sft": avg_loss,
            "loss_clean": avg_clean_loss,
            "loss_std": float(np.std(losses)) if losses else 0.0,
            "ppl": float(np.mean(ppls)) if ppls else 0.0,
            "ppl_std": float(np.std(ppls)) if ppls else 0.0,
            "v_variance": float(np.mean(v_variances)) if len(v_variances) > 0 else 0.0,
            "num_samples": len(losses),
        }

        # 【V4修复】注入增益：注入后loss相对clean loss的改善百分比
        if avg_clean_loss > 0 and not (hasattr(model, 'baseline_mode') and model.baseline_mode):
            injection_gain = avg_clean_loss - avg_loss
            injection_gain_pct = (injection_gain / avg_clean_loss) * 100
            results["injection_gain"] = float(injection_gain)
            results["injection_gain_pct"] = float(injection_gain_pct)

        # Gate 分布统计
        if gate_values_list:
            gate_values = np.concatenate(gate_values_list, axis=0)
            results["gate_distribution"] = {
                "mean": float(gate_values.mean(axis=0).mean()),
                "std": float(gate_values.std(axis=0).mean()),
                "min": float(gate_values.min()),
                "max": float(gate_values.max()),
            }

        logger.info(
            f"Evaluation results: loss={results['loss_sft']:.4f}, "
            f"ppl={results['ppl']:.2f}, v_var={results['v_variance']:.4f}"
        )

        return results


class MetricsTracker:
    """
    训练过程中的指标追踪器

    用于在训练过程中记录各项指标的变化趋势。
    """

    def __init__(self):
        self.history: dict[str, list] = {
            "train_loss": [],
            "eval_loss": [],
            "eval_ppl": [],
            "eval_v_variance": [],
            "learning_rate": [],
        }

    def update_train(self, loss: float):
        """更新训练损失"""
        self.history["train_loss"].append(loss)

    def update_eval(self, metrics: dict[str, Any]):
        """更新评估指标"""
        if "loss_sft" in metrics:
            self.history["eval_loss"].append(metrics["loss_sft"])
        if "ppl" in metrics:
            self.history["eval_ppl"].append(metrics["ppl"])
        if "v_variance" in metrics:
            self.history["eval_v_variance"].append(metrics["v_variance"])

    def update_lr(self, lr: float):
        """更新学习率"""
        self.history["learning_rate"].append(lr)

    def get_history(self) -> dict[str, list]:
        """获取历史记录"""
        return self.history

    def summary(self) -> dict[str, float]:
        """生成摘要统计"""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f"{key}_latest"] = values[-1]
                summary[f"{key}_best"] = min(values) if "loss" in key or "ppl" in key else max(values)
        return summary
