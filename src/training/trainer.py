"""
三阶段渐进训练器
支持混合精度训练、梯度裁剪、checkpoint 保存
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
try:
    from bitsandbytes.optim import AdamW8bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import PersonaSteerLoss, compute_sft_loss

logger = logging.getLogger(__name__)


class PersonaSteerTrainer:
    """
    PersonaSteer 三阶段渐进训练器

    支持三个训练阶段:
    - Stage 1: 仅训练 HyperNetwork，gate 冻结
    - Stage 2: 解冻 gate，联合训练
    - Stage 3: 加入对比学习损失

    Args:
        model: PersonaSteerModel 模型实例
        config: 训练配置字典
        train_loader: 训练数据加载器
        eval_loader: 评估数据加载器 (可选)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # 训练阶段配置
        self.stage = config.get("stage", 1)
        # 【修复】优先使用传入的device参数
        device_str = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        # 优化器配置
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # 损失配置
        self.sft_weight = config.get("sft_weight", 1.0)
        self.scl_weight = config.get("scl_weight", 0.1)
        self.temperature = config.get("temperature", 0.07)
        
        # 【新增】Gate约束配置
        self.gate_min_value = config.get("gate_min_value", 0.3)  # Gate最小激活值
        self.gate_reg_weight = config.get("gate_reg_weight", 0.1)  # Gate正则化权重（提高到0.1）
        self.gate_lr_multiplier = config.get("gate_lr_multiplier", 10.0)  # Gate学习率倍数
        
        # 【新增方案B】v_norm正则化配置
        self.v_norm_weight = config.get("v_norm_weight", 0.1)   # 增强: 0.01 → 0.1
        self.v_norm_target = config.get("v_norm_target", 5.0)   # 降低: 32.0 → 5.0

        # 【V4修复】双loss训练配置
        self.use_dual_loss = config.get("use_dual_loss", True)
        self.injected_loss_weight = config.get("injected_loss_weight", 0.5)

        # 训练轮次
        self.num_epochs = config.get("num_epochs", 3)
        self.current_epoch = 0
        self.global_step = 0

        # 混合精度
        self.use_amp = config.get("use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None

        # 损失函数
        self.loss_fn = PersonaSteerLoss(
            sft_weight=self.sft_weight,
            scl_weight=self.scl_weight,
            temperature=self.temperature,
        )

        # 优化器和学习率调度器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Checkpoint 配置
        self.output_dir = Path(config.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config.get("save_interval", 1)

        # 指标记录
        self.train_history = []
        self.best_loss = float("inf")

        # 阶段特定配置
        self._configure_stage()

    def _setup_optimizer(self) -> AdamW:
        """配置优化器"""
        # 阶段1: 仅训练 HyperNetwork
        # 阶段2: 训练 HyperNetwork + Gate
        # 阶段3: 全部训练

        if self.stage == 1:
            # 仅训练 HyperNetwork（排除冻结的 encoder）
            if self.model.hyper_network is None:
                raise ValueError("hyper_network is None. Please ensure encoder is loaded when creating the model.")
            params = [p for p in self.model.hyper_network.parameters() if p.requires_grad]
            logger.info("Stage 1: Training HyperNetwork only (gate frozen)")
        elif self.stage == 2:
            # 训练 HyperNetwork + Gate（排除冻结的 encoder）
            if self.model.hyper_network is None:
                raise ValueError("hyper_network is None. Please ensure encoder is loaded when creating the model.")
            params = [p for p in self.model.hyper_network.parameters() if p.requires_grad]
            params += [p for p in self.model.injection.gate.parameters() if p.requires_grad]
            logger.info("Stage 2: Training HyperNetwork + Gate")
        else:
            # 全部训练（HyperNetwork + Gate）
            params = [p for p in self.model.hyper_network.parameters() if p.requires_grad]
            if hasattr(self.model, 'injection') and hasattr(self.model.injection, 'gate'):
                params += [p for p in self.model.injection.gate.parameters() if p.requires_grad]
            logger.info("Stage 3: Training HyperNetwork + Gate (with SCL loss)")

        # 8-bit AdamW causes OOM during state initialization on shared-weight models
        trainable_count = sum(p.numel() for p in params)
        logger.info(f"Using standard AdamW optimizer (stage {self.stage}), trainable params: {trainable_count:,}")
        return AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _setup_scheduler(self) -> CosineAnnealingLR:
        """配置学习率调度器"""
        total_steps = len(self.train_loader) * self.num_epochs
        return CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.learning_rate * 0.1,
        )

    def _configure_stage(self):
        """根据阶段配置模型"""
        if self.model.hyper_network is None:
            raise ValueError("hyper_network is None. Cannot configure training stage.")

        if self.stage == 1:
            # 冻结 gate
            for param in self.model.injection.gate.parameters():
                param.requires_grad = False
        else:
            # 解冻 gate
            for param in self.model.injection.gate.parameters():
                param.requires_grad = True

        # 确保 hyper_network 可训练（排除共享的 encoder）
        # 【修复】encoder 与 backbone 共享权重，不应被解冻
        for name, param in self.model.hyper_network.named_parameters():
            if not name.startswith('encoder.'):
                param.requires_grad = True

    def _compute_gate_regularization(self) -> torch.Tensor:
        """
        计算Gate约束正则化损失

        基于Gate的实际输出值（而非参数值）进行约束：
        1. 方差鼓励：惩罚Gate输出方差过小（防止所有层门控值一样）
        2. 最小激活约束：防止Gate学到关闭注入的策略

        Returns:
            Gate正则化损失
        """
        # 使用Gate最近一次前向传播的实际输出
        gate_output = self.model.injection.current_gate_values
        if gate_output is None:
            return torch.tensor(0.0, device=self.device)

        # gate_output shape: (batch, num_layers), 值域 [0, 1]（经过Sigmoid）

        # 1. 方差鼓励：惩罚跨层方差过小
        layer_var = gate_output.var(dim=-1).mean()  # 每个样本跨层方差的平均
        variance_penalty = torch.relu(0.01 - layer_var)  # 方差至少 0.01

        # 2. 最小激活约束：防止Gate输出过低
        min_penalty = torch.relu(self.gate_min_value - gate_output).mean()

        return variance_penalty + min_penalty

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            dict: 平均损失指标
        """
        self.model.train()
        # 【修复】模型已在create_model时移动到正确设备，此处无需再移动

        total_loss = 0.0
        total_sft_loss = 0.0
        total_scl_loss = 0.0
        total_gate_entropy = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        amp_autocast = torch.amp.autocast('cuda') if self.use_amp else None

        for batch in pbar:
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(self.device)  # (num_turns, batch, seq_len)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            user_ids = batch["user_ids"]
            personalities = batch["personalities"]  # 【新增】
            num_turns_list = batch["num_turns"]

            batch_size = input_ids.size(1)

            # 多轮对话循环
            v_prev = torch.zeros(batch_size, self.model.v_dim).to(self.device)

            epoch_loss = 0.0
            epoch_sft_loss = 0.0
            epoch_scl_loss = 0.0
            epoch_gate_entropy = 0.0
            num_turns_processed = 0

            for turn_idx in range(input_ids.size(0)):
                turn_input_ids = input_ids[turn_idx]
                turn_labels = labels[turn_idx]
                turn_mask = attention_mask[turn_idx]
                turn_user_texts = [batch["user_texts"][b][turn_idx] for b in range(batch_size)]

                # 过滤有效样本 (该轮次存在的样本)
                valid_mask = torch.tensor(
                    [num_turns_list[b] > turn_idx for b in range(batch_size)],
                    device=self.device,
                )

                if not valid_mask.any():
                    continue

                # 只对有效行（本轮有真实数据的样本）做 forward
                valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                valid_input_ids = turn_input_ids[valid_idx]
                valid_labels = turn_labels[valid_idx]
                valid_user_texts = [turn_user_texts[i] for i in valid_idx.tolist()]
                valid_personalities = [personalities[i] for i in valid_idx.tolist()]
                valid_user_ids = [user_ids[i] for i in valid_idx.tolist()]
                valid_v_prev = v_prev[valid_idx]

                # Forward (injected)
                with amp_autocast if amp_autocast else torch.enable_grad():
                    logits, v_t, v_norm = self.model(
                        input_ids=valid_input_ids,
                        v_prev=valid_v_prev,
                        personality_texts=valid_personalities,
                        user_query_texts=valid_user_texts,
                    )

                # 计算注入后的损失
                loss_sft = compute_sft_loss(logits, valid_labels)

                # 【V4修复】双loss训练：计算clean loss（无注入）作为基础
                if self.use_dual_loss:
                    # 保存当前gate值
                    saved_gate = self.model.injection.current_gate_values
                    # 禁用注入
                    self.model.injection.current_gate_values = torch.zeros(
                        saved_gate.size(0), self.model.injection.num_inject_layers,
                        device=saved_gate.device
                    )
                    with torch.no_grad():
                        logits_clean, _, _ = self.model(
                            input_ids=valid_input_ids,
                            v_prev=valid_v_prev.detach(),
                            personality_texts=valid_personalities,
                            user_query_texts=valid_user_texts,
                        )
                    loss_clean = compute_sft_loss(logits_clean, valid_labels).detach()
                    # 恢复gate
                    self.model.injection.current_gate_values = saved_gate
                else:
                    loss_clean = torch.tensor(0.0, device=self.device)

                # 只在 Stage 3 计算对比损失（SCL），避免不必要的 GPU 内存开销
                if self.stage >= 3:
                    loss_scl = self.loss_fn.scl(v_t, valid_personalities, user_ids=valid_user_ids)
                else:
                    loss_scl = torch.tensor(0.0, device=self.device)

                # 组合损失
                # dual loss: 训练注入后生成不差于clean基线
                # loss = loss_injected + alpha * relu(loss_injected - loss_clean)
                # 效果: 最小化注入loss, 同时惩罚注入导致的质量下降
                if self.use_dual_loss:
                    alpha = self.injected_loss_weight
                    degradation_penalty = torch.relu(loss_sft - loss_clean)
                    loss = loss_sft + alpha * degradation_penalty
                else:
                    loss = self.sft_weight * loss_sft
                if self.stage >= 3:
                    loss += self.scl_weight * loss_scl
                    epoch_scl_loss += loss_scl.item()

                # Gate约束正则化（Stage 2及以上）
                if self.stage >= 2:
                    loss_gate_reg = self._compute_gate_regularization()
                    loss += self.gate_reg_weight * loss_gate_reg

                # v_norm正则化（soft constraint）
                # 惩罚v_norm偏离目标值，避免hard clip破坏方向信息
                loss_v_norm = ((v_norm - self.v_norm_target) ** 2).mean()
                loss += self.v_norm_weight * loss_v_norm

                # Gate entropy loss（防止 gate 集中在单层，鼓励各层均匀使用）
                gate_entropy_weight = self.config.get("gate_entropy_weight", 0.01)
                if gate_entropy_weight > 0 and hasattr(self.model, 'injection'):
                    loss_gate_entropy = self.model.injection.compute_gate_entropy_loss(v_t)
                    loss += gate_entropy_weight * loss_gate_entropy
                    epoch_gate_entropy += loss_gate_entropy.item()
                else:
                    loss_gate_entropy = torch.tensor(0.0)

                epoch_sft_loss += loss_sft.item()
                if self.use_dual_loss:
                    epoch_sft_loss += loss_clean.item()  # 记录clean loss用于监控
                epoch_loss += loss.item()
                num_turns_processed += 1

                # 调试信息
                if self.use_dual_loss:
                    logger.debug(f"Turn {turn_idx}: loss={loss.item():.4f}, loss_clean={loss_clean.item():.4f}, loss_injected={loss_sft.item():.4f}")
                else:
                    logger.debug(f"Turn {turn_idx}: loss={loss.item():.4f}, loss.requires_grad={loss.requires_grad}")

                # 反向传播
                if loss.requires_grad:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    logger.warning(f"Loss does not require grad! Skipping backward")

                # 更新有效行的干预向量，无效行保持原 v_prev（阻断跨轮梯度）
                if v_t.dim() == 3:
                    v_t_flat = v_t.mean(dim=1).detach()  # (valid_batch, v_dim)
                else:
                    v_t_flat = v_t.detach()
                v_prev = v_prev.clone()
                v_prev[valid_idx] = v_t_flat

            if num_turns_processed > 0:
                # 优化器更新（支持 AMP）
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                # 记录指标
                total_loss += epoch_loss / num_turns_processed
                total_sft_loss += epoch_sft_loss / num_turns_processed
                total_scl_loss += epoch_scl_loss / max(num_turns_processed, 1)
                total_gate_entropy += epoch_gate_entropy / max(num_turns_processed, 1)
                num_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{epoch_loss / num_turns_processed:.4f}",
                    "sft": f"{epoch_sft_loss / num_turns_processed:.4f}",
                })

                self.global_step += 1

        avg_metrics = {
            "loss": total_loss / max(num_batches, 1),
            "sft_loss": total_sft_loss / max(num_batches, 1),
            "scl_loss": total_scl_loss / max(num_batches, 1),
            "gate_entropy": total_gate_entropy / max(num_batches, 1),
        }

        return avg_metrics

    def train(self) -> dict[str, list]:
        """
        执行完整训练流程

        Returns:
            dict: 训练历史记录
        """
        logger.info(f"Starting training for stage {self.stage}")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # 训练
            metrics = self.train_epoch(epoch)
            self.train_history.append(metrics)

            logger.info(
                f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}, "
                f"sft={metrics['sft_loss']:.4f}, scl={metrics['scl_loss']:.4f}, "
                f"gate_entropy={metrics['gate_entropy']:.4f}"
            )

            # 保存 checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            # 保存最佳模型
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                self.save_checkpoint("best.pt")
                logger.info(f"New best model saved (loss={self.best_loss:.4f})")

        return {"train_history": self.train_history}

    def save_checkpoint(self, filename: str):
        """保存模型 checkpoint"""
        checkpoint_path = self.output_dir / filename

        # 收集可训练参数
        trainable_params = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": trainable_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "stage": self.stage,
            "config": self.config,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """加载 checkpoint"""
        import pickle
        
        try:
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=self.device,
                pickle_module=pickle
            )
        except Exception as e:
            logger.warning(f"Standard load failed: {e}, trying alternative method...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        if "model_state_dict" in checkpoint:
            # 只加载匹配的键
            model_state = checkpoint["model_state_dict"]
            current_state = self.model.state_dict()
            
            # 过滤匹配的键
            matched_state = {
                k: v for k, v in model_state.items() 
                if k in current_state and v.shape == current_state[k].shape
            }
            
            # 加载匹配的状态
            self.model.load_state_dict(matched_state, strict=False)
            
            logger.info(f"Loaded {len(matched_state)}/{len(model_state)} parameters from checkpoint")
            
            # 加载其他状态（忽略优化器状态，因为Stage间参数不同）
            if "current_epoch" in checkpoint:
                self.current_epoch = checkpoint["current_epoch"]
            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
            if "best_loss" in checkpoint:
                # Only restore best_loss if same stage (avoid cross-stage best_loss leak)
                ckpt_stage = checkpoint.get("stage", 1)
                if ckpt_stage == self.stage:
                    self.best_loss = checkpoint["best_loss"]
                else:
                    logger.info(f"Checkpoint stage ({ckpt_stage}) != current stage ({self.stage}), resetting best_loss to inf")
                    self.best_loss = float("inf")
                
            logger.info(f"Checkpoint loaded: epoch={self.current_epoch}, step={self.global_step}, best_loss={self.best_loss:.4f}")
            
            # 【注意】Stage间参数不同，不加载优化器和调度器状态
            logger.info("Skipped optimizer/scheduler state loading (different parameters between stages)")
        else:
            raise ValueError("Invalid checkpoint format")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ) -> "PersonaSteerTrainer":
        """从 checkpoint 创建训练器"""
        trainer = cls(model, config, train_loader, eval_loader)
        trainer.load_checkpoint(checkpoint_path)
        return trainer
