"""
PersonaSteer 完整模型

整合 HyperNetwork、SteeringInjection 和骨干模型，
实现基于人格画像的对话生成。
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import torch.nn.functional as F

from .hyper_network import HyperNetwork
from .injection import SteeringInjection
from .components import freeze_module, verify_frozen, count_parameters


@dataclass
class PersonaSteerConfig:
    """PersonaSteer 模型配置

    Attributes:
        v_dim: 干预向量维度
        hidden_dim: MLP 隐藏层维度
        num_hyper_layers: 超网络 MLP 层数
        inject_layers: 注入层索引列表
        layer_dim: 骨干模型隐藏层维度
        gate_hidden_dim: 门控 MLP 隐藏层维度
        dropout: Dropout 概率
        backbone_model_name: 骨干模型名称
        encoder_model_name: 编码器模型名称
        use_layer_embedding: 是否使用层嵌入
    """
    v_dim: int = 1024
    hidden_dim: int = 4096
    num_hyper_layers: int = 3
    inject_layers: List[int] = None
    layer_dim: int = 2048  # Qwen2.5-3B hidden_size
    gate_hidden_dim: int = 256
    dropout: float = 0.1
    backbone_model_name: str = "Qwen/Qwen2.5-3B"
    encoder_model_name: str = "Qwen/Qwen2.5-3B"
    vocab_size: int = 151936
    use_layer_embedding: bool = True  # 是否使用层嵌入

    def __post_init__(self):
        if self.inject_layers is None:
            self.inject_layers = [14, 15, 16, 17, 18, 19, 20, 21]


class PersonaSteerModel(nn.Module):
    """PersonaSteer 模型

    整合骨干模型、编码器、超网络和注入模块的完整模型。

    Args:
        config: 模型配置
        backbone: 骨干模型 (Qwen3-4B)
        encoder: 文本编码器 (Qwen3-Embedding)
    """

    def __init__(
        self,
        config: PersonaSteerConfig,
        backbone: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.v_dim = config.v_dim  # 便于外部访问

        # 骨干模型
        if backbone is not None:
            self.backbone = backbone
            # 获取隐藏层维度
            if hasattr(backbone, 'config'):
                self.layer_dim = backbone.config.hidden_size
            else:
                self.layer_dim = config.layer_dim
        else:
            self.backbone = None
            self.layer_dim = config.layer_dim

        # 编码器
        if encoder is not None:
            self.encoder = encoder
            freeze_module(self.encoder)
        else:
            self.encoder = None

        # 超网络
        if self.encoder is not None:
            # 获取encoder输出维度
            if hasattr(encoder, 'config'):
                encoder_dim = encoder.config.hidden_size
            else:
                encoder_dim = self.layer_dim  # 默认使用layer_dim

            self.hyper_network = HyperNetwork(
                encoder=self.encoder,
                v_dim=config.v_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_hyper_layers,
                encoder_dim=encoder_dim,
                num_inject_layers=len(config.inject_layers),  # 传入实际注入层数
                use_layer_embedding=config.use_layer_embedding if hasattr(config, 'use_layer_embedding') else True,
            )
        else:
            self.hyper_network = None

        # 注入模块
        self.injection = SteeringInjection(
            inject_layers=config.inject_layers,
            v_dim=config.v_dim,
            layer_dim=self.layer_dim,
            gate_hidden_dim=config.gate_hidden_dim,
            dropout=config.dropout,
        )

        # Hook 句柄列表
        self.hooks: List[Any] = []
        self.current_v_t: Optional[torch.Tensor] = None

        # 【V4修复】Baseline模式：禁用注入，让模型走纯backbone
        self.baseline_mode = False

    def _register_injection_hooks(self) -> None:
        """注册注入 hooks 到骨干模型"""
        if self.backbone is None:
            return

        # 清除现有 hooks
        self._clear_hooks()

        # 为每个注入层注册 forward hook
        for idx, layer_idx in enumerate(self.config.inject_layers):
            # 获取对应层
            if hasattr(self.backbone, 'model'):
                if hasattr(self.backbone.model, 'layers'):
                    layer = self.backbone.model.layers[layer_idx]
                else:
                    layer = self.backbone.model.h[idx]
            else:
                layer = self.backbone.layers[layer_idx]

            # 注册 hook
            hook = layer.register_forward_hook(
                self._create_injection_hook(idx)
            )
            self.hooks.append(hook)

    def _create_injection_hook(self, layer_idx: int):
        """创建注入 hook

        Args:
            layer_idx: 注入层索引

        Returns:
            Hook 函数
        """
        def hook(module, input, output):
            # 处理不同格式的输出
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # 注入干预
            injected = self.injection.inject(hidden_states, layer_idx)

            # 返回修改后的输出
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected

        return hook

    def _clear_hooks(self) -> None:
        """清除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_backbone(self, backbone: nn.Module) -> None:
        """设置骨干模型并注册 hooks

        Args:
            backbone: 骨干模型
        """
        self.backbone = backbone
        freeze_module(self.backbone)

        # 获取隐藏层维度
        if hasattr(backbone, 'config'):
            self.layer_dim = backbone.config.hidden_size

        # 重新创建注入模块
        self.injection = SteeringInjection(
            inject_layers=self.config.inject_layers,
            v_dim=self.config.v_dim,
            layer_dim=self.layer_dim,
            gate_hidden_dim=self.config.gate_hidden_dim,
            dropout=self.config.dropout,
        )

        # 注册 hooks
        self._register_injection_hooks()

    def set_encoder(self, encoder: nn.Module) -> None:
        """设置编码器

        Args:
            encoder: 文本编码器
        """
        self.encoder = encoder
        freeze_module(self.encoder)

        # 获取 encoder 输出维度
        if hasattr(encoder, 'config'):
            encoder_dim = encoder.config.hidden_size
        else:
            encoder_dim = self.layer_dim

        # 重新创建超网络
        self.hyper_network = HyperNetwork(
            encoder=self.encoder,
            v_dim=self.config.v_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_hyper_layers,
            encoder_dim=encoder_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        v_prev: torch.Tensor,
        personality_texts: List[str],
        user_query_texts: List[str],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 【方案A】Query-Aware + 【方案D】多层向量

        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            v_prev: 上一轮干预向量 (batch, v_dim)
            personality_texts: 人格描述文本列表
            user_query_texts: 用户query文本列表
            attention_mask: 注意力掩码 (batch, seq_len)

        Returns:
            logits: 预测 logits (batch, seq_len, vocab_size)
            v_t_layers: 当前干预向量 (batch, num_layers, v_dim) 【方案D】多层输出
            v_norm: 干预向量norm (batch,) 用于正则化
        """
        # Step 1: 生成干预向量
        if self.hyper_network is not None:
            # 临时禁用 injection hooks（encoder 和 backbone 共享权重时，encoder forward 不应触发注入）
            self.injection.injection_enabled = False
            v_t_layers, z_t, v_norm = self.hyper_network(personality_texts, user_query_texts, v_prev)  # 【方案A】传入两个文本
            self.injection.injection_enabled = True
        else:
            # 如果没有超网络，使用零向量
            v_t_layers = torch.zeros(
                input_ids.size(0),
                8,  # 默认8层
                self.config.v_dim,
                device=input_ids.device,
            )
            v_norm = torch.zeros(input_ids.size(0), device=input_ids.device)

        self.current_v_t = v_t_layers  # 【方案D】存储多层向量

        # Step 2: 设置注入向量（支持多层）
        if self.baseline_mode:
            # 【V4修复】Baseline模式：将gate设为0，完全禁用注入
            self.injection.current_gate_values = torch.zeros(
                v_t_layers.size(0), self.injection.num_inject_layers,
                device=v_t_layers.device
            )
            self.injection.current_v_t = v_t_layers  # 保留v_t用于记录
        else:
            self.injection.set_intervention_vector(v_t_layers)

        # Step 3: 骨干模型前向传播 (触发 hooks)
        if self.backbone is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
        else:
            # 如果没有骨干模型,返回占位 logits
            # 使用默认词表大小，实际使用时会从 config 获取
            logits = torch.randn(
                input_ids.size(0),
                input_ids.size(1),
                151936,  # Qwen3-4B vocab size
                device=input_ids.device,
            )

        return logits, v_t_layers, v_norm  # 【方案D】返回多层向量

    def generate(
        self,
        input_ids: torch.Tensor,
        v_prev: torch.Tensor,
        personality_texts: List[str],
        user_query_texts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成文本 - 【方案A】Query-Aware

        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            v_prev: 上一轮干预向量 (batch, v_dim)
            personality_texts: 人格描述文本列表
            user_query_texts: 用户query文本列表
            max_new_tokens: 最大新 token 数量
            temperature: 采样温度
            top_p: nucleus 采样概率

        Returns:
            generated_ids: 生成的 token IDs (batch, seq_len + max_new_tokens)
            v_t: 最终干预向量 (batch, v_dim)
        """
        self.eval()

        # baseline_mode: 直接调用 backbone.generate，不注入，使用 KV cache
        if self.baseline_mode and self.backbone is not None:
            with torch.no_grad():
                out = self.backbone.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    pad_token_id=self.backbone.config.eos_token_id,
                )
            dummy_v = torch.zeros(input_ids.size(0), self.v_dim, device=input_ids.device)
            return out, dummy_v

        generated = input_ids.clone()

        # 【修复】在token生成循环开始前计算一次v_t，整轮固定不变
        # 原bug：每个token步骤都重新调用forward+更新v_prev，导致干预向量发散
        with torch.no_grad():
            _, v_t_layers_fixed, _ = self.forward(
                input_ids=generated,
                v_prev=v_prev,
                personality_texts=personality_texts,
                user_query_texts=user_query_texts,
            )
            self.injection.set_intervention_vector(v_t_layers_fixed)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 骨干模型前向传播（hooks已设置好固定的v_t）
                if self.backbone is not None:
                    outputs = self.backbone(
                        input_ids=generated,
                        attention_mask=None,
                        use_cache=False,
                    )
                    logits = outputs.logits
                else:
                    logits = torch.randn(
                        generated.size(0), generated.size(1), 151936,
                        device=generated.device,
                    )

                # 获取下一个 token 的 logits
                next_token_logits = logits[:, -1, :] / temperature

                # Top-p 采样
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # 保留概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                # 遇到 EOS 或 im_end 停止（支持Qwen chat template结束符）
                # Qwen2.5: eos=151643(<|endoftext|>), im_end=151645
                # Qwen3: eos=151645(<|im_end|>)
                stop_ids = {151643, 151645}
                if self.backbone is not None and hasattr(self.backbone.config, 'eos_token_id'):
                    eos = self.backbone.config.eos_token_id
                    if isinstance(eos, list):
                        stop_ids.update(eos)
                    else:
                        stop_ids.add(eos)
                if next_token.item() in stop_ids:
                    break

        return generated, v_t_layers_fixed.mean(dim=1)  # 返回多层向量的平均

    def get_trainable_parameters(self) -> int:
        """获取可训练参数量

        Returns:
            可训练参数数量
        """
        return count_parameters(self)

    def get_frozen_parameters(self) -> int:
        """获取冻结参数量

        Returns:
            冻结参数数量
        """
        return sum(
            p.numel() for p in self.parameters()
            if not p.requires_grad
        )

    def verify_frozen_backbone(self) -> bool:
        """验证骨干模型是否冻结

        Returns:
            如果冻结返回 True
        """
        if self.backbone is None:
            return True
        return verify_frozen(self.backbone)

    def verify_frozen_encoder(self) -> bool:
        """验证编码器是否冻结

        Returns:
            如果冻结返回 True
        """
        if self.encoder is None:
            return True
        return verify_frozen(self.encoder)

    def get_injection_info(self) -> Dict[str, Any]:
        """获取注入层信息

        Returns:
            注入层配置信息
        """
        return {
            "inject_layers": self.config.inject_layers,
            "num_inject_layers": len(self.config.inject_layers),
            "v_dim": self.config.v_dim,
            "layer_dim": self.layer_dim,
        }


def create_personasteer_model(
    config: PersonaSteerConfig,
    load_models: bool = False,
) -> PersonaSteerModel:
    """创建 PersonaSteer 模型

    Args:
        config: 模型配置
        load_models: 是否加载实际的模型权重

    Returns:
        PersonaSteerModel 实例
    """
    model = PersonaSteerModel(config)

    if load_models:
        from transformers import AutoModel, AutoTokenizer

        # 加载骨干模型
        backbone = AutoModel.from_pretrained(
            config.backbone_model_name,
            torch_dtype=torch.float16,
        )
        model.set_backbone(backbone)

        # 加载编码器
        encoder = AutoModel.from_pretrained(config.encoder_model_name)
        model.set_encoder(encoder)

    return model
