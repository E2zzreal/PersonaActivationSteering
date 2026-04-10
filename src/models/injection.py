"""
动态门控注入模块

提供 DynamicGate 和 SteeringInjection 类，实现基于干预向量的隐藏状态注入。
使用层级门控机制控制每个注入层的干预强度。
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import torch.nn.functional as F


class InjectFunction(torch.autograd.Function):
    """自定义 autograd Function：断开注入梯度向 backbone 的传播。

    Forward: hidden_states + injected（结果一致）
    Backward: 梯度只回传给 injected（hypernetwork），不回传给 hidden_states（backbone）。
    这样 backbone 的中间 activations 可以被 gradient checkpointing 正常释放，
    避免冻结的 backbone 在 backward 时保留 ~7.7 GB 的无用 activations。
    """

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, injected: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(injected)
        return hidden_states + injected.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # hidden_states 的梯度 = None（backbone 冻结，不需要）
        # injected 的梯度 = grad_output（回传给 hypernetwork）
        return None, grad_output


class DynamicGate(nn.Module):
    """动态门控模块

    根据干预向量生成各注入层的门控值，控制干预强度。

    Args:
        v_dim: 干预向量维度
        num_layers: 注入层数量
        hidden_dim: 门控 MLP 隐藏层维度
    """

    def __init__(
        self,
        v_dim: int = 1024,
        num_layers: int = 8,
        hidden_dim: int = 256,
        gate_max: float = 1.0,
        gate_init_bias: float = -2.0,  # sigmoid(-2) ≈ 0.12，更合理的起点
    ):
        super().__init__()
        self.v_dim = v_dim
        self.num_layers = num_layers
        self.gate_max = gate_max

        self.gate_mlp = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_layers),
            nn.Sigmoid()  # 输出 0-1 之间的门控值
        )

        # 初始化最后一层Linear偏置，控制初始gate强度
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear) and module.out_features == self.num_layers:
                nn.init.constant_(module.bias, gate_init_bias)
                break

    def forward(self, v_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 gate 值和 entropy loss

        Args:
            v_t: (batch, v_dim) 或 (batch, num_layers, v_dim)

        Returns:
            gate_values: (batch, num_layers)，已 clamp 到 [0, gate_max]
            entropy_loss: 标量，gate 分布负熵（最小化 = 最大化均匀性）
        """
        if v_t.dim() == 3:
            v_flat = v_t.mean(dim=1).float()
        else:
            v_flat = v_t.float()

        gate_values = self.gate_mlp(v_flat)  # (batch, num_layers)

        # 训练和推理统一限制，防止单层饱和
        gate_values = gate_values.clamp(min=0.0, max=self.gate_max)

        # Gate entropy loss: 最大化各层 gate 的均匀性
        p = gate_values / (gate_values.sum(dim=-1, keepdim=True) + 1e-8)
        entropy_loss = (p * torch.log(p + 1e-8)).sum(dim=-1).mean()  # 负熵

        return gate_values, entropy_loss


class SteeringInjection(nn.Module):
    """ Steering 注入模块

    将干预向量注入到骨干模型的隐藏状态中。
    使用动态门控机制控制每个注入层的干预强度。

    Args:
        inject_layers: 注入层索引列表
        v_dim: 干预向量维度
        layer_dim: 骨干模型隐藏层维度
        gate_hidden_dim: 门控 MLP 隐藏层维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        inject_layers: List[int],
        v_dim: int = 1024,
        layer_dim: int = 2560,
        gate_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.inject_layers = inject_layers
        self.num_inject_layers = len(inject_layers)
        self.v_dim = v_dim
        self.layer_dim = layer_dim

        # 动态门控
        self.gate = DynamicGate(
            v_dim,
            self.num_inject_layers,
            gate_hidden_dim,
            gate_max=1.0,
            gate_init_bias=-2.0,
        )

        # 各注入层的投影器
        self.layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(v_dim, layer_dim),
                nn.LayerNorm(layer_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ) for _ in inject_layers
        ])

        # 存储当前注入向量和门控值
        self.current_v_t: Optional[torch.Tensor] = None
        self.current_gate_values: Optional[torch.Tensor] = None
        self.injection_enabled = True

    def set_intervention_vector(self, v_t: torch.Tensor) -> None:
        """设置当前干预向量 - 【方案D】支持多层向量

        Args:
            v_t: 干预向量 (batch, v_dim) 或 (batch, num_layers, v_dim)
        """
        # 【方案D】支持两种输入格式
        gate_dtype = next(self.gate.parameters()).dtype
        if v_t.dim() == 3:
            # 多层向量格式 (batch, num_layers, v_dim)
            self.current_v_t = v_t
            # 【修复】对所有层取平均后计算gate值，转换dtype适配gate_mlp
            v_t_mean = v_t.mean(dim=1).to(gate_dtype)  # (batch, v_dim)
            # 推理时丢弃 entropy_loss，clamp 已在 forward 中统一处理
            gate_values, _ = self.gate(v_t_mean)
            self.current_gate_values = gate_values  # (batch, num_layers)
        else:
            # 单层向量格式 (batch, v_dim) - 向后兼容
            self.current_v_t = v_t
            gate_values, _ = self.gate(v_t.to(gate_dtype))
            self.current_gate_values = gate_values

    def compute_gate_entropy_loss(self, v_t: torch.Tensor) -> torch.Tensor:
        """计算 gate entropy loss，用于训练时约束 gate 分布均匀性

        Args:
            v_t: (batch, v_dim) 或 (batch, num_layers, v_dim)

        Returns:
            entropy_loss: 标量
        """
        gate_dtype = next(self.gate.parameters()).dtype
        if v_t.dim() == 3:
            v_mean = v_t.mean(dim=1).to(gate_dtype)
        else:
            v_mean = v_t.to(gate_dtype)
        _, entropy_loss = self.gate(v_mean)
        return entropy_loss

    def inject(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """注入干预向量到隐藏状态 - 【方案D】支持多层向量

        Args:
            hidden_states: 隐藏状态 (batch, seq_len, layer_dim)
            layer_idx: 当前层在 inject_layers 中的索引

        Returns:
            注入后的隐藏状态 (batch, seq_len, layer_dim)
        """
        if not self.injection_enabled:
            return hidden_states
        if self.current_v_t is None:
            raise ValueError("Please call set_intervention_vector first")

        # 获取当前层的门控值
        gate = self.current_gate_values[:, layer_idx]  # (batch,)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)

        # 【方案D】获取当前层的干预向量
        if self.current_v_t.dim() == 3:
            # 多层向量格式 (batch, num_layers, v_dim)
            v_t_layer = self.current_v_t[:, layer_idx, :]  # (batch, v_dim)
        else:
            # 单层向量格式 (batch, v_dim) - 向后兼容
            v_t_layer = self.current_v_t

        # 投影干预向量到当前层维度，并保持与hidden_states相同的dtype
        proj = self.layer_projectors[layer_idx](v_t_layer)  # (batch, layer_dim)
        proj = proj.unsqueeze(1)  # (batch, 1, layer_dim)

        # 转换为与hidden_states相同的dtype
        proj = proj.to(hidden_states.dtype)
        gate = gate.to(hidden_states.dtype)

        # 门控注入（使用 InjectFunction 断开 backbone 梯度）
        return InjectFunction.apply(hidden_states, gate * proj)

    def inject_with_mask(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """带注意力掩码的注入

        Args:
            hidden_states: 隐藏状态 (batch, seq_len, layer_dim)
            layer_idx: 当前层在 inject_layers 中的索引
            attention_mask: 注意力掩码 (batch, seq_len)

        Returns:
            注入后的隐藏状态
        """
        injected = self.inject(hidden_states, layer_idx)

        # 应用注意力掩码
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            injected = injected * mask

        return injected

    def get_gate_distribution(self) -> torch.Tensor:
        """获取门控值分布

        Returns:
            门控值 (batch, num_layers)
        """
        return self.current_gate_values

    def reset(self) -> None:
        """重置当前状态"""
        self.current_v_t = None
        self.current_gate_values = None


class HierarchicalSteeringInjection(nn.Module):
    """层级 Steering 注入

    支持多层级的注入策略：
    1. 粗粒度：所有注入层使用相同的门控值
    2. 细粒度：每个注入层使用独立的门控值

    Args:
        inject_layers: 注入层索引列表
        v_dim: 干预向量维度
        layer_dim: 骨干模型隐藏层维度
    """

    def __init__(
        self,
        inject_layers: List[int],
        v_dim: int = 1024,
        layer_dim: int = 2560,
    ):
        super().__init__()
        self.inject_layers = inject_layers

        # 粗粒度门控 (所有层共享)
        self.coarse_gate = DynamicGate(v_dim, 1)

        # 细粒度门控 (每层独立)
        self.fine_gates = nn.ModuleList([
            DynamicGate(v_dim, 1) for _ in inject_layers
        ])

        # 各层投影器
        self.layer_projectors = nn.ModuleList([
            nn.Linear(v_dim, layer_dim) for _ in inject_layers
        ])

        self.current_v_t: Optional[torch.Tensor] = None
        self.injection_enabled = True

    def set_intervention_vector(
        self,
        v_t: torch.Tensor,
        use_coarse: bool = True,
    ) -> None:
        """设置干预向量并计算门控

        Args:
            v_t: 干预向量 (batch, v_dim)
            use_coarse: 是否使用粗粒度门控
        """
        self.current_v_t = v_t
        coarse_gate, _ = self.coarse_gate(v_t)  # (batch, 1)
        self.current_coarse_gate = coarse_gate

        if not use_coarse:
            self.current_fine_gates = [
                gate(v_t)[0] for gate in self.fine_gates
            ]  # list of (batch, 1)

    def inject(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """注入

        Args:
            hidden_states: 隐藏状态 (batch, seq_len, layer_dim)
            layer_idx: 注入层索引

        Returns:
            注入后的隐藏状态
        """
        if not self.injection_enabled:
            return hidden_states
        if self.current_v_t is None:
            raise ValueError("Call set_intervention_vector first")

        if hasattr(self, 'current_fine_gates'):
            gate = self.current_fine_gates[layer_idx]
        else:
            gate = self.current_coarse_gate

        # 确保 gate 形状正确 (batch,) -> (batch, 1, 1)
        if gate.dim() == 2:
            gate = gate.squeeze(-1)  # (batch,)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)

        proj = self.layer_projectors[layer_idx](self.current_v_t)  # (batch, layer_dim)
        proj = proj.unsqueeze(1)  # (batch, 1, layer_dim)

        return InjectFunction.apply(hidden_states, gate * proj)
