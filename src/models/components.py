"""
基础组件模块 - 可复用的神经网络组件

提供 ResidualMLP 等基础组件，用于构建 PersonaSteer 的核心模块。
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualMLP(nn.Module):
    """残差MLP块

    带残差连接的MLP块，用于特征变换和维度映射。
    支持可配置的隐藏层维度、激活函数和dropout。

    Args:
        dim: 输入输出维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (..., dim)

        Returns:
            输出张量 (..., dim)
        """
        h = self.activation(self.linear1(x))
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        # 残差连接 + LayerNorm
        return self.layer_norm(x + h)


class GatedResidualMLP(nn.Module):
    """带门控的残差MLP块

    在残差连接上添加可学习的门控机制，允许网络学习残差的重要性。

    Args:
        dim: 输入输出维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = ResidualMLP(dim, hidden_dim, dropout)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (..., dim)

        Returns:
            输出张量 (..., dim)
        """
        return x + self.gate * self.mlp(x)


class MultiHeadProjection(nn.Module):
    """多头投影模块

    将单个向量投影到多个不同的输出空间，用于生成多个干预向量。

    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        num_heads: 投影头数量
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (batch, in_dim)

        Returns:
            输出张量 (batch, num_heads, out_dim)
        """
        outputs = []
        for proj in self.projections:
            outputs.append(proj(x))
        return torch.stack(outputs, dim=1)


def count_parameters(module: nn.Module) -> int:
    """统计模块的可训练参数量

    Args:
        module: PyTorch 模块

    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def freeze_module(module: nn.Module) -> None:
    """冻结模块参数

    Args:
        module: PyTorch 模块
    """
    for param in module.parameters():
        param.requires_grad = False


def verify_frozen(module: nn.Module) -> bool:
    """验证模块是否已冻结

    Args:
        module: PyTorch 模块

    Returns:
        如果所有参数都冻结返回 True
    """
    return all(not p.requires_grad for p in module.parameters())
