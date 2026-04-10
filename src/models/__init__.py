"""
PersonaSteer 模型模块

提供核心模型组件:
- ResidualMLP 等基础组件
- HyperNetwork 超网络
- SteeringInjection 注入模块
- PersonaSteerModel 完整模型
"""

from .components import (
    ResidualMLP,
    GatedResidualMLP,
    MultiHeadProjection,
    count_parameters,
    freeze_module,
    verify_frozen,
)
from .hyper_network import HyperNetwork, HyperNetworkWithAttention
from .injection import (
    DynamicGate,
    SteeringInjection,
    HierarchicalSteeringInjection,
)
from .persona_steer import (
    PersonaSteerConfig,
    PersonaSteerModel,
    create_personasteer_model,
)

__all__ = [
    # Components
    "ResidualMLP",
    "GatedResidualMLP",
    "MultiHeadProjection",
    "count_parameters",
    "freeze_module",
    "verify_frozen",
    # Hyper Network
    "HyperNetwork",
    "HyperNetworkWithAttention",
    # Injection
    "DynamicGate",
    "SteeringInjection",
    "HierarchicalSteeringInjection",
    # Main Model
    "PersonaSteerConfig",
    "PersonaSteerModel",
    "create_personasteer_model",
]
