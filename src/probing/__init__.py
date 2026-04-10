"""
Probing 模块 - Layer × Head 粒度分析
用于确定最优注入层配置
"""

from .attribute_extractor import AttributeExtractor
from .head_probing import AttentionHeadProber
from .visualize import plot_heatmap, select_injection_layers

__all__ = [
    "AttributeExtractor",
    "AttentionHeadProber",
    "plot_heatmap",
    "select_injection_layers",
]
