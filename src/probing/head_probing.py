"""
激活值收集器 - 收集 Qwen3-4B 各层各头的激活值

使用 PyTorch hook 机制收集 attention head 的输出，
并计算与属性值的 Spearman 相关性。
"""

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class AttentionHeadProber:
    """Attention Head Prober - 收集各层各头的激活值并计算相关性"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化 Prober

        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备
        """
        self.device = device

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # 加载模型 (使用 FP16 减少显存)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        self.model.eval()

        # 模型配置
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads

        # 存储激活值: {layer_idx: {head_idx: tensor}}
        self.activations: dict[int, dict[int, torch.Tensor]] = {}

        # 注册 hooks
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hooks 来收集激活值"""
        self._hooks = []
        for layer_idx in range(self.num_layers):
            self.activations[layer_idx] = None

            # 获取 attention 层的输出
            # Qwen3: self.model.layers / Qwen2.5: self.model.model.layers
            if hasattr(self.model, 'layers'):
                layer = self.model.layers[layer_idx]
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layer = self.model.model.layers[layer_idx]
            else:
                raise ValueError(f"Cannot find layers in model structure: {type(self.model)}")
            
            attn = layer.self_attn

            # 注册 hook 到 attention 模块
            hook = attn.register_forward_hook(
                self._create_hook(layer_idx),
            )
            self._hooks.append(hook)

    def _create_hook(self, layer_idx: int):
        """创建指定层的 hook

        收集 attention 输出并重塑为 (batch, num_heads, head_dim)
        """

        def hook(module, input, output):
            # output 可能是 tuple (hidden_states, ...) 或单个 tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states shape: (batch, seq_len, hidden_size)
            batch_size, seq_len, hidden_size = hidden_states.shape

            # 重塑为 (batch, seq_len, num_heads, head_dim)
            reshaped = hidden_states.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            )

            # 取最后一个 token 的表示 (batch, num_heads, head_dim)
            last_token = reshaped[:, -1, :, :].detach().cpu()

            # 存储激活值
            self.activations[layer_idx] = last_token

        return hook

    def _clear_hooks(self):
        """清除所有已注册的 hooks"""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
            self._hooks = []

    def collect(
        self,
        texts: list[str],
        max_length: int = 512,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        收集文本的激活值

        Args:
            texts: 输入文本列表
            max_length: 最大序列长度
            batch_size: 批处理大小

        Returns:
            activations: (N, num_layers, num_heads, head_dim) 激活值张量
        """
        all_activations = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
            batch_texts = texts[i : i + batch_size]

            # tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            # 前向传播 (会触发 hooks 收集激活值)
            with torch.no_grad():
                self.model(**inputs)

            # 从 hooks 中收集各层激活值
            # 每层激活值: (batch, num_heads, head_dim)
            layer_activations = []
            for layer_idx in range(self.num_layers):
                if self.activations[layer_idx] is not None:
                    layer_activations.append(self.activations[layer_idx])
                else:
                    # fallback: 使用零张量
                    layer_activations.append(
                        torch.zeros(len(batch_texts), self.num_heads, self.head_dim)
                    )

            # stack 为 (batch, num_layers, num_heads, head_dim)
            batch_acts = torch.stack(layer_activations, dim=1)
            all_activations.append(batch_acts)

        # 拼接所有批次: (N, num_layers, num_heads, head_dim)
        activations = torch.cat(all_activations, dim=0)

        return activations

    def compute_spearman(
        self,
        activations: torch.Tensor,
        attr_values: list[float],
    ) -> torch.Tensor:
        """
        计算激活值与属性值的 Spearman 相关性

        Args:
            activations: (N, num_layers, num_heads, head_dim) 激活值
            attr_values: (N,) 属性值列表

        Returns:
            correlation_matrix: (num_layers, num_heads) 相关性矩阵
        """
        attr_tensor = torch.tensor(attr_values, dtype=torch.float32)

        correlation_matrix = torch.zeros(self.num_layers, self.num_heads)

        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                # 提取该层的激活值
                # shape: (N, head_dim)
                head_activations = activations[:, layer_idx, head_idx, :]

                # 对 head_dim 维度求均值
                # shape: (N,)
                head_mean = head_activations.mean(dim=1)

                # 计算 Spearman 相关性
                corr, _ = spearmanr(head_mean.numpy(), attr_tensor.numpy())
                correlation_matrix[layer_idx, head_idx] = corr

        return correlation_matrix

    def compute_correlation_matrix(
        self,
        texts: list[str],
        attributes: dict[str, list[float]],
    ) -> dict[str, torch.Tensor]:
        """
        计算所有属性与各层各头的相关性矩阵

        Args:
            texts: 输入文本列表
            attributes: 属性字典，{attr_name: [values]}

        Returns:
            correlation_matrices: {attr_name: (num_layers, num_heads) tensor}
        """
        # 收集激活值
        activations = self.collect(texts)

        # 计算每个属性的相关性
        correlation_matrices = {}

        for attr_name, attr_values in attributes.items():
            corr_matrix = self.compute_spearman(activations, attr_values)
            correlation_matrices[attr_name] = corr_matrix

        return correlation_matrices


def create_prober(
    model_name: str = "Qwen/Qwen3-4B",
    device: str = None,
) -> AttentionHeadProber:
    """
    创建 Prober 的工厂函数

    Args:
        model_name: 模型名称
        device: 设备 (自动检测)

    Returns:
        AttentionHeadProber 实例
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return AttentionHeadProber(model_name=model_name, device=device)
