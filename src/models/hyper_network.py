"""
超网络模块 - 从用户话语生成干预向量

HyperNetwork 负责从用户当前话语和历史干预向量生成新的干预向量。
使用冻结的编码器提取语义特征，并通过残差MLP进行特征融合和变换。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .components import ResidualMLP


class HyperNetwork(nn.Module):
    """超网络

    从用户话语生成干预向量的神经网络模块。
    使用冻结的编码器提取语义特征，并与历史干预向量融合。

    Args:
        encoder: 冻结的文本编码器 (Qwen3-Embedding)
        v_dim: 干预向量维度
        hidden_dim: MLP 隐藏层维度
        num_layers: 投影 MLP 层数
    """

    def __init__(
        self,
        encoder: nn.Module,
        v_dim: int = 1024,
        hidden_dim: int = 4096,
        num_layers: int = 3,
        encoder_dim: int = None,
        v_norm_clip: float = None,  # 【废弃】不再使用hard clip
        use_soft_constraint: bool = True,  # 【新增】使用soft constraint
        num_inject_layers: int = 8,  # 【方案D】注入层数量
        use_layer_embedding: bool = True,  # 【方案D】是否使用层嵌入
    ):
        super().__init__()
        self.encoder = encoder
        self.v_dim = v_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.v_norm_clip = None  # 【修复】禁用hard clip
        self.use_soft_constraint = use_soft_constraint  # 【新增】
        self.num_inject_layers = num_inject_layers  # 【方案D】
        self.use_layer_embedding = use_layer_embedding  # 【方案D】

        # Encoder 输出投影器（如果encoder输出维度与v_dim不同）
        if encoder_dim is not None and encoder_dim != v_dim:
            self.encoder_projector = nn.Linear(encoder_dim, v_dim)
        else:
            self.encoder_projector = None

        # 历史向量投影器
        self.history_projector = nn.Linear(v_dim, v_dim)

        # 【方案A】Query-aware gate fusion（替代 buggy Cross-Attention）
        # 输入: [persona_emb; query_emb]，输出: alpha 融合权重
        self.query_gate = nn.Sequential(
            nn.Linear(v_dim * 2, v_dim // 4),
            nn.SiLU(),
            nn.Linear(v_dim // 4, 1),
            nn.Sigmoid(),
        )

        # 【方案D】层嵌入 - 为每层生成独特的嵌入
        if use_layer_embedding:
            self.layer_embedding = nn.Embedding(num_inject_layers, v_dim)
            # 初始化层嵌入，使用较小的方差
            nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.02)
        else:
            self.layer_embedding = None

        # 投影 MLP 序列
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualMLP(v_dim, hidden_dim))
        layers.append(nn.LayerNorm(v_dim))
        self.projector = nn.Sequential(*layers)

        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode_text(self, texts: list[str], tokenizer=None) -> torch.Tensor:
        """编码文本为向量

        Args:
            texts: 文本列表
            tokenizer: 可选的外部分词器

        Returns:
            编码向量 (batch, v_dim)
        """
        import logging
        logger = logging.getLogger(__name__)

        # 使用传入的 tokenizer 或模型自带的
        if tokenizer is None:
            if hasattr(self.encoder, 'tokenizer'):
                tokenizer = self.encoder.tokenizer
            else:
                raise ValueError("需要提供 tokenizer 参数或使用带有 tokenizer 的 encoder")

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )

        # 移动到 encoder 所在设备
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        # 获取 embeddings
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output.unsqueeze(1)
        else:
            # 假设输出就是 embeddings
            embeddings = outputs

        # Mean pooling
        encoded = torch.mean(embeddings, dim=1)
        return encoded

    def forward(
        self,
        personality_texts: list[str],
        user_query_texts: list[str],
        v_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 【方案A】Query-Aware + 【方案D】多层输出

        Args:
            personality_texts: 人格描述文本列表
            user_query_texts: 当前用户query列表
            v_prev: 上一轮干预向量 (batch, v_dim)

        Returns:
            v_t_layers: 当前干预向量 (batch, num_layers, v_dim) 【方案D】多层输出
            z_t: 编码器输出向量 (batch, v_dim)
            v_norm: 干预向量平均norm (batch,)
        """
        import logging
        logger = logging.getLogger(__name__)

        # Step 1: 使用冻结的编码器提取语义特征
        # 获取 tokenizer
        if hasattr(self.encoder, 'tokenizer'):
            tokenizer = self.encoder.tokenizer
        elif hasattr(self, '_tokenizer'):
            tokenizer = self._tokenizer
        else:
            raise ValueError("Encoder 没有 tokenizer，请使用 set_tokenizer 方法设置")

        # 【方案A】分离编码 personality 和 query
        encoder_device = next(self.encoder.parameters()).device

        # 编码 personality
        personality_inputs = tokenizer(
            personality_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        personality_inputs = {k: v.to(encoder_device) for k, v in personality_inputs.items()}

        with torch.no_grad():
            personality_outputs = self.encoder(**personality_inputs)

        if hasattr(personality_outputs, 'last_hidden_state'):
            z_personality = personality_outputs.last_hidden_state.mean(dim=1)
        elif hasattr(personality_outputs, 'pooler_output'):
            z_personality = personality_outputs.pooler_output
        else:
            z_personality = personality_outputs.mean(dim=1)

        # 编码 query
        query_inputs = tokenizer(
            user_query_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        query_inputs = {k: v.to(encoder_device) for k, v in query_inputs.items()}

        with torch.no_grad():
            query_outputs = self.encoder(**query_inputs)

        if hasattr(query_outputs, 'last_hidden_state'):
            z_query = query_outputs.last_hidden_state.mean(dim=1)
        elif hasattr(query_outputs, 'pooler_output'):
            z_query = query_outputs.pooler_output
        else:
            z_query = query_outputs.mean(dim=1)

        # 转换为float并对齐设备
        target_device = v_prev.device
        z_personality = z_personality.float().to(target_device)
        z_query = z_query.float().to(target_device)

        # 投影到 v_dim
        if self.encoder_projector is not None:
            z_personality = self.encoder_projector(z_personality)
            z_query = self.encoder_projector(z_query)

        # 【方案A】Query-aware gate fusion
        # alpha 越大 → 越依赖 personality；alpha 越小 → 越依赖 query
        gate_input = torch.cat([z_personality, z_query], dim=-1)  # (batch, v_dim*2)
        alpha = self.query_gate(gate_input)                         # (batch, 1)
        z_fused = alpha * z_personality + (1 - alpha) * z_query    # (batch, v_dim)
        z_t = z_fused  # 用于返回

        # Step 2: 融合历史向量
        h_prev = self.history_projector(v_prev.float())
        fused = z_fused + h_prev

        # Step 3: 【方案D】为每层生成独立的干预向量
        batch_size = fused.size(0)
        
        if self.use_layer_embedding and self.layer_embedding is not None:
            # 获取层嵌入 (num_layers, v_dim)
            layer_indices = torch.arange(self.num_inject_layers, device=fused.device)
            layer_embeds = self.layer_embedding(layer_indices)  # (num_inject_layers, v_dim)
            
            # 扩展fused以匹配层数 (batch, num_layers, v_dim)
            fused_expanded = fused.unsqueeze(1).expand(batch_size, self.num_inject_layers, self.v_dim)
            
            # 添加层嵌入
            fused_with_layer = fused_expanded + layer_embeds.unsqueeze(0)  # (batch, num_layers, v_dim)
            
            # 投影生成多层向量
            # 将所有层展平处理
            fused_flat = fused_with_layer.view(batch_size * self.num_inject_layers, self.v_dim)
            v_t_flat = self.projector(fused_flat)
            v_t_layers = v_t_flat.view(batch_size, self.num_inject_layers, self.v_dim)
        else:
            # 不使用层嵌入，直接复制
            v_t = self.projector(fused)  # (batch, v_dim)
            v_t_layers = v_t.unsqueeze(1).expand(batch_size, self.num_inject_layers, self.v_dim)

        # Step 4: 计算v_norm用于soft constraint
        v_norm = v_t_layers.norm(dim=-1).mean(dim=-1)  # (batch,) 平均所有层的norm

        return v_t_layers, z_t, v_norm  # 【方案D】返回多层向量

    def set_tokenizer(self, tokenizer):
        """设置分词器

        Args:
            tokenizer: 分词器对象
        """
        self._tokenizer = tokenizer
        return self

    def get_trainable_params(self) -> int:
        """获取可训练参数量

        Returns:
            可训练参数数量 (不包括冻结的 encoder)
        """
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )


class HyperNetworkWithAttention(nn.Module):
    """带注意力机制的超网络

    在历史向量融合时使用注意力机制，更好地整合历史信息。

    Args:
        encoder: 冻结的文本编码器
        v_dim: 干预向量维度
        hidden_dim: MLP 隐藏层维度
        num_layers: 投影 MLP 层数
    """

    def __init__(
        self,
        encoder: nn.Module,
        v_dim: int = 1024,
        hidden_dim: int = 4096,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hyper_net = HyperNetwork(encoder, v_dim, hidden_dim, num_layers)

        # 注意力机制
        self.query_proj = nn.Linear(v_dim, v_dim)
        self.key_proj = nn.Linear(v_dim, v_dim)
        self.value_proj = nn.Linear(v_dim, v_dim)
        self.attention_norm = nn.LayerNorm(v_dim)

    def forward(
        self,
        user_texts: list[str],
        v_prev: torch.Tensor,
        v_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            user_texts: 当前用户话语列表
            v_prev: 上一轮干预向量 (batch, v_dim)
            v_history: 历史干预向量列表 [(batch, v_dim), ...]

        Returns:
            v_t: 当前干预向量 (batch, v_dim)
            z_t: 编码器输出向量 (batch, v_dim)
        """
        # 如果有历史向量，使用注意力融合
        if v_history is not None and len(v_history) > 0:
            history_tensor = torch.stack(v_history, dim=1)  # (batch, history_len, v_dim)

            # 计算注意力
            q = self.query_proj(v_prev.float().to(self.query_proj.weight.device)).unsqueeze(1)  # (batch, 1, v_dim)
            k = self.key_proj(history_tensor)  # (batch, history_len, v_dim)
            v = self.value_proj(history_tensor)  # (batch, history_len, v_dim)

            # 注意力分数
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hyper_net.v_dim ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # 注意力输出
            attn_output = torch.matmul(attn_weights, v)  # (batch, 1, v_dim)
            attn_output = attn_output.squeeze(1)  # (batch, v_dim)

            # 融合到 v_prev
            v_prev = self.attention_norm(v_prev + attn_output)

        return self.hyper_net(user_texts, v_prev)
