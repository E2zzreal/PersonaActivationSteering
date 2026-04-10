"""
训练损失函数
包含 SFT Loss 和 Supervised Contrastive Loss

修复说明 (2026-03-25):
- 原始实现：使用user_id构建正例，但数据中每个user_id唯一 → 零正例
- 修复方案：使用personality相似度构建正例
  1. 同一personality模板为正例
  2. personality embedding相似度>threshold为正例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    计算监督微调损失 (语言模型标准交叉熵损失)

    Args:
        logits: (batch, seq_len, vocab_size) 模型输出的 logits
        labels: (batch, seq_len) 目标 token ids
        ignore_index: 忽略的 label 值（通常为 -100）

    Returns:
        torch.Tensor: 标量损失值
    """
    # Reshape: (batch * seq_len, vocab_size) x (batch * seq_len)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    有监督对比损失

    【修复版】使用personality相似度构建正例，而非user_id。
    
    支持两种模式：
    - personality_mode: 同一personality字符串为正例
    - embedding_mode: personality embedding相似度>threshold为正例
    
    使用 InfoNCE 损失进行优化。

    Args:
        temperature: 温度参数，控制分布锐度
        pos_threshold: personality embedding相似度阈值（仅embedding_mode）
    """

    def __init__(
        self,
        temperature: float = 0.07,
        pos_threshold: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.pos_threshold = pos_threshold

    def forward(
        self,
        v_t: torch.Tensor,
        personalities: list[str],
        personality_embeddings=None,
        user_ids=None,  # 新增：优先用 user_id 构建正样本对
    ) -> torch.Tensor:
        """
        计算对比损失 - 【方案D】支持多层向量

        Args:
            v_t: (batch, hidden_dim) 或 (batch, num_layers, hidden_dim) 干预向量
            personalities: list[str] personality描述列表
            personality_embeddings: (batch, embed_dim) personality embedding（可选）

        Returns:
            torch.Tensor: 标量损失值
        """
        # 【方案D】支持多层向量输入
        if v_t.dim() == 3:
            # 多层向量格式 (batch, num_layers, hidden_dim)
            # 对所有层取平均用于对比学习
            v_t = v_t.mean(dim=1)  # (batch, hidden_dim)
        
        batch_size = v_t.size(0)

        # 单样本时返回零损失
        if batch_size <= 1:
            return torch.tensor(0.0, device=v_t.device, requires_grad=True)

        # L2 归一化
        v_norm = F.normalize(v_t, dim=-1)

        # 计算相似度矩阵 (batch, batch)
        sim_matrix = torch.matmul(v_norm, v_norm.T) / self.temperature

        # 优先用 user_id 构建正样本对（最可靠）
        if user_ids is not None:
            pos_mask = torch.zeros(batch_size, batch_size)
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and user_ids[i] == user_ids[j]:
                        pos_mask[i, j] = 1.0
            pos_mask = pos_mask.to(v_t.device)
        # 模式1：使用personality embedding相似度
        elif personality_embeddings is not None:
            pos_mask = self._build_pos_mask_by_embedding(
                personality_embeddings,
                self.pos_threshold
            ).to(v_t.device)
        # 模式2：使用同一personality字符串
        else:
            pos_mask = self._build_pos_mask_by_personality(personalities).to(v_t.device)

        # 去除自相似度
        mask_diag = torch.eye(batch_size, device=v_t.device)
        sim_matrix = sim_matrix * (1 - mask_diag)
        pos_mask = pos_mask * (1 - mask_diag)

        # InfoNCE 损失
        exp_sim = torch.exp(sim_matrix)

        # 分母: 每行的 exp 之和
        denom = exp_sim.sum(dim=1, keepdim=True)
        # 避免除零
        denom = torch.clamp(denom, min=1e-8)

        # log probability
        log_prob = sim_matrix - torch.log(denom)

        # 只对正例计算损失
        # 确保有正例
        if pos_mask.sum() < 1e-8:
            # 【修复】如果没有正例，返回零损失而非报错
            return torch.tensor(0.0, device=v_t.device, requires_grad=True)

        loss = -(pos_mask * log_prob).sum() / pos_mask.sum()

        return loss

    def _build_pos_mask_by_personality(self, personalities: list[str]) -> torch.Tensor:
        """
        基于personality字符串构建正例掩码
        
        同一personality的样本为正例
        
        Args:
            personalities: personality描述列表
            
        Returns:
            (batch, batch) 正例掩码矩阵
        """
        batch_size = len(personalities)
        pos_mask = torch.zeros(batch_size, batch_size)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and personalities[i] == personalities[j]:
                    pos_mask[i, j] = 1.0
        
        return pos_mask

    def _build_pos_mask_by_embedding(
        self, 
        personality_embeddings: torch.Tensor,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """
        基于personality embedding相似度构建正例掩码
        
        相似度>threshold的样本为正例
        
        Args:
            personality_embeddings: (batch, embed_dim) personality embedding
            threshold: 相似度阈值
            
        Returns:
            (batch, batch) 正例掩码矩阵
        """
        # L2归一化
        emb_norm = F.normalize(personality_embeddings, dim=-1)
        
        # 计算余弦相似度
        sim = torch.matmul(emb_norm, emb_norm.T)
        
        # 构建正例掩码
        pos_mask = (sim > threshold).float()
        
        return pos_mask


class PersonaSteerLoss(nn.Module):
    """
    组合损失函数

    结合 SFT 损失和对比损失，支持可配置的权重。
    
    【修复版】支持personality-based对比学习

    Args:
        sft_weight: SFT 损失权重
        scl_weight: 对比损失权重
        temperature: 对比损失温度参数
        pos_threshold: personality embedding相似度阈值
    """

    def __init__(
        self,
        sft_weight: float = 1.0,
        scl_weight: float = 0.1,
        temperature: float = 0.07,
        pos_threshold: float = 0.7,
    ):
        super().__init__()
        self.sft_weight = sft_weight
        self.scl_weight = scl_weight
        self.scl = SupervisedContrastiveLoss(temperature, pos_threshold)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        v_t: torch.Tensor,
        personalities: list[str],
        personality_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算组合损失

        Args:
            logits: (batch, seq_len, vocab_size) 模型输出的 logits
            labels: (batch, seq_len) 目标 token ids
            v_t: (batch, hidden_dim) 干预向量
            personalities: list[str] personality描述列表
            personality_embeddings: (batch, embed_dim) personality embedding（可选）

        Returns:
            tuple: (total_loss, sft_loss, scl_loss)
        """
        loss_sft = compute_sft_loss(logits, labels)
        loss_scl = self.scl(v_t, personalities, personality_embeddings)

        loss_total = self.sft_weight * loss_sft + self.scl_weight * loss_scl

        return loss_total, loss_sft, loss_scl
