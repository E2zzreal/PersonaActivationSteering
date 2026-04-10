#!/usr/bin/env python3
"""
PersonaSteer V2 训练可视化脚本
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 训练数据
stages = ['Stage 1', 'Stage 2', 'Stage 3']
best_loss = [0.0943, 0.0341, 0.0006]
best_epoch = [2, 2, 4]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss对比柱状图
ax1 = axes[0, 0]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax1.bar(stages, best_loss, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Best Loss', fontsize=12)
ax1.set_title('各Stage最佳Loss对比', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, loss in zip(bars, best_loss):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. Loss改善率
ax2 = axes[0, 1]
improvement = [0, 64, 98]  # 相对于前一阶段的改善百分比
colors2 = ['#95a5a6', '#27ae60', '#c0392b']
bars2 = ax2.bar(stages, improvement, color=colors2, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Loss改善率 (%)', fontsize=12)
ax2.set_title('各Stage Loss改善率', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, imp in zip(bars2, improvement):
    if imp > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., imp,
                 f'↓{imp}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. 模拟训练曲线（基于checkpoint数据估算）
ax3 = axes[1, 0]

# Stage 1 训练曲线（估算）
epochs1 = np.linspace(0, 3, 100)
loss1 = 12.13 * np.exp(-epochs1 * 2) + 0.09 + np.random.randn(100) * 0.01
loss1 = np.clip(loss1, 0.08, 12.2)

# Stage 2 训练曲线
epochs2 = np.linspace(0, 3, 100)
loss2 = 0.094 * np.exp(-epochs2 * 1.5) + 0.034 + np.random.randn(100) * 0.005
loss2 = np.clip(loss2, 0.03, 0.1)

# Stage 3 训练曲线
epochs3 = np.linspace(0, 5, 100)
loss3 = 0.034 * np.exp(-epochs3 * 1.2) + 0.0006 + np.random.randn(100) * 0.0005
loss3 = np.clip(loss3, 0.0005, 0.04)

ax3.plot(epochs1, loss1, label='Stage 1', linewidth=2, color='#3498db')
ax3.plot(epochs2 + 3, loss2, label='Stage 2', linewidth=2, color='#2ecc71')
ax3.plot(epochs3 + 6, loss3, label='Stage 3', linewidth=2, color='#e74c3c')

ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('训练Loss曲线（估算）', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 标注最佳点
ax3.scatter([2], [0.0943], color='#3498db', s=100, zorder=5, marker='*')
ax3.scatter([5], [0.0341], color='#2ecc71', s=100, zorder=5, marker='*')
ax3.scatter([10], [0.0006], color='#e74c3c', s=100, zorder=5, marker='*')

# 4. 累计改善
ax4 = axes[1, 1]
total_improvement = [0.0943, 0.0341, 0.0006]
cumulative = [100, 36, 0.6]  # 相对于初始Loss的百分比

ax4.plot(stages, cumulative, 'o-', linewidth=3, markersize=12, color='#9b59b6')
ax4.fill_between(stages, cumulative, alpha=0.3, color='#9b59b6')
ax4.set_ylabel('相对于Stage 1 Loss (%)', fontsize=12)
ax4.set_title('Loss累计改善', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 110)

for i, (stage, cum) in enumerate(zip(stages, cumulative)):
    ax4.annotate(f'{cum:.1f}%', (stage, cum), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()

# 保存图表
output_dir = '/home/kemove/.openclaw/workspace/memory'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'personasteer_training_analysis.png'), dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_dir}/personasteer_training_analysis.png")

plt.close()

# 创建第二个图表：注入层对比
fig2, ax = plt.subplots(figsize=(10, 6))

models = ['Qwen2.5-3B', 'Qwen3-4B\n(错误层)', 'Qwen3-4B\n(正确层)']
inject_layers = ['[10-17]\n(Probing推荐)', '[26-33]\n(错误选择)', '[0-7]\n(Probing推荐)']
loss_values = [0.094, 7.5, 0.001]

colors = ['#27ae60', '#e74c3c', '#27ae60']
bars = ax.bar(models, loss_values, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Stage 1 Loss', fontsize=12)
ax.set_title('注入层位置对Loss的影响', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# 添加注入层标注
for bar, layers in zip(bars, inject_layers):
    ax.text(bar.get_x() + bar.get_width()/2., 0.5,
            layers, ha='center', va='bottom', fontsize=10)

# 添加数值标签
for bar, loss in zip(bars, loss_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.5,
            f'{loss:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'inject_layer_comparison.png'), dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_dir}/inject_layer_comparison.png")

plt.close()

print("\n可视化完成!")
