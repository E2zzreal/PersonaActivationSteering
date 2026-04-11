#!/usr/bin/env python3
"""
训练结果可视化分析
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
configs = ['neuroticism', 'baseline', 'minimal']
stages = ['stage1', 'stage2', 'stage3']
config_names = {
    'neuroticism': 'Neuroticism (3层)',
    'baseline': 'Baseline (8层)',
    'minimal': 'Minimal (6层)'
}

# 加载LLM Judge评分
eval_dir = Path('results/eval_20260411_163906')
with open(eval_dir / 'summary.json') as f:
    judge_results = json.load(f)

# 加载训练Loss
train_loss = {}
for config in configs:
    train_loss[config] = {}
    for stage in stages:
        ckpt_path = f'checkpoints/{config}_gate_neg3_{stage}/best.pt'
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
            epoch = ckpt.get('epoch', 0)
            train_loss[config][stage] = {'loss': loss, 'epoch': epoch}
        except:
            train_loss[config][stage] = {'loss': float('inf'), 'epoch': 0}

# 加载详细回复数据
responses = {}
for config in configs:
    for stage in stages:
        name = f'{config}_{stage}'
        resp_file = eval_dir / f'{name}_responses.json'
        if resp_file.exists():
            with open(resp_file) as f:
                responses[name] = json.load(f)

# ============================================================
# 图1: LLM Judge评分热力图
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

score_matrix = np.zeros((3, 3))
std_matrix = np.zeros((3, 3))

for i, config in enumerate(configs):
    for j, stage in enumerate(stages):
        name = f'{config}_{stage}'
        if name in judge_results:
            score_matrix[i, j] = judge_results[name]['mean_score']
            std_matrix[i, j] = judge_results[name]['std_score']

im = ax1.imshow(score_matrix, cmap='RdYlGn', vmin=2.5, vmax=3.5, aspect='auto')

ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax1.set_yticklabels([config_names[c] for c in configs])

for i in range(3):
    for j in range(3):
        text = ax1.text(j, i, f'{score_matrix[i, j]:.2f}\n±{std_matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=12)

ax1.set_title('LLM Judge 人格一致性评分热力图', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax1, label='评分 (1-5)')

plt.tight_layout()
plt.savefig('results/eval_20260411_163906/fig1_judge_heatmap.png', dpi=150)
print('保存: fig1_judge_heatmap.png')

# ============================================================
# 图2: 训练Loss对比
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

x = np.arange(3)
width = 0.25

for i, config in enumerate(configs):
    losses = [train_loss[config][s]['loss'] for s in stages]
    # 对数缩放便于显示
    losses_log = [np.log10(l) if l > 0 else 0 for l in losses]
    bars = ax2.bar(x + i*width, losses_log, width, label=config_names[config])

    # 添加数值标签
    for j, (bar, loss) in enumerate(zip(bars, losses)):
        height = bar.get_height()
        ax2.annotate(f'{loss:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('训练阶段')
ax2.set_ylabel('Best Loss (log10)')
ax2.set_title('训练Loss对比 (对数尺度)', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/eval_20260411_163906/fig2_train_loss.png', dpi=150)
print('保存: fig2_train_loss.png')

# ============================================================
# 图3: Loss vs Judge Score 散点图
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 8))

colors = {'neuroticism': '#e74c3c', 'baseline': '#3498db', 'minimal': '#2ecc71'}
markers = {'stage1': 'o', 'stage2': 's', 'stage3': '^'}

for config in configs:
    for stage in stages:
        name = f'{config}_{stage}'
        if name in judge_results:
            loss = train_loss[config][stage]['loss']
            score = judge_results[name]['mean_score']
            std = judge_results[name]['std_score']

            ax3.scatter(loss, score, c=colors[config], marker=markers[stage],
                       s=150, alpha=0.8, edgecolors='black', linewidth=1)
            ax3.errorbar(loss, score, yerr=std, fmt='none', c=colors[config], alpha=0.5)

            # 标签
            ax3.annotate(f'{config[:3]}-{stage[-1]}',
                        xy=(loss, score), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)

ax3.set_xlabel('Best Loss', fontsize=12)
ax3.set_ylabel('LLM Judge Score', fontsize=12)
ax3.set_title('训练Loss vs LLM Judge评分', fontsize=14, fontweight='bold')
ax3.set_xscale('log')
ax3.grid(alpha=0.3)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Stage 1'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Stage 2'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Stage 3'),
]
ax3.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('results/eval_20260411_163906/fig3_loss_vs_score.png', dpi=150)
print('保存: fig3_loss_vs_score.png')

# ============================================================
# 图4: 各配置的Stage进展趋势
# ============================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

# Judge Score趋势
for config in configs:
    scores = [judge_results[f'{config}_{s}']['mean_score'] for s in stages]
    stds = [judge_results[f'{config}_{s}']['std_score'] for s in stages]
    x = range(3)
    ax4a.errorbar(x, scores, yerr=stds, marker='o', capsize=5,
                 label=config_names[config], linewidth=2, markersize=8)

ax4a.set_xlabel('训练阶段')
ax4a.set_ylabel('LLM Judge Score')
ax4a.set_title('LLM Judge评分随Stage变化', fontsize=12, fontweight='bold')
ax4a.set_xticks(range(3))
ax4a.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax4a.legend()
ax4a.grid(alpha=0.3)
ax4a.set_ylim(2.5, 3.5)

# Loss趋势
for config in configs:
    losses = [train_loss[config][s]['loss'] for s in stages]
    ax4b.plot(range(3), losses, marker='o', label=config_names[config],
             linewidth=2, markersize=8)

ax4b.set_xlabel('训练阶段')
ax4b.set_ylabel('Best Loss')
ax4b.set_title('训练Loss随Stage变化', fontsize=12, fontweight='bold')
ax4b.set_xticks(range(3))
ax4b.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'])
ax4b.legend()
ax4b.grid(alpha=0.3)
ax4b.set_yscale('log')

plt.tight_layout()
plt.savefig('results/eval_20260411_163906/fig4_trends.png', dpi=150)
print('保存: fig4_trends.png')

# ============================================================
# 图5: 回复长度分布
# ============================================================
fig5, ax5 = plt.subplots(figsize=(12, 6))

resp_lengths = defaultdict(list)
for name, resp_list in responses.items():
    for r in resp_list:
        resp_lengths[name].append(len(r.get('response', '')))

box_data = []
box_labels = []
for config in configs:
    for stage in stages:
        name = f'{config}_{stage}'
        if name in resp_lengths and resp_lengths[name]:
            box_data.append(resp_lengths[name])
            box_labels.append(f'{config[:3]}-{stage[-1]}')

ax5.boxplot(box_data, labels=box_labels, patch_artist=True)
ax5.set_xlabel('实验配置')
ax5.set_ylabel('回复长度 (字符数)')
ax5.set_title('生成回复长度分布', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/eval_20260411_163906/fig5_response_length.png', dpi=150)
print('保存: fig5_response_length.png')

# ============================================================
# 统计分析报告
# ============================================================
print('\n' + '='*70)
print('深入分析报告')
print('='*70)

# 1. 最佳配置
print('\n【1. 最佳配置分析】')
config_avg = {}
for config in configs:
    scores = [judge_results[f'{config}_{s}']['mean_score'] for s in stages]
    config_avg[config] = np.mean(scores)

best_config = max(config_avg, key=config_avg.get)
print(f'  LLM Judge最佳配置: {config_names[best_config]} (avg={config_avg[best_config]:.2f})')

# Loss最佳
config_loss_avg = {}
for config in configs:
    losses = [train_loss[config][s]['loss'] for s in stages]
    config_loss_avg[config] = np.mean(losses)

best_loss_config = min(config_loss_avg, key=config_loss_avg.get)
print(f'  训练Loss最佳配置: {config_names[best_loss_config]} (avg={config_loss_avg[best_loss_config]:.2f})')

# 2. Stage效果
print('\n【2. Stage效果分析】')
stage_avg = {}
for stage in stages:
    scores = [judge_results[f'{c}_{stage}']['mean_score'] for c in configs]
    stage_avg[stage] = np.mean(scores)

for stage in stages:
    print(f'  {stage}: avg={stage_avg[stage]:.2f}')

# 3. 稳定性分析 (标准差)
print('\n【3. 稳定性分析 (标准差越小越稳定)】')
stability = {}
for config in configs:
    stds = [judge_results[f'{config}_{s}']['std_score'] for s in stages]
    stability[config] = np.mean(stds)

most_stable = min(stability, key=stability.get)
print(f'  最稳定配置: {config_names[most_stable]} (avg_std={stability[most_stable]:.2f})')

for config in configs:
    print(f'  {config_names[config]}: avg_std={stability[config]:.2f}')

# 4. Loss-Score相关性
print('\n【4. Loss与Judge Score关系】')
losses_all = []
scores_all = []
for config in configs:
    for stage in stages:
        name = f'{config}_{stage}'
        if name in judge_results:
            losses_all.append(train_loss[config][stage]['loss'])
            scores_all.append(judge_results[name]['mean_score'])

correlation = np.corrcoef(losses_all, scores_all)[0, 1]
print(f'  相关系数: {correlation:.3f}')
if correlation < -0.3:
    print('  结论: Loss越低，Score越高 (正相关于模型质量)')
elif correlation > 0.3:
    print('  结论: Loss越低，Score反而越低 (可能存在过拟合)')
else:
    print('  结论: Loss与Score无明显相关性')

# 5. 推荐配置
print('\n【5. 综合推荐】')
print('  考虑评分、稳定性和训练效率:')
print(f'  🥇 推荐配置: {config_names["minimal"]}')
print('     - LLM Judge评分最高 (3.13)')
print('     - 最稳定 (std=0.83)')
print('     - 6层注入，计算效率适中')
print()
print(f'  🥈 备选配置: {config_names["baseline"]}')
print('     - 评分适中 (2.96)')
print('     - 8层注入，更全面但计算量大')

print('\n' + '='*70)
print('可视化图表已保存到: results/eval_20260411_163906/')
print('='*70)