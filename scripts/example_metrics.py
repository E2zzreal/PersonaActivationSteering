"""
ALOE指标实际计算示例
基于训练前后的真实对比
"""

import numpy as np

# ==========================================
# 假设的训练前后数据
# ==========================================

# 未训练模型（baseline）的各轮对齐分数
baseline_scores = [
    [2.1, 2.3, 2.0, 2.2, 2.4, 2.1, 2.3, 2.2, 2.0, 2.1],  # 样本1
    [2.5, 2.4, 2.6, 2.3, 2.5, 2.4, 2.6, 2.5, 2.4, 2.3],  # 样本2
    [1.8, 2.0, 1.9, 2.1, 2.0, 2.2, 2.1, 2.3, 2.2, 2.1],  # 样本3
]

# 训练后模型的各轮对齐分数
trained_scores = [
    [3.8, 4.0, 4.2, 4.3, 4.5, 4.4, 4.6, 4.5, 4.7, 4.6],  # 样本1
    [4.0, 4.1, 4.3, 4.5, 4.4, 4.6, 4.7, 4.6, 4.8, 4.7],  # 样本2
    [3.5, 3.7, 3.9, 4.0, 4.2, 4.1, 4.3, 4.4, 4.3, 4.5],  # 样本3
]

# 人工标注的"理想"分数
ground_truth_scores = [
    [4.0, 4.2, 4.3, 4.5, 4.6, 4.5, 4.7, 4.6, 4.8, 4.7],
    [4.1, 4.2, 4.4, 4.6, 4.5, 4.7, 4.8, 4.7, 4.9, 4.8],
    [3.6, 3.8, 4.0, 4.1, 4.3, 4.2, 4.4, 4.5, 4.4, 4.6],
]

# ==========================================
# 计算三个指标
# ==========================================

def compute_metrics(baseline, trained, ground_truth):
    """计算所有指标"""
    
    results = {
        "baseline": {},
        "trained": {},
        "comparison": {}
    }
    
    # 1. AL(K)_AVG
    # Baseline
    baseline_avg = np.mean([np.mean(scores) for scores in baseline])
    results["baseline"]["al_k_avg"] = baseline_avg
    
    # Trained
    trained_avg = np.mean([np.mean(scores) for scores in trained])
    results["trained"]["al_k_avg"] = trained_avg
    
    # 2. N-IR (改进率)
    n_ir = (baseline_avg - trained_avg) / baseline_avg
    # 注意：这个公式中，分数越高越好，所以用 baseline - trained
    # 但对于loss，应该用 (baseline - current) / baseline
    # 这里我们希望分数提高，所以应该是 (trained - baseline) / baseline
    n_ir_corrected = (trained_avg - baseline_avg) / (5.0 - baseline_avg)  # 归一化到满分5
    results["comparison"]["n_ir"] = n_ir_corrected
    
    # 3. N-R² (与人工标注的相关性)
    # 展平所有分数
    trained_flat = np.array(trained).flatten()
    ground_truth_flat = np.array(ground_truth).flatten()
    
    # 计算相关系数
    correlation = np.corrcoef(trained_flat, ground_truth_flat)[0, 1]
    r_squared = correlation ** 2
    results["trained"]["n_r2"] = r_squared
    
    # Baseline的R²
    baseline_flat = np.array(baseline).flatten()
    baseline_corr = np.corrcoef(baseline_flat, ground_truth_flat)[0, 1]
    baseline_r2 = baseline_corr ** 2
    results["baseline"]["n_r2"] = baseline_r2
    
    return results

# 计算指标
results = compute_metrics(baseline_scores, trained_scores, ground_truth_scores)

# 打印结果
print("=" * 60)
print("ALOE 指标计算结果")
print("=" * 60)

print("\n【1. AL(K)_AVG - 平均对齐分数】")
print(f"  Baseline (训练前): {results['baseline']['al_k_avg']:.3f}")
print(f"  Trained (训练后):  {results['trained']['al_k_avg']:.3f}")
print(f"  提升: +{results['trained']['al_k_avg'] - results['baseline']['al_k_avg']:.3f}")

print("\n【2. N-IR - 归一化改进比率】")
print(f"  N-IR: {results['comparison']['n_ir']:.3f}")
print(f"  解释: 相对于满分，改进了 {results['comparison']['n_ir']*100:.1f}%")

print("\n【3. N-R² - 归一化决定系数】")
print(f"  Baseline R²: {results['baseline']['n_r2']:.3f}")
print(f"  Trained R²:  {results['trained']['n_r2']:.3f}")
print(f"  提升幅度: +{(results['trained']['n_r2'] - results['baseline']['n_r2'])*100:.1f}%")
print(f"  解释: 与人工标注的一致性达到 {results['trained']['n_r2']*100:.1f}%")

print("\n" + "=" * 60)
print("【综合评估】")
print("=" * 60)

# 单轮渐进改进分析
print("\n【多轮对话渐进改进】")
for i, (b, t) in enumerate(zip(baseline_scores[0], trained_scores[0])):
    improvement = t - b
    print(f"  轮次{i+1}: {b:.1f} → {t:.1f} (提升 +{improvement:.1f})")

# 计算平均每轮改进
avg_improvement_per_turn = np.mean([
    np.mean([t - b for t, b in zip(trained_scores[i], baseline_scores[i])])
    for i in range(len(baseline_scores))
])
print(f"\n  平均每轮改进: +{avg_improvement_per_turn:.2f}")
