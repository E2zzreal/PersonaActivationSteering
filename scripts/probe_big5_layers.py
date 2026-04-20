#!/usr/bin/env python
"""Big Five Personality Probing Experiment

对冻结的 Qwen3-4B backbone 做线性探针分析：
- 每层 hidden state → 线性探针 → 预测 [O,C,E,A,N] 分数
- 找出哪些层编码人格信息最强
- 指导最优注入层选择

用法：
  python scripts/probe_big5_layers.py --device cuda:0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def collect_hidden_states(
    model, tokenizer, texts: list[str], device: str
) -> dict[int, np.ndarray]:
    """收集每层的 hidden states（mean pooling）

    Returns:
        {layer_idx: (num_texts, hidden_dim)} numpy arrays
    """
    encoder = model.model  # backbone.model = transformer without LM head
    num_layers = model.config.num_hidden_layers

    # 注册 hooks 收集每层输出
    hidden_states_by_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    hooks = []
    for layer_idx in range(num_layers):
        layer = encoder.layers[layer_idx]

        def make_hook(idx):
            def hook_fn(module, input, output):
                # output 可能是 tuple
                h = output[0] if isinstance(output, tuple) else output
                # mean pooling over sequence
                hidden_states_by_layer[idx].append(h.float().mean(dim=1).detach().cpu())
            return hook_fn

        hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

    # 逐条推理（避免 OOM）
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            encoder(**inputs)

    # 清理 hooks
    for h in hooks:
        h.remove()

    # 整理为 numpy
    result = {}
    for layer_idx, tensors in hidden_states_by_layer.items():
        result[layer_idx] = torch.cat(tensors, dim=0).numpy()

    return result


def run_probing(
    hidden_states: dict[int, np.ndarray],
    targets: np.ndarray,
    dim_names: list[str] = None,
) -> dict:
    """对每层、每个 Big Five 维度训练线性探针

    Args:
        hidden_states: {layer_idx: (n_samples, hidden_dim)}
        targets: (n_samples, 5) Big Five scores
        dim_names: ['O', 'C', 'E', 'A', 'N']

    Returns:
        {layer_idx: {dim_name: r2_score, 'overall': mean_r2}}
    """
    if dim_names is None:
        dim_names = ['O', 'C', 'E', 'A', 'N']

    results = {}
    for layer_idx, X in sorted(hidden_states.items()):
        layer_result = {}
        r2s = []
        for d, name in enumerate(dim_names):
            y = targets[:, d]
            # 如果 y 方差为 0，跳过
            if np.std(y) < 1e-6:
                layer_result[name] = 0.0
                continue
            # Leave-one-out cross validation with Ridge
            model = Ridge(alpha=1.0)
            y_pred = cross_val_predict(model, X, y, cv=min(5, len(y)))
            r2 = r2_score(y, y_pred)
            layer_result[name] = round(r2, 4)
            r2s.append(r2)
        layer_result['overall'] = round(np.mean(r2s), 4) if r2s else 0.0
        results[layer_idx] = layer_result
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--big5_config', default='configs/big5_personalities.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='results/big5_probing/layer_probe.json')
    # 数据增强：每个 personality 用多条用户输入生成不同 context
    parser.add_argument('--user_inputs_path',
                        default='/home/kemove/Desktop/PersonaSteer/results/parallel_dialogues/dialogues.json')
    parser.add_argument('--n_contexts', type=int, default=5,
                        help='每个 personality 用几条不同的用户输入作为 context')
    args = parser.parse_args()

    # 加载 Big Five 配置
    big5_cfg = json.loads(Path(args.big5_config).read_text())
    personas = big5_cfg['personalities']
    print(f'{len(personas)} Big Five personalities')

    # 构建探测文本：personality description + user context
    # 为什么需要 context：personality 信息在不同 context 下的 hidden state 位置可能不同
    user_inputs = []
    try:
        dialogues = json.loads(Path(args.user_inputs_path).read_text())
        user_inputs = [d['turns'][0]['user_input'] for d in dialogues[:args.n_contexts]]
    except Exception:
        user_inputs = ["Hello, how are you doing today?"] * args.n_contexts

    texts = []
    scores = []
    for persona in personas:
        b5 = persona['big5']
        score = [b5['O'], b5['C'], b5['E'], b5['A'], b5['N']]
        for ui in user_inputs:
            # 模拟实际推理场景：system(personality) + user(input)
            text = f"Personality: {persona['description']}\n\nUser: {ui}"
            texts.append(text)
            scores.append(score)

    targets = np.array(scores)
    print(f'{len(texts)} 个探测样本 ({len(personas)} personalities × {len(user_inputs)} contexts)')

    # 加载模型
    print(f'Loading Qwen3-4B on {args.device}...')
    dev_id = int(args.device.split(':')[1]) if ':' in args.device else 0
    model = AutoModelForCausalLM.from_pretrained(
        '/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B',
        trust_remote_code=True, torch_dtype=torch.float16,
        device_map={"": dev_id})
    tokenizer = AutoTokenizer.from_pretrained(
        '/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B', trust_remote_code=True)
    num_layers = model.config.num_hidden_layers
    print(f'{num_layers} layers, collecting hidden states...')

    # 收集 hidden states
    hidden_states = collect_hidden_states(model, tokenizer, texts, args.device)

    # 释放显存
    del model
    torch.cuda.empty_cache()

    # 线性探针
    print('Training linear probes...')
    results = run_probing(hidden_states, targets)

    # 输出结果
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, 'w'), indent=2)

    # 打印热力图
    print(f'\n{"Layer":>6s}  {"O":>7s}  {"C":>7s}  {"E":>7s}  {"A":>7s}  {"N":>7s}  {"Overall":>8s}')
    print('-' * 58)
    best_layer, best_r2 = -1, -1
    for layer_idx in range(num_layers):
        r = results[str(layer_idx)] if str(layer_idx) in results else results.get(layer_idx, {})
        overall = r.get('overall', 0)
        bar = '█' * int(max(0, overall) * 20)
        print(f'{layer_idx:>6d}  {r.get("O",0):>7.3f}  {r.get("C",0):>7.3f}  '
              f'{r.get("E",0):>7.3f}  {r.get("A",0):>7.3f}  {r.get("N",0):>7.3f}  '
              f'{overall:>7.3f}  {bar}')
        if overall > best_r2:
            best_r2, best_layer = overall, layer_idx

    print(f'\n最佳層: {best_layer} (R²={best_r2:.4f})')
    print(f'推荐注入层: 以 layer {best_layer} 为中心的窗口')
    print(f'输出: {args.output}')


if __name__ == '__main__':
    main()
