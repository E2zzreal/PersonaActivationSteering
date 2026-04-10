#!/bin/bash
# Stage 3 Checkpoints 完整评估流程
# 1. 生成对话
# 2. LLM Judge评估

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/eval_stage3_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Stage 3 checkpoints列表
CHECKPOINTS=(
    "stage3_auto:checkpoints/stage3_auto/best.pt"
    "stage3_gate_init_0:checkpoints/stage3_gate_init_0/best.pt"
    "stage3_gate_reg_0.01_lr1e4:checkpoints/stage3_gate_reg_0.01_lr1e4/best.pt"
    "stage3_gate_reg_0.05_lr5e5:checkpoints/stage3_gate_reg_0.05_lr5e5/best.pt"
)

# 步骤1: 生成对话
generate_conversations() {
    log "=== 步骤1: 生成对话 ==="
    
    cat > /tmp/gen_conversations.py << 'GENEOF'
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm

project_root = Path("/home/kemove/Desktop/PersonaSteer")
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModel

def load_model(checkpoint_path, device="cuda:0"):
    base_model_path = "/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B"
    
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    persona_config = PersonaSteerConfig(
        inject_layers=[8, 9, 10, 11, 12, 13, 14, 15],
        v_dim=1024, hidden_dim=4096,
        layer_dim=backbone_config.hidden_size, gate_hidden_dim=256,
    )

    encoder = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    model = PersonaSteerModel(config=persona_config, encoder=encoder)
    if model.hyper_network is not None:
        model.hyper_network.v_norm_clip = 10.0
        model.hyper_network._tokenizer = tokenizer

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.set_backbone(backbone)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)

    model.backbone.to(device)
    model.hyper_network.to(device)
    model.injection.to(device)
    model.eval()
    
    return model, tokenizer

def generate_conversations(checkpoint_path, output_path, num_samples=50):
    device = "cuda:0"
    model, tokenizer = load_model(checkpoint_path, device)
    
    with open("data/split/val.jsonl") as f:
        data = [json.loads(line) for line in f][:num_samples]
    
    results = []
    
    for sample in tqdm(data, desc="生成对话"):
        personality = sample.get("personality", "")
        conversations = sample.get("conversations", [])
        
        generated_turns = []
        
        for i in range(min(4, len(conversations)//2)):
            user_msg = conversations[i*2].get("content", "") if i*2 < len(conversations) else None
            if not user_msg:
                break
            
            messages = [{"role": "user", "content": user_msg}]
            result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            inputs = result.to(device) if not hasattr(result, 'input_ids') else result.input_ids.to(device)
            
            with torch.no_grad():
                v_prev = torch.zeros(1, 1024, dtype=torch.float32, device=device)
                outputs, _ = model.generate(
                    input_ids=inputs, v_prev=v_prev,
                    personality_texts=[personality],
                    user_query_texts=[user_msg],
                    max_new_tokens=80, temperature=0.7, top_p=0.9,
                )
                new_tokens = outputs[0][inputs.shape[-1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            generated_turns.append({
                "turn": i + 1,
                "user": user_msg,
                "assistant": response
            })
        
        results.append({
            "user_id": sample.get("user_id"),
            "personality": personality,
            "generated_conversations": generated_turns
        })
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"已生成 {len(results)} 个样本，保存到 {output_path}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    
    generate_conversations(args.checkpoint, args.output, args.num_samples)
GENEOF

    for entry in "${CHECKPOINTS[@]}"; do
        IFS=':' read -r name ckpt_path <<< "$entry"
        output_file="results/conversations_${name}_${TIMESTAMP}.json"
        
        log "生成对话: $name"
        CUDA_VISIBLE_DEVICES=0 python /tmp/gen_conversations.py \
            --checkpoint "$ckpt_path" \
            --output "$output_file" \
            --num_samples 50
        
        log "完成: $output_file"
    done
    
    log "所有对话生成完成!"
}

# 步骤2: LLM Judge评估
run_llm_judge() {
    log "=== 步骤2: LLM Judge评估 ==="
    
    cat > /tmp/llm_judge_eval.py << 'JUDGEEOF'
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

project_root = Path("/home/kemove/Desktop/PersonaSteer")
sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("错误: 需要安装 openai 包")
    sys.exit(1)

class LLMJudge:
    def __init__(self, api_key=None, base_url=None, model="GPT-5.2"):
        self.model = model
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"LLM Judge 初始化成功: {model}")
        else:
            raise ValueError("需要API key")
    
    def evaluate_response(self, response, personality, user_msg):
        prompt = f"""评估回复与人格的一致性。

人格描述: {personality}

用户消息: {user_msg}

助手回复: {response}

评分标准 (1-5分):
1分 = 完全不一致
2分 = 大部分不一致  
3分 = 中立
4分 = 大部分一致
5分 = 完全一致

请只输出一个数字(1-5):"""

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的对话评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=10,
            )
            content = result.choices[0].message.content.strip()
            for char in content:
                if char.isdigit():
                    return float(min(max(int(char), 1), 5))
            return 3.0
        except Exception as e:
            print(f"API错误: {e}")
            return 3.0

def evaluate_conversations(conversations_file, output_file, judge):
    with open(conversations_file) as f:
        data = json.load(f)
    
    results = []
    all_scores = []
    
    for sample in tqdm(data, desc="评估中"):
        personality = sample.get("personality", "")
        conversations = sample.get("generated_conversations", [])
        
        sample_scores = []
        
        for turn in conversations:
            user_msg = turn.get("user", "")
            response = turn.get("assistant", "")
            
            if response and personality:
                score = judge.evaluate_response(response, personality, user_msg)
                sample_scores.append(score)
                all_scores.append(score)
        
        results.append({
            "user_id": sample.get("user_id"),
            "scores": sample_scores,
            "avg_score": np.mean(sample_scores) if sample_scores else 0
        })
    
    with open(output_file, 'w') as f:
        json.dump({
            "detailed_results": results,
            "overall_avg": np.mean(all_scores),
            "overall_std": np.std(all_scores),
            "total_turns": len(all_scores)
        }, f, indent=2)
    
    return np.mean(all_scores)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge_model", default="GPT-5.2")
    args = parser.parse_args()
    
    api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BLSC_BASE_URL", "https://llmapi.blsc.cn")
    
    if not api_key:
        print("错误: 未找到API key，请设置 BLSC_API_KEY 环境变量")
        sys.exit(1)
    
    judge = LLMJudge(api_key=api_key, base_url=base_url, model=args.judge_model)
    
    avg_score = evaluate_conversations(args.conversations, args.output, judge)
    print(f"\n平均分数: {avg_score:.2f}")
JUDGEEOF

    for entry in "${CHECKPOINTS[@]}"; do
        IFS=':' read -r name _ <<< "$entry"
        conv_file="results/conversations_${name}_${TIMESTAMP}.json"
        eval_file="results/judge_eval_${name}_${TIMESTAMP}.json"
        
        if [ -f "$conv_file" ]; then
            log "LLM Judge评估: $name"
            python /tmp/llm_judge_eval.py \
                --conversations "$conv_file" \
                --output "$eval_file" \
                --judge_model "GPT-5.2"
            
            log "完成: $eval_file"
        fi
    done
    
    log "所有LLM Judge评估完成!"
}

# 汇总结果
summarize_results() {
    log "=== 汇总评估结果 ==="
    
    python3 << 'SUMMARYEOF'
import json
import glob
from pathlib import Path

print("\n" + "="*70)
print("Stage 3 Checkpoints 评估结果汇总")
print("="*70)

eval_files = sorted(glob.glob("results/judge_eval_stage3_*_*.json"))

if not eval_files:
    print("未找到评估结果文件")
else:
    print(f"\n找到 {len(eval_files)} 个评估结果:")
    
    all_results = {}
    for eval_file in eval_files:
        name = Path(eval_file).stem.replace("judge_eval_", "")
        with open(eval_file) as f:
            data = json.load(f)
        
        avg = data.get("overall_avg", 0)
        std = data.get("overall_std", 0)
        turns = data.get("total_turns", 0)
        
        all_results[name] = {"avg": avg, "std": std, "turns": turns}
        
        print(f"\n{name}:")
        print(f"  平均分: {avg:.2f} ± {std:.2f}")
        print(f"  评估轮数: {turns}")
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["avg"], reverse=True)
    
    print("\n" + "="*70)
    print("排名:")
    print("="*70)
    for i, (name, scores) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {name}: {scores['avg']:.2f}")

SUMMARYEOF
}

# 主流程
main() {
    log "=== Stage 3 Checkpoints 完整评估流程 ==="
    
    generate_conversations
    run_llm_judge
    summarize_results
    
    log "=== 评估流程完成 ==="
}

main "$@"
