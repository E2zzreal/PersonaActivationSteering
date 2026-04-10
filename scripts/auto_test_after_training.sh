#!/bin/bash
# 训练完成后自动运行生成测试

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

LOG_FILE="logs/auto_test_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 等待训练完成
wait_for_training() {
    log "等待Stage 3训练完成..."
    
    while pgrep -f "train.py" > /dev/null; do
        running=$(pgrep -f "train.py" | wc -l)
        log "运行中的训练进程: $running"
        
        # 显示进度
        for log_file in logs/stage3_*.log; do
            if [ -f "$log_file" ]; then
                name=$(basename $log_file .log)
                progress=$(tail -100 "$log_file" 2>/dev/null | grep -oP "Epoch \d+/\d+" | tail -1)
                log "  $name: $progress"
            fi
        done
        
        sleep 300  # 每5分钟检查
    done
    
    log "所有训练已完成!"
}

# 运行生成测试
run_generation_test() {
    log "=== 开始生成测试 ==="
    
    # 查找所有Stage 3 checkpoints
    checkpoints=()
    for ckpt_dir in checkpoints/stage3_*/; do
        if [ -f "${ckpt_dir}best.pt" ]; then
            checkpoints+=("${ckpt_dir}best.pt")
        fi
    done
    
    log "找到 ${#checkpoints[@]} 个Stage 3 checkpoints"
    
    # 创建测试脚本
    cat > /tmp/run_gen_test.py << 'PYEOF'
import sys
import torch
import gc
import json
from pathlib import Path
from datetime import datetime

project_root = Path("/home/kemove/Desktop/PersonaSteer")
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModel

SAMPLES = [
    {"personality": "He is easy-going, articulate, and creative. He enjoys deep conversations.", 
     "user_msg": "Hey! How are you today? I've been thinking about starting a new hobby."},
    {"personality": "She is ambitious, organized, and detail-oriented. She loves reading books.", 
     "user_msg": "What do you think about the importance of time management?"},
    {"personality": "He is independent, empathetic, and reflective. He values meaningful connections.", 
     "user_msg": "How do you feel about making new friends?"},
]

def test_checkpoint(checkpoint_path, device="cuda:0"):
    base_model_path = "/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B"
    
    gc.collect()
    torch.cuda.empty_cache()
    
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
    
    results = []
    
    for i, sample in enumerate(SAMPLES):
        messages = [{"role": "user", "content": sample['user_msg']}]
        result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = result.to(device) if not hasattr(result, 'input_ids') else result.input_ids.to(device)

        with torch.no_grad():
            v_prev = torch.zeros(1, 1024, dtype=torch.float32, device=device)
            outputs, _ = model.generate(
                input_ids=inputs, v_prev=v_prev,
                personality_texts=[sample['personality']],
                user_query_texts=[sample['user_msg']],
                max_new_tokens=80, temperature=0.7, top_p=0.9,
            )
            new_tokens = outputs[0][inputs.shape[-1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # 检查退化
        is_ok = len(response) > 15 and not any(p in response for p in ['\n\n\n\n', '?????', '.....', ',,,,'])
        
        results.append({
            "sample": i + 1,
            "personality": sample['personality'],
            "user_msg": sample['user_msg'],
            "response": response,
            "status": "normal" if is_ok else "degenerate"
        })
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    import glob
    
    print("=" * 70)
    print("PersonaSteer Stage 3 生成质量测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 找所有checkpoint
    checkpoints = sorted(glob.glob("checkpoints/stage3_*/best.pt"))
    
    if not checkpoints:
        print("未找到Stage 3 checkpoints!")
        return
    
    print(f"\n找到 {len(checkpoints)} 个checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")
    
    all_results = {}
    
    for ckpt_path in checkpoints:
        name = Path(ckpt_path).parent.name
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")
        
        try:
            results = test_checkpoint(ckpt_path)
            all_results[name] = results
            
            for r in results:
                print(f"\n--- Sample {r['sample']} ---")
                print(f"Personality: {r['personality'][:50]}...")
                print(f"User: {r['user_msg'][:50]}...")
                print(f"Assistant: {r['response'][:100]}...")
                print(f"状态: {'✓ 正常' if r['status']=='normal' else '⚠️ 退化'}")
                
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    output_file = f"results/stage3_generation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"结果已保存: {output_file}")
    print(f"{'='*70}")
    
    # 统计
    normal_count = sum(1 for results in all_results.values() for r in results if r['status']=='normal')
    total_count = sum(len(results) for results in all_results.values())
    print(f"\n统计: {normal_count}/{total_count} 样本正常 ({normal_count*100/total_count:.1f}%)")

if __name__ == "__main__":
    main()
PYEOF

    # 运行测试
    python /tmp/run_gen_test.py 2>&1 | tee -a "$LOG_FILE"
    
    log "生成测试完成!"
}

# 主流程
main() {
    log "=== 自动测试脚本启动 ==="
    wait_for_training
    sleep 10  # 等待GPU完全释放
    run_generation_test
    log "=== 全部完成 ==="
}

main "$@"
