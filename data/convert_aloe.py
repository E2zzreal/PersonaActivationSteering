#!/usr/bin/env python3
"""ALOE数据转换为PersonaSteer V2格式"""

import json
from pathlib import Path

def convert_aloe():
    raw_dir = Path("aloe_raw/datasets")
    processed_dir = Path("processed")
    processed_dir.mkdir(exist_ok=True)
    
    # 读取原始数据
    train_samples = []
    eval_samples = []
    
    with open(raw_dir / "conversations.jsonl", "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            
            # 转换对话格式
            conversations = []
            for conv in data.get("conversations", []):
                user_msg = conv.get("user", "")
                # 使用chosen对应的回复
                chosen = conv.get("chosen", "preferred")
                asst = conv.get("assistant", {})
                asst_msg = asst.get(chosen, asst.get("preferred", ""))
                
                conversations.append({"role": "user", "content": user_msg})
                conversations.append({"role": "assistant", "content": asst_msg})
            
            sample = {
                "user_id": f"u{i:04d}",
                "profile": data.get("profile", "").strip('"').strip(),
                "personality": data.get("personality", "").strip('"').strip(),
                "conversations": conversations
            }
            
            # 90%训练，10%评估
            if i % 10 == 0:
                eval_samples.append(sample)
            else:
                train_samples.append(sample)
    
    # 写入处理后的数据
    with open(processed_dir / "train.jsonl", "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    with open(processed_dir / "eval.jsonl", "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"转换完成: 训练集 {len(train_samples)} 条, 评估集 {len(eval_samples)} 条")

if __name__ == "__main__":
    convert_aloe()