#CUDA_VISIBLE_DEVICES=0 python qwen-infer.py
import json
import pandas as pd
import torch
import re
from datetime import datetime
from modelscope import snapshot_download, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import transformers
import os    

from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen2___5-Math-7B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
device = "cuda"
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./output/mawps/Qwen2.5_in_lora/tokenizer")

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./output/mawps/Qwen2.5_in_lora/lora_weights")

# 1. 数据集读取（参考网页26的JSON解析方法）
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 2. 答案提取（参考网页17的正则表达式方法）
def extract_pred_answer(response):
    # 匹配 \boxed{} 格式或自然语言描述的答案
    patterns = [
        r'\\boxed{([\d\.]+)}',
        r'answer (?:is|为)[：:\s]*(\d+\.?\d*)',
        r'最终结果[：:\s]*(\d+\.?\d*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return None

# 3. 准确率计算流程
def evaluate_math_dataset(dataset_path, log_file):
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    print("共"+str(total)+"条测试数据")
    correct = 0
    with open(log_file, "w") as log:
        global_start = datetime.now()
        log.write(f"◆ 全局开始时间: {global_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n\n")
        
        for idx, item in enumerate(dataset, 1):
            # 生成思维链（参考网页46的POT方法）
            question_start = datetime.now()
            
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": item["Question"]}
            ]
        
        # 生成响应（参考网页65的AlphaCodium流程）
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码并提取答案（参考网页30的JSON处理）
            response = tokenizer.batch_decode(
                [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]
        
        # 答案验证（参考网页17的交叉验证机制）
            pred_answer = extract_pred_answer(response)
            gt_answer = float(item["Answer"])
        
            if pred_answer is not None and abs(pred_answer - gt_answer) < 1e-3:
                correct += 1
        
        # 打印详细结果
            log.write(f"▶ 问题 {idx}/{total} | 开始于: {question_start.strftime('%H:%M:%S.%f')[:-3]}\n")
            log.write(f"问题：{item['Question']}\n")
            log.write(f"生成解答：{response}\n")
            log.write(f"预测答案：{pred_answer} | 正确答案：{gt_answer}{'✅' if pred_answer == gt_answer else '❌'}\n")
            question_end = datetime.now()
            duration = (question_end - question_start).total_seconds()
            log.write(f"◈ 处理耗时: {duration:.3f}s | 结束于: {question_end.strftime('%H:%M:%S.%f')[:-3]}\n")
            log.write("="*80 + "\n\n")

        global_end = datetime.now()
        total_duration = (global_end - global_start).total_seconds()
        log.write(f"◆ 全局结束时间: {global_end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
        log.write(f"■ 总处理问题: {total}个\n")
        log.write(f"■ 正确率: {correct/total:.1%}\n")
        log.write(f"■ 总耗时: {total_duration:.2f}秒")
    
    accuracy = correct / total
    return accuracy

# 执行评估
if __name__ == "__main__":
    dataset_path = "./datasets/mawps/test.json"
    log_file="./output/mawps/rkd-result.txt"
    accuracy = evaluate_math_dataset(dataset_path, log_file)
    print(f"\n模型在{len(load_dataset(dataset_path))}个样本上的准确率：{accuracy:.2%}")