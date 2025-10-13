#CUDA_VISIBLE_DEVICES=0 python RKD-test.py
import json
import re
import pandas as pd
import torch
from openai import OpenAI 
from datetime import datetime
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
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

Critic_model_name = "./Qwen/Qwen2___5-7B-Instruct"
Generator_model_name = "./Qwen/Qwen2___5-Math-7B-Instruct"
device = "cuda" # the device to load the model onto
Critic_model = AutoModelForCausalLM.from_pretrained(
    Critic_model_name,
    torch_dtype="auto",
    device_map="auto"
)
Critic_tokenizer = AutoTokenizer.from_pretrained(Critic_model_name)
Generator_model = AutoModelForCausalLM.from_pretrained(
    Generator_model_name,
    torch_dtype="auto",
    device_map="auto"
)
Generator_tokenizer = AutoTokenizer.from_pretrained(Generator_model_name)

def is_correct_answer(pred_answer, gt_answer):
    #验证预测答案是否正确（容差1e-3）
    return pred_answer is not None and abs(pred_answer - gt_answer) < 0.1

def extract_answer(text):
    """使用正则表达式提取boxed答案（兼容多格式）"""
    match = re.search(r'\\boxed{([^}]+)}', text)
    equation_match = re.findall(r'[$${]\s*.*?\s*=\s*(\d+)\s*[$$}]', text)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    
    if equation_match:
        try:
            return float(equation_match[-1])
        except:
            pass
        
    return None

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
#评估提取
def extract_error(text):
   # 使用正则表达式匹配 is_correct，不区分大小写
    is_correct_match = re.search(r'"is_correct"\s*:\s*(\w+)', text, re.IGNORECASE)
    
    # 匹配多行错误信息
    errors_match = re.search(r'"errors"\s*:\s*\[([^\]]+)\]', text, re.DOTALL)
    
    if is_correct_match:
        # 转换为小写后比较
        is_correct = is_correct_match.group(1).lower() == 'true'
        
        if is_correct:
            return None
        else:
            if errors_match:
                # 提取并清理多行错误信息
                error_text = errors_match.group(1)
                # 使用正则表达式提取每个错误字符串，去除引号和多余空白
                errors = re.findall(r'"([^"]*)"', error_text)
                
                # 将多行错误合并
                return '\n'.join(errors)
    
    return None

def generate_cot(prompt):
    """生成思维链（参考网页55的CoT提示结构）"""
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, 
        { "role": "user", "content": prompt}]
    
    # 生成参数优化（参考网页68的生成配置）
    inputs = Generator_tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(device)
    
    outputs = Generator_model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.3, 
        pad_token_id=Generator_tokenizer.eos_token_id
    )
    
    return Generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_cot(cot_text,question):
    """评价思维链（基于网页55的CoT验证方法）"""
    num_prompt = """Only analyze the numerical calculation in reasoning process.Just Check whether each calculation and result is correct.
    
 for example, 
    question:Xiao Ming has 2 apples, Xiao Hong has 8 bananas, Xiao Ming gets 3 more apples, how many fruits does Xiao Ming have in total?
    reasoning process:there are 2 apples,there are 8 bananas,so there are 2+8+3=13 fruits, the answer is 13
    output:{{
    "is_correct": false,
    "errors": ["The error in the reasoning step is that the bananas belonging to Xiaohong are mistakenly counted into Xiaoming's total number of fruits, which leads to confusion in the calculation results and the addition of irrelevant items."]
}}
for right Semantic Logic,output:{{
    "is_correct":ture
}}  
"""
    sem_prompt = """Only analyze the Semantic Logic in reasoning process.1. Confirm whether the reasoning steps are in accordance with the requirements of the question.2. Whether the meaning of the symbol is misunderstood.
    Please return the analysis results and JSON format, including the following fields:
{{
    "is_correct": Boolean value,
    "errors": List of strings
}}
    """
    
    # 构造评价请求（参考网页44的模型调用方式）
    inputs = Critic_tokenizer.apply_chat_template(
        [
        { "role": "user", "content": "question:"+question+"Reasoning process to be analyzed:"+cot_text},
            {"role": "user", "content": sem_prompt}
        ],
        return_tensors="pt"
    ).to(device)
    
    outputs = Critic_model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        pad_token_id=Critic_tokenizer.eos_token_id
    )
    
    return Critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_with_feedback(prompt, feedback=None):
    """带反馈的生成函数（支持迭代生成）"""
    system_msg = "Resolve the math problem based on the following error analysis" 
    
    messages = [
        {"role": "system","content": system_msg + " ,put your final answer within \\boxed{}."},
        {"role": "user","content":  "\nquestion:"+prompt+"\nerror analysis:"+feedback}]
    
    inputs = Generator_tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(device)
    
    outputs = Generator_model.generate(
        inputs,
        max_new_tokens=1024,
        pad_token_id=Generator_tokenizer.eos_token_id,
        #repetition_penalty=1.1,  # 抑制重复生成
        do_sample=True,
        temperature=0.3
        #top_k=50
    )
    
    return Generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_math_dataset(dataset_path, log_file, error_json_path):
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    errors = []
    print("共"+str(total)+"条测试数据")
    correct = 0
    with open(log_file, "w") as log:
        global_start = datetime.now()
        log.write(f"◆ 全局开始时间: {global_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n\n")
        
        for idx, item in enumerate(dataset, 1):
            # 生成思维链（参考网页46的POT方法）
            question_start = datetime.now()

            p = item["Question"]
            gt_answer = float(item["Answer"])
            
            initial_cot = generate_cot(p)
            #print("\n初始推理：\n", initial_cot)
            pred_answer = extract_answer(initial_cot)
            cot = initial_cot
            if is_correct_answer(pred_answer, gt_answer):
                correct += 1
            else:
                evaluation_result = evaluate_cot(initial_cot,p)
                #print("\n评估推理：\n",evaluation_result)
                # 带反馈重新生成
                feedback = extract_error(evaluation_result)
                if feedback:
                    revised_cot = generate_with_feedback(p, feedback)
                    pred_answer = extract_answer(revised_cot)
                    cot = revised_cot
                    if is_correct_answer(pred_answer, gt_answer):
                        correct += 1
                else:
                    errors.append({
                    "question": p,
                    "correct_answer": gt_answer,
                    "pred_answer": pred_answer,
                    "feedback":feedback,
                    "cot": cot
                })
                #print("\n重新推理：\n",revised_cot)
            
        # 答案验证（参考网页17的交叉验证机制）
            
            
        # 打印详细结果
            log.write(f"▶ 问题 {idx}/{total} | 开始于: {question_start.strftime('%H:%M:%S.%f')[:-3]}\n")
            log.write(f"问题：{item['Question']}\n")
            log.write(f"生成解答：{cot}\n")
            log.write(f"预测答案：{pred_answer} | 正确答案：{gt_answer}{'✅' if pred_answer == gt_answer else '❌'}\n")
            question_end = datetime.now()
            duration = (question_end - question_start).total_seconds()
            log.write(f"◈ 处理耗时: {duration:.3f}s | 结束于: {question_end.strftime('%H:%M:%S.%f')[:-3]}\n")
            log.write("="*80 + "\n\n")

        with open(error_json_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=4, ensure_ascii=False)

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
    dataset_path = "./datasets/svamp/test.json"
    log_file="./output/svamp/r-result.txt"
    error_json_path = "./output/svamp/r-error-result.json"
    accuracy = evaluate_math_dataset(dataset_path, log_file, error_json_path)
    print(f"\n模型在{len(load_dataset(dataset_path))}个样本上的准确率：{accuracy:.2%}")
    dataset_path = "./datasets/mathqa/test.json"
    log_file="./output/mathqa/r-result.txt"
    error_json_path = "./output/mathqa/r-error-result.json"
    accuracy = evaluate_math_dataset(dataset_path, log_file, error_json_path)
    print(f"\n模型在{len(load_dataset(dataset_path))}个样本上的准确率：{accuracy:.2%}")