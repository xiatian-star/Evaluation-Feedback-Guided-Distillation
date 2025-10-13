#CUDA_VISIBLE_DEVICES=0 python baseline.py
import json
import pandas as pd
import torch
import re
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

model_name = "./Qwen/Qwen2___5-Math-7B-Instruct"
#model_name = "./Qwen/Qwen2___5-Math-7B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto").eval()

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
def evaluate_cot_1(cot_text, question):
    prompt = """
    Please analyze the mathematical reasoning process from the following perspectives:
1."Logical error": Check whether the reasoning between steps is coherent and whether it violates mathematical axioms,
2."Numerical error": Verify whether the calculation process and final result are correct,
3."Semantic error": Confirm whether the problem requirements or symbol meanings are misunderstood
Please return the analysis results and a JSON format, including the following fields:
{{
    "is_correct": Boolean value,
    "errors": List of strings
}}
JSON output:{{
    "is_correct": false,
    "errors": ["Missing step: The specific process of moving terms is not clearly stated,Final result error: The correct result should be x = -1, but it was not solved correctly,Symbol meaning unclear: The meaning of x is not explicitly defined"]
}}
Example output:{{
    "is_correct": true,
    "errors": []
}}"""
    full_prompt = prompt +  '\nQuestion: '+ question + '\nReasoning process to be analyzed: ' + cot_text 
    #client = ZhipuAI(api_key = 'd62be502aa03f2b4ae5f012ae9d731b6.s2EUmQDf9bvnaNVS')
    client = OpenAI(
        api_key="8a095de28ebaac2ac54beccd063a61a3.EDlbk53lQrVspeML",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    completion = client.chat.completions.create(
        model='glm-4-flash',  # Determines the quality, speed, and cost.
        temperature=0.9,            # Level of creativity in the response
        #prompt=full_prompt,           # What the user typed in 
        messages=[
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, 
        { "role": "user", "content": question}
        ],
        max_tokens=1024,             # Maximum tokens in the prompt AND response
        #n=n_generation,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )

    
    return completion.choices[0].message.content
# 3. 准确率计算流程
def evaluate_math_dataset(dataset_path, log_file,error_json_path):
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    print("共"+str(total)+"条测试数据")
    correct = 0
    errors = []
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
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码并提取答案（参考网页30的JSON处理）
            response = tokenizer.batch_decode(
                [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]
          
            #response = evaluate_cot_1(" ",item["Question"])
        
        # 答案验证（参考网页17的交叉验证机制）
            pred_answer = extract_pred_answer(response)
            gt_answer = float(item["Answer"])
        
            if pred_answer is not None and abs(pred_answer - gt_answer) < 0.1:
                correct += 1
            else:
                errors.append({
                    "question": item["Question"],
                    "correct_answer": gt_answer,
                    "pred_answer": pred_answer,
                    "cot": response
                })
        
        # 打印详细结果
            log.write(f"▶ 问题 {idx}/{total} | 开始于: {question_start.strftime('%H:%M:%S.%f')[:-3]}\n")
            log.write(f"问题：{item['Question']}\n")
            log.write(f"生成解答：{response}\n")
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
    
    
    dataset_path = "./datasets/mawps/test.json"
    log_file="./output/mawps/IN-result.txt"
    error_json_path = "./output/mawps/IN-error-result.json"
    accuracy = evaluate_math_dataset(dataset_path, log_file, error_json_path)
    print(f"\n模型在{len(load_dataset(dataset_path))}个样本上的准确率：{accuracy:.2%}")
    '''
    dataset_path = "./datasets/svamp/test.json"
    log_file="./output/svamp/IN-result.txt"
    error_json_path = "./output/svamp/IN-error-result.json"
    accuracy = evaluate_math_dataset(dataset_path, log_file, error_json_path)
    print(f"\n模型在{len(load_dataset(dataset_path))}个样本上的准确率：{accuracy:.2%}")
    
    '''


