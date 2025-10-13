#RKD-inference.py
#CUDA_VISIBLE_DEVICES=0 python RKD-inference.py
import json
import re
import pandas as pd
import torch
from openai import OpenAI 
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

# 在modelscope上下载Qwen模型到本地目录下
#model_dir = snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir="./", revision="master")
#model_dir = snapshot_download("Qwen/Qwen2.5-Math-7B", cache_dir="./", revision="master")
#transformers.utils.OFFLINE_MODE = True
# Transformers加载模型权重

#tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2___5-Math-7B-Instruct", use_fast=False, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen2___5-Math-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
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
'''
 for example, 
    question:Xiao Ming has 2 apples, Xiao Hong has 8 bananas, Xiao Ming gets 3 more apples, how many fruits does Xiao Ming have in total?
for error reasoning process:Xiao Ming has 2 apples,Xiao Ming gets 3 more apples,so there are 2+3=6 fruits, the answer is 6
    output:{{
    "is_correct": false,
    "errors": ["The error is a numerical calculation error,2+3=5"]
}}
for right numerical calculation,output:{{
    "is_correct":ture
}}
'''
def extract_answer(text):
    """使用正则表达式提取boxed答案（兼容多格式）"""
    match = re.search(r'\\boxed{([^}]+)}', text)
    return match.group(1) if match else None

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
        max_new_tokens=512,
        temperature=0.3, 
        pad_token_id=Generator_tokenizer.eos_token_id
    )
    
    return Generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_cot_1(cot_text, question):
    prompt = """
    Please analyze reasoning process from the following perspectives step by step:
1."Logical error": Check whether the reasoning between steps is coherent and whether it violates mathematical axioms,
2."Numerical error": Verify whether the calculation process and final result are correct,
3."Semantic error": Confirm whether the problem requirements or symbol meanings are misunderstood
Please return the analysis results and JSON format, including the following fields:
{{
    "is_correct": Boolean value,
    "errors": List of strings
}}
Example JSON:{{
    "is_correct": false,
    "errors": ["Missing step: The specific process of moving terms is not clearly stated,Final result error: The correct result should be x = -1, but it was not solved correctly,Symbol meaning unclear: The meaning of x is not explicitly defined"]
}}
else Example JSON:{{
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
        model='glm-4-Plus',  # Determines the quality, speed, and cost.
        temperature=0.1,            # Level of creativity in the response
        #prompt=full_prompt,           # What the user typed in 
        messages=[
            {"role": "system", "content": "You are an expert in math problems."},
        {"role": "user", "content": '\nQuestion: '+ question + '\nReasoning process to be analyzed: ' + cot_text+prompt }
        ],
        max_tokens=512,             # Maximum tokens in the prompt AND response
        #n=n_generation,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )

    
    return completion.choices[0].message.content
    

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
        pad_token_id=Critic_tokenizer.eos_token_id
    )
    
    return Critic_tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_structured_feedback(eval_result):
    """将评价结果转换为自然语言反馈"""
    try:
        result_json = json.loads(re.search(r'{.*}', eval_result, re.DOTALL).group())
        feedback_lines = []
        
        if result_json["logical_errors"]:
            feedback_lines.append("逻辑错误：" + "；".join(result_json["logical_errors"]))
        if result_json["numerical_errors"]:
            feedback_lines.append("数值错误：" + "；".join(result_json["numerical_errors"]))
        if result_json["semantic_errors"]:
            feedback_lines.append("语义错误：" + "；".join(result_json["semantic_errors"]))
            
        return "\n".join(feedback_lines)
    except Exception as e:
        return "无法解析错误细节，请检查推理过程是否符合数学规范"

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
        max_new_tokens=512,
        pad_token_id=Generator_tokenizer.eos_token_id
        #repetition_penalty=1.1,  # 抑制重复生成
        #do_sample=True,
        #top_k=50
    )
    
    return Generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

#主执行流程
def math_solver(prompt, max_retry=1):
    print(f"\n问题：{prompt}")
    
    # 初始生成
    initial_cot = generate_with_feedback(prompt)
    print("\n初始推理：\n", initial_cot)
    
    # 初始评价
    evaluation_result = evaluate_cot(initial_cot)
    
    try:
        result_json = json.loads(re.search(r'{.*}', evaluation_result, re.DOTALL).group())
        if result_json["is_correct"]:
            answer = extract_answer(initial_cot)
            return f"最终答案：{answer}（验证通过）"
    except:
        pass
    
    # 错误处理流程
    for attempt in range(max_retry):
        feedback = get_structured_feedback(evaluation_result)
        print(f"\n第{attempt+1}次错误反馈：\n{feedback}")
        
        # 带反馈重新生成
        revised_cot = generate_with_feedback(prompt, feedback)
        print(f"\n第{attempt+1}次修正推理：\n{revised_cot}")
        
        # 验证修正结果
        evaluation_result = evaluate_cot(revised_cot)
        try:
            result_json = json.loads(re.search(r'{.*}', evaluation_result, re.DOTALL).group())
            if result_json["is_correct"]:
                answer = extract_answer(revised_cot)
                return f"最终答案：{answer}（经{attempt+1}次修正后验证通过）"
        except:
            continue
            
    return "错误无法修正，建议人工检查"

# 测试用例
"""
Resolve the math problem based on the following error analysis:is_correct: false,logical_errors:  The reasoning incorrectly includes the number of kids Julia played cards with on Wednesday, which is irrelevant to the question about tag.,numerical_errors: The final result is incorrect. The correct total number of kids Julia played tag with is 20, not 40.,semantic_errors: Misinterpretation of the problem requirements: The question specifically asks for the number of kids Julia played tag with, but the reasoning includes kids from a cards activity."]
feedback="Error analysis:is_correct: false,logical_errors:  The reasoning incorrectly includes the number of kids Julia played cards with on Wednesday, which is irrelevant to the question about tag.,numerical_errors: The final result is incorrect. The correct total number of kids Julia played tag with is 20, not 40.,semantic_errors: Misinterpretation of the problem requirements: The question specifically asks for the number of kids Julia played tag with, but the reasoning includes kids from a cards activity."
 initial_cot="To determine the total number of kids Julia played tag with, we need to add the number of kids she played with on each day. Here are the steps:1. Identify the number of kids Julia played with on Monday: 7 kids.2. Identify the number of kids Julia played with on Tuesday: 13 kids.3. Add the number of kids from Monday and Tuesday: \(7 + 13 = 20\).4. Identify the number of kids Julia played with on Wednesday: 20 kids.5. Add the number of kids from Wednesday to the total from Monday and Tuesday: \(20 + 20 = 40\).Therefore, the total number of kids Julia played tag with altogether is \(\boxed{40}\)."

feedback="is_correct false,errors:Logical error: The number of kids on Wednesday is incorrectly stated as '22' instead of '20', which is should be twenty.Numerical error: The sum 22 + 22 equals 44, but it should be 20 + 20 = 15.,Semantic error: The meaning of '22' is unclear, should be clearly defined as '20' (twenty)."   

"""

test_prompts = [
    "question：julia played tag with 7 kids on monday and 13 kids on tuesday.she played cards wtih 20 kids on wednesday . how many kids did she play tag with altogether ?"]
for p in test_prompts:
    initial_cot = generate_cot(p)
    print("\n初始推理：\n", initial_cot)
    evaluation_result = evaluate_cot(initial_cot,p)
    print("\n评估推理：\n",evaluation_result)
    # 带反馈重新生成
    feedback = extract_error(evaluation_result)
    revised_cot = generate_with_feedback(p, feedback)
    print("\n重新推理：\n",revised_cot)



