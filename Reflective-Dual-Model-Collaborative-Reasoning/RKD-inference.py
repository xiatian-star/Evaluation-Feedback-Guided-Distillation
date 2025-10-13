#RKD-inference.py
#CUDA_VISIBLE_DEVICES=0 python RKD-inference.py
import json
import pandas as pd
import torch

from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
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
#model_dir = snapshot_download("Qwen/Qwen2.5-Math-7B-Instruct", cache_dir="./", revision="master")
#model_dir = snapshot_download("Qwen/Qwen2.5-Math-7B", cache_dir="./", revision="master")
#transformers.utils.OFFLINE_MODE = True
# Transformers加载模型权重

#tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2___5-Math-7B-Instruct", use_fast=False, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen2___5-Math-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)

#model_name = "./Qwen/Qwen2___5-Math-7B-Instruct"
model_name = "./Qwen/Qwen2___5-Math-7B"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

# CoT
messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

# TIR
'''
messages = [
    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]
'''

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)