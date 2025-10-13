#CUDA_VISIBLE_DEVICES=0 python qwen-optimized-training.py
import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    prepare_model_for_kbit_training  # 新增
)
from transformers import TrainerCallback

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


# 模型和数据集路径
model_name = "./Qwen/Qwen2___5-Math-7B-Instruct"
dataset_path = "datasets/mawps/cot-train.json"
device = "cuda" if torch.cuda.is_available() else "cpu"



# 数据处理函数
def process_func(example):
    MAX_LENGTH = 512  # 可以根据需要调整
    
    instruction = tokenizer(
        f"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}<|im_end|>\n<|im_start|>user\n{example['Question']}<|im_end|>\n<|im_start|>assistant\n", 
        add_special_tokens=False
    )
    
    response = tokenizer(f"{example['CoT']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 初始化 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=False, 
    trust_remote_code=True
)

# 设置 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

#model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# LoRA 配置 - 针对大显存优化
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,  # 增加 Lora 秩以提高性能
    lora_alpha=64,  # 增加 alpha 值
    lora_dropout=0.1
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
# 加载数据集

# 数据集处理
train_df = pd.read_json(dataset_path)
train_ds = Dataset.from_pandas(train_df)
tokenized_id = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 针对 32G 显存优化的训练参数
training_args = TrainingArguments(
    output_dir="./output/mawps/Qwen2.5_in_lora",
    per_device_train_batch_size=4,  # 对于32G显存可以增加到8
    gradient_accumulation_steps=4,  # 减少梯度累积步骤
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,  # 添加权重衰减
    save_on_each_node=True,
    gradient_checkpointing=True,
    fp16=True,  # 混合精度训练
    logging_dir='./logs',
    report_to="none"
)

# 数据收集器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    padding=True,
    return_tensors="pt"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 仅保存 LoRA 权重
trainer.model.save_pretrained("./output/mawps/Qwen2.5_in_lora/lora_weights")
tokenizer.save_pretrained("./output/mawps/Qwen2.5_in_lora/tokenizer")

print("训练完成，LoRA 权重已保存")

