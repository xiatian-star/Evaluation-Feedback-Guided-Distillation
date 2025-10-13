import re
import json
import json
import re
import pandas as pd
import torch
from openai import OpenAI 

prompt1 = """你是一个专业的数学题目设计师，请根据初始题目生成结构相似但运算逻辑变化的变体问题。请严格按照以下规则执行：

【运算替换规则】
1. 主干分析：识别原题的核心运算序列（如：先加后乘 → 先减后除）
2. 对称转换：将正运算转为逆运算（加减互转、乘除互转、幂与根号互转）
3. 参数适配：调整数值使新题合理（如原题"买3苹果各5元总价？" → 变体"有15元买5元苹果能买几个？"）
4. 场景守恒：保持原题场景类型（如购物问题保持商业场景）

【生成要求】
 1.生成2个不同变体，每个含完整题目和答案
2.检查答案正确性，如果答案不对请更改，最终输出json格式的正确的变体

【输出格式】
变体1：
{题目：...[变体问题描述]...
答案：...[变体答案]...}
...（后续变体同理）..."""

prompt2 = """你是一个数学问题重构专家，请根据原始数学问题生成跨领域同构变体。请严格遵循以下规则：

【场景重构规则】
1. 抽象分析：识别原题的数学操作序列（如：乘法关系→线性增长）
2. 领域迁移：如购物、组合、物理、数学、金融、生态、通信等
3. 现实合理性：验证新场景下数据范围的可行性（如人口增长率不超过5%）
4. 操作守恒：确保所有数学步骤可完全复现

【生成要求】
 1.生成2个不同变体，每个含完整题目和答案
2.检查答案正确性，如果答案不对请更改，最终输出json格式的正确的变体

【输出模板】
变体1：
{题目：...[变体问题描述]...
答案：...[变体答案]...}
...（后续变体同理）...

"""

def extract_final_variants(text: str) -> dict:
    """从文本中提取最后出现的变体1和变体2内容
    
    Args:
        text: 包含变体内容的原始文本
        
    Returns:
        dict: 包含两个变体的字典，结构{
            "变体1": "题目...答案...",
            "变体2": "题目...答案..."
        }
    """
    variants = []
    # 匹配所有变体块（兼容变体1-99）
    variant_blocks = re.findall(r'变体\d+：\s*{([\s\S]*?)\n}', text, re.DOTALL)
    
    # 处理最后两个变体（变体1和变体2）
    for block in variant_blocks[-2:]:
        # 提取题目（匹配到答案行或块结束）
        question = re.search(r'题目：\s*((?:(?!答案：).)*)', block, re.DOTALL)
        # 提取答案（匹配答案行直到块结束）
        answer = re.search(r'答案：\s*(.*?)(?=\n|$)', block, re.DOTALL)
        
        if question and answer:
            variants.append({
                "题目": question.group(1).strip(),
                "答案": answer.group(1).strip()
            })
    
    with open('aug_data.json', 'a+', encoding='utf-8') as f:
        json.dump(variants, f, ensure_ascii=False, indent=4
                  
def gernerator_var(cot_text, question):
    prompt = """You are a professional math question designer. Please generate variant questions with similar structures but different operation logic based on the initial question. Please strictly follow the following rules:

[Operation replacement rules]

1. Trunk analysis: Identify the core operation sequence of the original question (such as: first add then multiply → first subtract then divide)

2. Symmetric transformation: Convert positive operations to inverse operations (addition and subtraction, multiplication and division, power and square root)

3. Parameter adaptation: Adjust the value to make the new question reasonable (such as the original question "Buy 3 apples for 5 yuan each?" → variant "How many 5 yuan apples can I buy with 15 yuan?")

4. Scene conservation: Keep the original scene type (such as shopping problems keep the business scene)

[Generation requirements]
1. Generate 2 different variants, each with a complete question and answer
2. Check the correctness of the answer. If the answer is incorrect, please change it. Finally, output the correct variant in json format

[Output format]
Variant 1:
{Question:...[Variant question description]...
Answer:...[Variant answer]...}
...(The same applies to subsequent variants)..."""
    
    full_prompt = prompt +  '\nQuestion: '+ question 
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