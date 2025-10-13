# README - 数学问题生成与推理项目

## 项目概述
本项目包含两个主要模块：
1. Diverse-Data-Generation：多样化的数学问题数据生成
2. Reflective-Dual-Model-Collaborative-Reasoning：基于双模型协作的数学问题推理

# temp
## 项目结构

### Diverse-Data-Generation
包含多个数学数据集(MathQA, MAWPS, SVAMP)的训练和评估脚本：
- `train_and_evaluate.py`：核心训练和评估代码
- `train_aug_*.py`：不同数据集的增强训练脚本
- `test_*.py`：测试脚本
- `utils/`：工具函数包
  - `augmentation.py`：数据增强相关函数
  - `expressions_transfer.py`：表达式转换工具
  - `pre_data.py`：数据预处理工具
  - `logger.py`：日志工具

### Reflective-Dual-Model-Collaborative-Reasoning
包含基于Qwen模型的推理系统：
- `qwen-*.py`：Qwen模型相关脚本
- `RKD-*.py`：推理知识蒸馏相关代码
- `data_aug.py`：数据增强代码

## 环境配置

### 依赖安装
```bash
pip install -r Diverse-Data-Generation/requirements.txt
```

主要依赖包括：
- transformers==4.22.2
- torch
- sympy==1.4
- pandas
- nltk==3.6.6

## 使用说明

### 数据生成
```bash
# MathQA数据集
./mathqa.sh

# MAWPS数据集
./mawps.sh

# SVAMP数据集
./svamp.sh
```

### 模型训练
```bash
# 训练MathQA模型
python train_aug_mathqa.py -gpu 0 -generation -run_name mathqa_gts_genei

# 训练MAWPS模型
python train_aug_mawps.py -freeze_emb -full_cv -gpu 0 -aug_size 5 -run_name mawps_gts_aug_b

# 训练SVAMP模型
python train_aug_svamp.py -gpu 1 -epochs 40 -generation -emb_name "roberta-large" -embedding_size 1024 -aug_size 5 -run_name svamp_large_aug_b
```

### 模型推理
```bash
# 使用RKD推理
python RKD-inference.py

# 使用baseline推理
python baseline.py
```

## 注意事项
1. 确保GPU内存足够（建议≥16GB）
2. 如遇CUDA版本问题，运行：`unset LD_LIBRARY_PATH`
3. 模型文件路径需根据实际情况调整

## 项目特色
1. 支持多种数学问题数据集
2. 提供数据增强功能
3. 实现双模型协作推理
4. 支持分布式训练

## 许可证
MIT License
