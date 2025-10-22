# EFGD

## 
Source code for "Evaluation Feedback-Guided Distillation: Enhancing Math Word Problem Solvers with Large Language Models".

Please note that the currently available code has not undergone a thorough refinement process. We intend to upload a more polished and robust version of the code after the paper's publication.

## Project Structure

### Diverse-Data-Generation
This module is mainly responsible for generating diversified variants of the original MWP questions for data augmentation, and dynamically evaluating the learning status of the student model to generate targeted training examples for weak points.


### Reflective-Dual-Model-Collaborative-Reasoning
The framework aims to identify and correct numerical calculation and logical errors that occur in the reasoning process of the thought chain through interactive verification between the generation model and the evaluation model.

## Data
There are three datasets used in this work, MathQA, MAWPS, SVAMP.

## Environment Configuration

### Dependency Installation
```bash
pip install -r Diverse-Data-Generation/requirements.txt
```

The main dependencies include:
- transformers==4.22.2
- torch
- sympy==1.4
- pandas
- nltk==3.6.6

## Instructions

### Data Generation
```bash
# MathQA
./mathqa.sh

# MAWPS
./mawps.sh

# SVAM
./svamp.sh
```

### Model training
```bash
# MathQA
python train_aug_mathqa.py -gpu 0 -generation -run_name mathqa_gts_genei

# MAWPS
python train_aug_mawps.py -freeze_emb -full_cv -gpu 0 -aug_size 5 -run_name mawps_gts_aug_b

# SVAMP
python train_aug_svamp.py -gpu 1 -epochs 40 -generation -emb_name "roberta-large" -embedding_size 1024 -aug_size 5 -run_name svamp_large_aug_b
```


## Notes
1. Ensure sufficient GPU memory (≥16GB recommended)
2. If you encounter CUDA version issues, run `unset LD_LIBRARY_PATH`
3. The model file path needs to be adjusted based on your needs.



