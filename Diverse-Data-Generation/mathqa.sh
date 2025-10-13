#!/bin/bash

#python -u train_aug_mathqa.py -gpu 0 -freeze_emb -generation -aug_size 5 -run_name mathqa_gts_aug_5
#for n in 0 0.5 1

#do
#    python -u train_aug_svamp.py -gpu 1 -generation -val_size 2  -generation_threshold $n -run_name svamp_gts_generation_$n &
#    python -u train_aug_svamp.py -gpu 1 -generation -val_size 2  -generation_threshold $n -run_name svamp_base_generation_$n &
#    python -u train_aug_svamp.py -gpu 1 -val_size 2 -epochs 40 -emb_name "/home/lyy/CEMAL-master/components/roberta-large" -embedding_size 1024 -generation  -generation_threshold $n -run_name svamp_large_generation_$n 
#done
#python -u train_aug_mathqa.py -gpu 0 -freeze_emb -run_name mathqa_gts_b
#python -u train_aug_mathqa.py -gpu 0 -run_name mathqa_base_b 
#python -u train_aug_mathqa.py -gpu 1 -emb_name "/home/lyy/CEMAL-master/components/roberta-large" -embedding_size 1024 -run_name mathqa_large_b

python -u train_aug_mathqa.py -gpu 0 -generation -run_name mathqa_gts_genei
#python -u train_aug_mathqa.py -gpu 1 -generation -run_name mathqa_base_no
#python -u train_aug_mathqa.py -gpu 1 -generation -epochs 40 -emb_name "/home/lyy/CEMAL-master/components/roberta-large" -embedding_size 1024 -run_name mathqa_large_genri
