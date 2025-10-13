#!/bin/bash

#for n in 5 10 20

#do
#    python -u train_aug_svamp.py -gpu 1 -freeze_emb -generation -aug_size $n -run_name svamp_gts_aug_$n
#    python -u train_aug_svamp.py -gpu 1 -generation -aug_size $n -run_name svamp_base_aug_$n
#    python -u train_aug_svamp.py -gpu 0 -epochs 40 -emb_name "/home/lyy/CEMAL-master/components/roberta-large" -embedding_size 1024 -generation -aug_size $n -run_name svamp_large_aug_$n 
#done


#for n in 0 0.5 1

#do
#    python -u train_aug_svamp.py -gpu 1 -generation -val_size 2  -generation_threshold $n -run_name svamp_gts_generation_$n &
#    python -u train_aug_svamp.py -gpu 1 -generation -val_size 2  -generation_threshold $n -run_name svamp_base_generation_$n &
#    python -u train_aug_svamp.py -gpu 1 -val_size 2 -epochs 40 -emb_name "/home/lyy/CEMAL-master/components/roberta-large" -embedding_size 1024 -generation  -generation_threshold $n -run_name svamp_large_generation_$n 
#done
#python -u train_aug_svamp.py -gpu 0 -freeze_emb -generation -aug_size 5 -run_name svamp_gts_aug
#python -u train_aug_svamp.py -gpu 1 -aug_size 5 -generation -run_name svamp_base_aug
python -u train_aug_svamp.py -gpu 1 -epochs 40 -generation -emb_name "/home/lyy/CEMAL-master/components/roberta-large" -embedding_size 1024 -aug_size 5 -run_name svamp_large_aug_b 