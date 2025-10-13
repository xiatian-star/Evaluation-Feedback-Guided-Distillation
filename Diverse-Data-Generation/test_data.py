# coding: utf-8
import time
import torch.optim
from collections import OrderedDict
from attrdict import AttrDict
import pandas as pd
#try:
#	import cPickle as pickle
#except ImportError:
import pickle

import pdb

from args import build_parser

from train_and_evaluate import *
from components.models import *
from components.contextual_embeddings import *
from utils.helper import *
from utils.logger import *
from utils.expressions_transfer import *
from utils.augmentation import generation_type_1, generation_type_2, transfer_num_no_tokenize

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = './mathqa_outputs/'
model_folder = 'mathqa_models'
outputs_folder = './mathqa_outputs/'
result_folder = './mathqa_outputs/'
data_path = './data/'
board_path = './mathqa_outputs/'

is_train = True
train_ls, dev_ls = load_raw_data(data_path, 'mathqa', is_train)
#train_ls = load_mawps_data("./data/mathqa/MathQA_test.json")
print(train_ls[0])
pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_mathqa_num(train_ls, dev_ls)
print(pairs_trained[0])
print(pairs_tested[0])
#pairs_trained = [(['bobby', 'ate', 'some', 'pieces', 'of', 'candy', '.', 'then', 'he', 'ate', 'NUM', 'more', '.', 'if', 'he', 'ate', 'a', 'total', 'of', 'NUM', 'pieces', 
#'of', 'candy', 'how', 'many', 'pieces', 'of', 'candy', 'had', 'he', 'eaten', 'at', 'the', 'start', '?'], ['-', 'N1', 'N0'], ['25', '43'], [10, 19], 1)]
#(['bobby', 'ate', 'some', 'pieces', 'of', 'candy', '.', 'then', 'he', 'ate', 'NUM', 'more', '.', 'if', 'he', 'ate', 'a', 'total', 'of', 'NUM', 'pieces', 
# 'of', 'candy', 'how', 'many', 'pieces', 'of', 'candy', 'had', 'he', 'eaten', 'at', 'the', 'start', '?'], ['-', 'N1', 'N0'], ['25', '43'], [10, 19], 1), 
#augmented_pairs_1 = load_augmented_data('./data/', '', 'add1.csv')
#augmented_pairs_1 = [{'Index': 'bobby ate NUM pieces of candy . then he ate NUM more . he also ate NUM pieces of chocolate . how many more pieces of candy than chocolate did bobby eat ?', 'Question': 'bobby ate N0 pieces of candy . then he ate N1 more . he also ate N2 pieces of chocolate . how many pieces of candy did he eat in total ?', 'Equation': 'Answer: N0 + N1'}, {'Index': 'bobby had NUM pieces of candy . he ate some pieces of candy . then he ate NUM more . if he still has NUM pieces of candy left how many pieces of candy had he eaten at the start ?', 'Question': 'Tom had N0 gallons of water in his tank. He used some water for cleaning his car, then he used N1 more gallons for watering the plants. If he has N2 gallons of water left in the tank, how many gallons of water did he use for cleaning his car?', 'Equation': 'Answer: (N0 - N1 - N2)'}]
#augmented_pairs_1, copy_nums = transfer_num_augmented(augmented_pairs_1, pairs_trained, copy_nums, generate_nums)
#augmented_pairs_1
