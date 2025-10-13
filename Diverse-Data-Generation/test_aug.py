# coding: utf-8
import time
import torch.optim
from collections import OrderedDict
from attrdict import AttrDict
import pandas as pd
try:
	import cPickle as pickle
except ImportError:
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
log_folder = './'
parser = build_parser()
args = parser.parse_args()
config = args

run_name = config.run_name
config.log_path = os.path.join(log_folder, run_name)
log_file = os.path.join(config.log_path, 'log.txt')
logger = get_logger(run_name, log_file, logging.DEBUG)


epoch = 0
out_file_name1 = './aug/' + 'val_type1'+'.csv'
out_file_name2 = './aug/' + 'val_type2'+'.csv'
#print(out_file_name)
data_path = './data/mathqa'
input_lang = None
output_lang = None
train_ls, dev_ls = load_raw_data(data_path, config.dataset, True)
#print(len(train_ls))
train_ls = train_ls[16548:]
#print(len(train_ls))
pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_mathqa_num(train_ls, dev_ls, config.challenge_disp)
# train_pairs: ([list of token ids of question], len(ques), [list of token ids of equation], len(equation), [list of numbers], [list of indexes of numbers], [number stack])
input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
#validation_pairs = prepare_data_augmented(config, logger, validation_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

val_pairs = load_val_data(data_path, config.dataset, True)
val_pairs, copy_nums = transfer_mathqaval_num(val_pairs)
val_pairs = prepare_data_augmented(config, logger, val_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
print(len(train_pairs))
print(len(test_pairs))
print(len(val_pairs))


"""for i in range(len(train_pairs)):
	question = [input_lang.index2word[word] for word in train_pairs[i][0]]
	answer = [output_lang.index2word[word] for word in train_pairs[i][2]]
	generation_type_1(' '.join(question), out_file_name1, question, answer)
	generation_type_1(' '.join(question), out_file_name1, question, answer)
	generation_type_1(' '.join(question), out_file_name1, question, answer)
	generation_type_2(' '.join(question), out_file_name2, question, answer)
	generation_type_2(' '.join(question), out_file_name2, question, answer)
	generation_type_2(' '.join(question), out_file_name2, question, answer)"""

"""for i in range(len(val_pairs)):
	question = [input_lang.index2word[word] for word in val_pairs[i][0]]
	answer = [output_lang.index2word[word] for word in val_pairs[i][2]]
	generation_type_1(' '.join(question), out_file_name1, question, answer)
	generation_type_1(' '.join(question), out_file_name1, question, answer)
	generation_type_1(' '.join(question), out_file_name1, question, answer)
	generation_type_2(' '.join(question), out_file_name2, question, answer)
	generation_type_2(' '.join(question), out_file_name2, question, answer)
	generation_type_2(' '.join(question), out_file_name2, question, answer)"""


#generation_type_1(' '.join(question), out_file_name, question, answer)
