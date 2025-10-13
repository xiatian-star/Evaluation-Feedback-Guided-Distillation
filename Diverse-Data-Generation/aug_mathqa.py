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
from utils.augmentation import generation_freetype_1, generation_freetype_2, transfer_num_no_tokenize

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
out_file_name1 = './aug/' + 'glmfree' + 'type1epoch_'+str(epoch)+'.csv'
out_file_name2 = './aug/' + 'glmfree' + 'type2epoch_'+str(epoch)+'.csv'
#print(out_file_name)
data_path = './data/mathqa'
input_lang = None
output_lang = None
train_ls, dev_ls = load_raw_data(data_path, config.dataset, True)
pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_mathqa_num(train_ls, dev_ls, config.challenge_disp)
# train_pairs: ([list of token ids of question], len(ques), [list of token ids of equation], len(equation), [list of numbers], [list of indexes of numbers], [number stack])
input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
#validation_pairs = prepare_data_augmented(config, logger, validation_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

#if is_train:
#			with open(vocab1_path, 'wb') as f:
#				pickle.dump(input_lang, f, protocol=pickle.HIGHEST_PROTOCOL)

#for train_batch in validation_pairs:

#random_problem = random.choice(train_pairs)
#question = [input_lang.index2word[word] for word in train_batch[0]
#answer = [output_lang.index2word[word] for word in train_batch[2]]
for random_problem in train_pairs:
	question = [input_lang.index2word[word] for word in random_problem[0]]
	answer = [output_lang.index2word[word] for word in random_problem[2]]
	generation_freetype_1(' '.join(question), out_file_name1, question, answer)
	generation_freetype_1(' '.join(question), out_file_name1, question, answer)
	generation_freetype_1(' '.join(question), out_file_name1, question, answer)
	generation_freetype_2(' '.join(question), out_file_name2, question, answer)
	generation_freetype_2(' '.join(question), out_file_name2, question, answer)
	generation_freetype_2(' '.join(question), out_file_name2, question, answer)