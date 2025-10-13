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
from utils.augmentation import generation_freetype_1, generation_freetype_2, mwp_filter

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

def main():
	parser = build_parser()
	args = parser.parse_args()
	config = args


	if config.mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	if is_train:

		run_name = config.run_name
		config.log_path = os.path.join(log_folder, run_name)
		config.model_path = os.path.join(model_folder, run_name)
		config.board_path = os.path.join(board_path, run_name)
		config.outputs_path = os.path.join(outputs_folder, run_name)

		vocab1_path = os.path.join(config.model_path, 'vocab1.p')
		vocab2_path = os.path.join(config.model_path, 'vocab2.p')
		config_file = os.path.join(config.model_path, 'config.p')
		log_file = os.path.join(config.log_path, 'log.txt')

		if config.results:
			config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

		if is_train:
			create_save_directories(config.log_path)
			create_save_directories(config.model_path)
			create_save_directories(config.outputs_path)
		else:
			create_save_directories(config.log_path)
			create_save_directories(config.result_path)

		logger = get_logger(run_name, log_file, logging.DEBUG)

		logger.info('Experiment Name: {}'.format(config.run_name))
		logger.debug('Created Relevant Directories')

		logger.info('Loading Data...')
		#print(os.getcwd())
		train_ls, dev_ls = load_raw_data(data_path, 'mathqa', True)

		#print(train_ls[0])		
		pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_mathqa_num(train_ls, dev_ls, config.challenge_disp)
		print(pairs_trained[0])
		
		#print(validation_pairs[1])
		logger.debug('Data Loaded...')
		if is_train:
			logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
		logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
		logger.debug('Extra Numbers: {}'.format(generate_nums))
		logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))
		if is_train:
			logger.info('Creating Vocab...')
			input_lang = None
			output_lang = None
		else:
			logger.info('Loading Vocab File...')

			with open(vocab1_path, 'rb') as f:
				input_lang = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				output_lang = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, input_lang.n_words))

		input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
		#引入训练集增强数据
		augmented_pairs = load_augmented_data(data_path, 'mathqa', 'train_aug.csv')
		#print(len(augmented_pairs))
		augmented_pairs = load_augmented_data(data_path, 'mathqa', 'train_type1.csv')
		augmented_pairs = load_augmented_data(data_path, 'mathqa', 'train_type1.csv')
		augmented_pairs, copy_nums = transfer_num_augmented_during_training(augmented_pairs, train_pairs[0], copy_nums, generate_nums)
		
		aug_pairs = prepare_data_augmented(config, logger, augmented_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
		#print(len(aug_pairs))

		train_pairs = train_pairs + aug_pairs
		print(len(train_pairs))

		validation_pairs = load_val_data(data_path, 'mathqa', True)
		validation_pairs, copy_nums = transfer_mathqaval_num(validation_pairs)	
		validation_pairs = prepare_data_augmented(config, logger, validation_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
        #引入验证集增强数据
		val_aug_pairs = load_augmented_data(data_path, 'mathqa', 'val_aug.csv')
		#print(len(val_aug_pairs))
		#val_aug_pairs = load_augmented_data(data_path, 'mathqa', 'val_aug_type1.csv')
		#val_aug_pairs = load_augmented_data(data_path, 'mathqa', 'val_aug_type2.csv')
		val_aug_pairs, copy_nums = transfer_num_augmented_during_training(val_aug_pairs, train_pairs[0], copy_nums, generate_nums)
		val_aug_pairs = prepare_data_augmented(config, logger, val_aug_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
		#print(len(val_aug_pairs))
		validation_pairs = validation_pairs + val_aug_pairs
		#print(len(validation_pairs))
		checkpoint = get_latest_checkpoint(config.model_path, logger)

		if is_train:
			with open(vocab1_path, 'wb') as f:
				pickle.dump(input_lang, f, protocol=pickle.HIGHEST_PROTOCOL)
			with open(vocab2_path, 'wb') as f:
				pickle.dump(output_lang, f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Vocab saved at {}'.format(vocab1_path))

			generate_num_ids = []
			for num in generate_nums:
				generate_num_ids.append(output_lang.word2index[num])

			config.len_generate_nums = len(generate_nums)
			config.copy_nums = copy_nums

			with open(config_file, 'wb') as f:
				pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Config File Saved')

			# train_pairs: ([list of token ids of question], len(ques), [list of token ids of equation], len(equation), [list of numbers], [list of indexes of numbers], [number stack])

			logger.info('Initializing Models...')

			# Initialize models
			embedding = None
			if config.embedding == 'bert':
				embedding = BertEncoder(config.emb_name, device, config.freeze_emb)
			elif config.embedding == 'roberta':
				embedding = RobertaEncoder(config.emb_name, device, config.freeze_emb)
			elif config.embedding == 'deberta':
				embedding = DebertaEncoder(config.emb_name, device, config.freeze_emb)
			else:
				embedding = Embedding(config, input_lang, input_size=input_lang.n_words, embedding_size=config.embedding_size, dropout=config.dropout)

			# encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			encoder = EncoderSeq(cell_type=config.cell_type, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			predict = Prediction(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), input_size=len(generate_nums), dropout=config.dropout)
			generate = GenerateNode(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), embedding_size=config.embedding_size, dropout=config.dropout)
			merge = Merge(hidden_size=config.hidden_size, embedding_size=config.embedding_size, dropout=config.dropout)
			# the embedding layer is only for generated number embeddings, operators, and paddings

			logger.debug('Models Initialized')
			logger.info('Initializing Optimizers...')

			embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=config.emb_lr, weight_decay=config.embedding_decay)
			encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			predict_optimizer = torch.optim.Adam(predict.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			generate_optimizer = torch.optim.Adam(generate.parameters(), lr=config.lr, weight_decay=config.weight_decay)
			merge_optimizer = torch.optim.Adam(merge.parameters(), lr=config.lr, weight_decay=config.weight_decay)

			logger.debug('Optimizers Initialized')
			logger.info('Initializing Schedulers...')

			embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=20, gamma=0.5)
			encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
			predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
			generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
			merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

			logger.debug('Schedulers Initialized')

			logger.info('Loading Models on GPU {}...'.format(config.gpu))

			# Move models to GPU
			if USE_CUDA:
				embedding.to(device)
				encoder.to(device)
				predict.to(device)
				generate.to(device)
				merge.to(device)

			logger.debug('Models loaded on GPU {}'.format(config.gpu))

			# generate_num_ids = []
			# for num in generate_nums:
			# 	generate_num_ids.append(output_lang.word2index[num])

			max_val_acc = 0.0
			max_train_acc = 0.0
			eq_acc = 0.0
			best_epoch = -1
			min_train_loss = float('inf')

			logger.info('Starting Training Procedure')
			for epoch in range(config.epochs):
				logger.info('Number of training samples in Epoch {}: {}'.format(str(epoch), str(len(train_pairs))))
				out_file_name = './aug/' + run_name + '_epoch_'+str(epoch)+'.csv'
				aug_file_name = './aug/' + run_name + '_augepoch_'+str(epoch)+'.csv'
				loss_total = 0
				input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, config.batch_size)

				od = OrderedDict()
				od['Epoch'] = epoch + 1
				print_log(logger, od)

				start = time.time()
				for idx in range(len(input_lengths)):
					# loss = train_tree(
					# 	input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
					# 	num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
					# 	encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx])
					loss = train_tree(
						config, input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
						num_stack_batches[idx], num_size_batches[idx], generate_num_ids, embedding, encoder, predict, generate, merge,
						embedding_optimizer, encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, input_lang, output_lang, 
						num_pos_batches[idx])
					loss_total += loss
					print("Completed {} / {}...".format(idx, len(input_lengths)), end = '\r', flush = True)

				embedding_scheduler.step()
				encoder_scheduler.step()
				predict_scheduler.step()
				generate_scheduler.step()
				merge_scheduler.step()

				logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_since(time.time() - start)))

				if loss_total / len(input_lengths) < min_train_loss:
					min_train_loss = loss_total / len(input_lengths)

				train_value_ac = 0
				train_equation_ac = 0
				train_eval_total = 1
				if (epoch == 10 or epoch == 20):
					train_eval_total = 0
					logger.info('Computing Validation Accuracy')
					start = time.time()
					with torch.no_grad():
						for train_batch in validation_pairs:
							# train_res = evaluate_tree(train_batch[0], train_batch[1], generate_num_ids, encoder, predict, generate,
							# 						 merge, output_lang, train_batch[5], beam_size=config.beam_size)
							train_res = evaluate_tree(config, train_batch[0], train_batch[1], generate_num_ids, embedding, encoder, predict, generate,
													merge, input_lang, output_lang, train_batch[5], beam_size=config.beam_size)
							train_val_ac, train_equ_ac, _, _ = compute_prefix_tree_result(train_res, train_batch[2], output_lang, train_batch[4], train_batch[6])

							if train_val_ac:
								train_value_ac += 1
							else:
								#if (epoch > 5 and epoch < 15 and epoch % 3 == 0) or (epoch > 10 and epoch % 20 ==0):
								if config.generation:
									try:
										continue_flag = False
										
										question = [input_lang.index2word[word] for word in train_batch[0]]
										answer = [output_lang.index2word[word] for word in train_batch[2]]
										
										ops = ['+','-','*','/','(',')','^']
										for eq_idx in range(len(answer)):
											if answer[eq_idx] not in ops and answer[eq_idx][0] != 'N':
												continue_flag = True
										if continue_flag:
											continue
										#logger.info('Starting auging')
										generation_freetype_1(' '.join(question), out_file_name, question, answer)
										generation_freetype_2(' '.join(question), out_file_name, question, answer)
										#logger.info('aug Completed')
									except:
										continue
							if train_equ_ac:
								train_equation_ac += 1
							train_eval_total += 1
					if config.generation:
						logger.debug('Validation Accuracy Computed...\nTime Taken: {}'.format(time_since(time.time() - start)))
						mwp_filter(out_file_name, aug_file_name)
						augmented_pairs = load_augmented_data(aug_file_name)
						augmented_pairs, copy_nums = transfer_num_augmented_during_training(augmented_pairs, train_batch, copy_nums, generate_nums)
						augmented_pairs = prepare_data_augmented(config, logger, augmented_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
						train_pairs += augmented_pairs
				logger.info('Starting Testing')
				value_ac = 0
				equation_ac = 0
				eval_total = 0
				start = time.time()

				with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
					f_out.write('---------------------------------------\n')
					f_out.write('Epoch: ' + str(epoch) + '\n')
					f_out.write('---------------------------------------\n')
					f_out.close()

				ex_num = 0
				for test_batch in test_pairs:
					# test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
					# 						 merge, output_lang, test_batch[5], beam_size=config.beam_size)
					#print(test_batch)
					test_res = evaluate_tree(config, test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, predict, generate,
											 merge, input_lang, output_lang, test_batch[5], beam_size=config.beam_size)
					val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])

					cur_result = 0
					if val_ac:
						value_ac += 1
						cur_result = 1
					if equ_ac:
						equation_ac += 1
					eval_total += 1

					with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
						f_out.write('Example: ' + str(ex_num) + '\n')
						f_out.write('Source: ' + stack_to_string(sentence_from_indexes(input_lang, test_batch[0])) + '\n')
						f_out.write('Target: ' + stack_to_string(sentence_from_indexes(output_lang, test_batch[2])) + '\n')
						f_out.write('Generated: ' + stack_to_string(sentence_from_indexes(output_lang, test_res)) + '\n')
						if config.challenge_disp:
							f_out.write('Type: ' + test_batch[7] + '\n')
							f_out.write('Variation Type: ' + test_batch[8] + '\n')
							f_out.write('Annotator: ' + test_batch[9] + '\n')
							f_out.write('Alternate: ' + str(test_batch[10]) + '\n')
						if config.nums_disp:
							src_nums = len(test_batch[4])
							tgt_nums = 0
							pred_nums = 0
							for k_tgt in sentence_from_indexes(output_lang, test_batch[2]):
								if k_tgt not in ['+', '-', '*', '/', '^']:
									tgt_nums += 1
							for k_pred in sentence_from_indexes(output_lang, test_res):
								if k_pred not in ['+', '-', '*', '/', '^']:
									pred_nums += 1
							f_out.write('Numbers in question: ' + str(src_nums) + '\n')
							f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
							f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
						f_out.write('Result: ' + str(cur_result) + '\n' + '\n')
						f_out.close()

					ex_num+=1

				if float(train_value_ac) / train_eval_total > max_train_acc:
					max_train_acc = float(train_value_ac) / train_eval_total

				if float(value_ac) / eval_total > max_val_acc:
					max_val_acc = float(value_ac) / eval_total
					eq_acc = float(equation_ac) / eval_total
					best_epoch = epoch+1

					state = {
							'epoch' : epoch,
							'best_epoch': best_epoch-1,
							'embedding_state_dict': embedding.state_dict(),
							'encoder_state_dict': encoder.state_dict(),
							'predict_state_dict': predict.state_dict(),
							'generate_state_dict': generate.state_dict(),
							'merge_state_dict': merge.state_dict(),
							'embedding_optimizer_state_dict': embedding_optimizer.state_dict(),
							'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
							'predict_optimizer_state_dict': predict_optimizer.state_dict(),
							'generate_optimizer_state_dict': generate_optimizer.state_dict(),
							'merge_optimizer_state_dict': merge_optimizer.state_dict(),
							'embedding_scheduler_state_dict': embedding_scheduler.state_dict(),
							'encoder_scheduler_state_dict': encoder_scheduler.state_dict(),
							'predict_scheduler_state_dict': predict_scheduler.state_dict(),
							'generate_scheduler_state_dict': generate_scheduler.state_dict(),
							'merge_scheduler_state_dict': merge_scheduler.state_dict(),
							'voc1': input_lang,
							'voc2': output_lang,
							'train_loss_epoch' : loss_total / len(input_lengths),
							'min_train_loss' : min_train_loss,
							'val_acc_epoch' : float(value_ac) / eval_total,
							'max_val_acc' : max_val_acc,
							'equation_acc' : eq_acc,
							'max_train_acc' : max_train_acc,
							'generate_nums' : generate_nums
						}

					if config.save_model:
						save_checkpoint(state, epoch, logger, config.model_path, config.ckpt)

				od = OrderedDict()
				od['Epoch'] = epoch + 1
				od['best_epoch'] = best_epoch
				od['train_loss_epoch'] = loss_total / len(input_lengths)
				od['min_train_loss'] = min_train_loss
				od['train_acc_epoch'] = float(train_value_ac) / train_eval_total
				od['max_train_acc'] = max_train_acc
				od['val_acc_epoch'] = float(value_ac) / eval_total
				od['equation_acc_epoch'] = float(equation_ac) / eval_total
				od['max_val_acc'] = max_val_acc
				od['equation_acc'] = eq_acc
				print_log(logger, od)

				logger.debug('Validation Completed...\nTime Taken: {}'.format(time_since(time.time() - start)))

			if config.results:
				store_results(config, max_train_acc, max_val_acc, eq_acc, min_train_loss, best_epoch)
				logger.info('Scores saved at {}'.format(config.result_path))

		else:
			gpu = config.gpu
			mode = config.mode
			dataset = config.dataset
			batch_size = config.batch_size
			old_run_name = config.run_name
			with open(config_file, 'rb') as f:
				config = AttrDict(pickle.load(f))
				config.gpu = gpu
				config.mode = mode
				config.dataset = dataset
				config.batch_size = batch_size

			logger.info('Initializing Models...')

			# Initialize models
			embedding = None
			if config.embedding == 'bert':
				embedding = BertEncoder(config.emb_name, device, config.freeze_emb)
			elif config.embedding == 'roberta':
				embedding = RobertaEncoder(config.emb_name, device, config.freeze_emb)
			else:
				embedding = Embedding(config, input_lang, input_size=input_lang.n_words, embedding_size=config.embedding_size, dropout=config.dropout)

			# encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			encoder = EncoderSeq(cell_type=config.cell_type, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			predict = Prediction(hidden_size=config.hidden_size, op_nums=output_lang.n_words - config.copy_nums - 1 - config.len_generate_nums, input_size=config.len_generate_nums, dropout=config.dropout)
			generate = GenerateNode(hidden_size=config.hidden_size, op_nums=output_lang.n_words - config.copy_nums - 1 - config.len_generate_nums, embedding_size=config.embedding_size, dropout=config.dropout)
			merge = Merge(hidden_size=config.hidden_size, embedding_size=config.embedding_size, dropout=config.dropout)
			# the embedding layer is only for generated number embeddings, operators, and paddings

			logger.debug('Models Initialized')

			epoch_offset, min_train_loss, max_train_acc, max_val_acc, equation_acc, best_epoch, generate_nums = load_checkpoint(config, embedding, encoder, predict, generate, merge, config.mode, checkpoint, logger, device)

			logger.info('Prediction from')
			od = OrderedDict()
			od['epoch'] = epoch_offset
			od['min_train_loss'] = min_train_loss
			od['max_train_acc'] = max_train_acc
			od['max_val_acc'] = max_val_acc
			od['equation_acc'] = equation_acc
			od['best_epoch'] = best_epoch
			print_log(logger, od)

			generate_num_ids = []
			for num in generate_nums:
				generate_num_ids.append(output_lang.word2index[num])

			value_ac = 0
			equation_ac = 0
			eval_total = 0
			start = time.time()

			with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
				f_out.write('---------------------------------------\n')
				f_out.write('Test Name: ' + old_run_name + '\n')
				f_out.write('---------------------------------------\n')
				f_out.close()

			test_res_ques, test_res_act, test_res_gen, test_res_scores = [], [], [], []

			ex_num = 0
			for test_batch in test_pairs:
				test_res = evaluate_tree(config, test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, predict, generate,
										 merge, input_lang, output_lang, test_batch[5], beam_size=config.beam_size)
				val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])

				cur_result = 0
				if val_ac:
					value_ac += 1
					cur_result = 1
				if equ_ac:
					equation_ac += 1
				eval_total += 1

				test_res_ques.append(stack_to_string(sentence_from_indexes(input_lang, test_batch[0])))
				test_res_act.append(stack_to_string(sentence_from_indexes(output_lang, test_batch[2])))
				test_res_gen.append(stack_to_string(sentence_from_indexes(output_lang, test_res)))
				test_res_scores.append(cur_result)

				with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
					f_out.write('Example: ' + str(ex_num) + '\n')
					f_out.write('Source: ' + stack_to_string(sentence_from_indexes(input_lang, test_batch[0])) + '\n')
					f_out.write('Target: ' + stack_to_string(sentence_from_indexes(output_lang, test_batch[2])) + '\n')
					f_out.write('Generated: ' + stack_to_string(sentence_from_indexes(output_lang, test_res)) + '\n')
					if config.nums_disp:
						src_nums = len(test_batch[4])
						tgt_nums = 0
						pred_nums = 0
						for k_tgt in sentence_from_indexes(output_lang, test_batch[2]):
							if k_tgt not in ['+', '-', '*', '/']:
								tgt_nums += 1
						for k_pred in sentence_from_indexes(output_lang, test_res):
							if k_pred not in ['+', '-', '*', '/']:
								pred_nums += 1
						f_out.write('Numbers in question: ' + str(src_nums) + '\n')
						f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
						f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
					f_out.write('Result: ' + str(cur_result) + '\n' + '\n')
					f_out.close()

				ex_num+=1

			results_df = pd.DataFrame([test_res_ques, test_res_act, test_res_gen, test_res_scores]).transpose()
			results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score']
			csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
			results_df.to_csv(csv_file_path, index = False)
			logger.info('Accuracy: {}'.format(sum(test_res_scores)/len(test_res_scores)))


'''
		augmented_pairs_1 = load_augmented_data(data_path, config.dataset, 'type_1_augmentation.csv')
		augmented_pairs_1, copy_nums = transfer_num_augmented(augmented_pairs_1, pairs_trained, copy_nums, generate_nums)
		augmented_pairs_2 = load_augmented_data(data_path, config.dataset, 'type_2_augmentation.csv')
		
		augmented_pairs_2, copy_nums = transfer_num_augmented(augmented_pairs_2, pairs_trained, copy_nums, generate_nums)
		validation_pairs = augmented_pairs_1 + augmented_pairs_2
		remove_duplicate_pairs = []
		problem_description_list = []
		for pair in validation_pairs:
			if pair[0] not in problem_description_list:
				remove_duplicate_pairs.append(pair)
				problem_description_list.append(pair[0])
		validation_pairs = remove_duplicate_pairs
		

		#pairs_trained += augmented_pairs_1
		#pairs_trained += augmented_pairs_2

		logger.debug('Data Loaded...')
		if is_train:
			logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
		logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
		logger.debug('Extra Numbers: {}'.format(generate_nums))
		logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))

		# pairs: ([list of words in question], [list of infix Equation tokens incl brackets and N0, N1], [list of numbers], [list of indexes of numbers])
		# generate_nums: Unmentioned numbers used in eqns in atleast 5 examples ['1', '3.14']
		# copy_nums: Maximum number of numbers in a single sentence: 15

		# pairs: ([list of words in question], [list of prefix Equation tokens w/ metasymbols as N0, N1], [list of numbers], [list of indexes of numbers])

		if is_train:
			logger.info('Creating Vocab...')
			input_lang = None
			output_lang = None
		else:
			logger.info('Loading Vocab File...')

			with open(vocab1_path, 'rb') as f:
				input_lang = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				output_lang = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, input_lang.n_words))

		input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

		validation_pairs = prepare_data_augmented(config, logger, validation_pairs, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)
'''        

if __name__ == '__main__':
	main()