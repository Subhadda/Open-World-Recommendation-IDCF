import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_GC_Model
from utils import generate_data
from datetime import datetime
import torch

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)

parser = argparse.ArgumentParser(description='PMF')
parser.add_argument('--gpus', default='0', help='gpus')
args = parser.parse_args()

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda')

LEARNING_RATE = 0.0001 # EXTRA 0.0001
DECAYING_FACTOR = 0.95 # EXTRA 0.95
LAMBDA_REC = 1.
BATCH_SIZE_TRAIN = 1024 # EXTRA 1024
BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 500 # 500

DATASET = 'ml-1m'
SPLIT_WAY = 'threshold'
EXTRA = False
THRESHOLD = 30
SUPP_RATIO = 0.8
TRAINING_RATIO = 1
datadir = '../../../data/'
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']
n_rating = config[DATASET]['n_rating']

train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list, edge_UI = \
generate_data(datadir=datadir, 
				dataset=DATASET, 
				split_way=SPLIT_WAY,
				supp_ratio=SUPP_RATIO, 
				threshold=THRESHOLD,
				training_ratio=TRAINING_RATIO)

supp_users = torch.tensor(user_supp_list, dtype = torch.long)
if EXTRA:
	train_set = torch.tensor(train_set_supp)
else:
	train_set = torch.tensor(train_set_que)
test_set = torch.tensor(test_set_que)
edge_IU = []	
for n in range(n_rating):
	edge_UI[n] = torch.tensor(edge_UI[n])
	edge_IU_n = edge_UI[n].transpose(1, 0).contiguous()
	edge_IU.append(edge_IU_n)

train_set = train_set[torch.randperm(train_set.size(0))]
val_set = train_set[int(0.95*train_set.size(0)):]
train_set = train_set[:int(0.95*train_set.size(0))]

def sequence_adjust(seq):
	seq_new = seq
	if len(seq) <= 0:
		seq_new = [np.random.randint(0, n_item) for i in range(HIS_SAMPLE_NUM)]
	if len(seq) > HIS_MAXLEN:
		random.shuffle(seq)
		seq_new = seq[:HIS_MAXLEN]
	return seq_new

def train(model, optimizer, i):
	model.train()
	optimizer.zero_grad()
	
	train_set_que_i = train_set[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
	
	train_set_i_x = train_set_que_i[:, :2].long().to(device)
	train_set_i_y = train_set_que_i[:, 2].float().to(device)
	train_set_his_i = [torch.tensor(
		sequence_adjust( user_his_dic[train_set_que_i[k][0].item()] ),
		dtype = torch.long
		)   for k in range(train_set_que_i.size(0))]
	train_set_hl_i = [train_set_his_i[k].size(0) for k in range(train_set_que_i.size(0))]
	train_set_his_i = torch.nn.utils.rnn.pad_sequence(train_set_his_i, batch_first = True, padding_value = 0.).to(device)
	train_set_hl_i = torch.tensor(train_set_hl_i, dtype=torch.long).to(device)
	edge_UI_i = [edge_UI[n][train_set_i_x[:, 0]].to(device) for n in range(n_rating)]
	edge_IU_i = [edge_IU[n][train_set_i_x[:, 1]].to(device) for n in range(n_rating)]
	if EXTRA:
		pred_y, user_emb_ind, user_emb_trd = model(train_set_i_x, train_set_his_i, train_set_hl_i, edge_UI_i, edge_IU_i, mode='EXTRA')
		loss = torch.sum((train_set_i_y - pred_y) ** 2)
		user_emb_trd_ = user_emb_trd.unsqueeze(0).repeat(user_emb_ind.size(0), 1, 1)
		user_emb_ind_ = user_emb_ind.unsqueeze(1).repeat(1, user_emb_trd.size(0), 1)
		dot_prod = torch.sum(torch.mul(user_emb_trd_, user_emb_ind_), dim=-1)
		loss_con = - torch.mean(
					dot_prod.diagonal() - torch.logsumexp(dot_prod, dim=-1)
				)
		loss += 10.0 * loss_con
	else:
		pred_y = model(train_set_i_x, train_set_his_i, train_set_hl_i, edge_UI_i, edge_IU_i)
		loss = torch.sum((train_set_i_y - pred_y) ** 2)
	loss.backward()
	optimizer.step()
	return loss.item()

def test(model, test_set, i):
	model.eval()
	with torch.no_grad():
		test_set_i = test_set[i*BATCH_SIZE_TEST : (i+1)*BATCH_SIZE_TEST]
		test_set_i_x = test_set_i[:, :2].long().to(device)
		test_set_i_y = test_set_i[:, 2].float().to(device)
		test_set_his_i = [torch.tensor(
			sequence_adjust( user_his_dic[test_set_i[k][0].item()] ),
			dtype = torch.long
			)   for k in range(test_set_i.size(0))]
		test_set_hl_i = [test_set_his_i[k].size(0) for k in range(test_set_i.size(0))]
		test_set_his_i = torch.nn.utils.rnn.pad_sequence(test_set_his_i, batch_first = True, padding_value = 0.).to(device)
		test_set_hl_i = torch.tensor(test_set_hl_i, dtype=torch.long).to(device)
		edge_UI_i = [edge_UI[n][test_set_i_x[:, 0]].to(device) for n in range(n_rating)]
		edge_IU_i = [edge_IU[n][test_set_i_x[:, 1]].to(device) for n in range(n_rating)]

		pred_y = model(test_set_i_x, test_set_his_i, test_set_hl_i, edge_UI_i, edge_IU_i)
		loss_r = torch.sum((test_set_i_y - pred_y) ** 2)
	y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
	l1 = np.sum( np.abs(y_hat - y) )
	l2 = np.sum( np.square(y_hat - y) )

	return loss_r.item(), l1, l2

def save_model(model, path):
	if EXTRA:
		torch.save(model.state_dict(), path + 'model-extra.pkl')
	else:
		torch.save(model.state_dict(), path+'model-inter.pkl')

def load_model(model