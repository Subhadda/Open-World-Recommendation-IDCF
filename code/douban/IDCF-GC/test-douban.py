import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_GC_Model, GCMCModel
from utils import *
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
parser.add_argument('--extra', action="store_true", help='whether extra or inter')
args = parser.parse_args()

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda')

LEARNING_RATE = 0.001
DECAYING_FACTOR = 1.
LAMBDA_REG = 0.05
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 1 # 500

DATASET = 'douban'
SPLIT_WAY = 'threshold'
THRESHOLD = 30
SUPP_RATIO = 0.8
TRAINING_RATIO = 1
EXTRA = args.extra

datadir = '../../../data/'
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']
n_rating = config[DATASET]['n_rating']

train_set_supp, train_set_que, test_set_supp, test_set_que, user_supp_list, user_his_dic, edge_UI = \
generate_data(datadir=datadir, 
				dataset=DATASET, 
				split_way=SPLIT_WAY,
				supp_ratio=SUPP_RATIO, 
				threshold=THRESHOLD,
				training_ratio=TRAINING_RATIO)

user_supp_num = len(user_supp_list)
user_que_num = n_user - user_supp_num

if SPLIT_WAY == 'all':
	train_set_supp = torch.tensor(train_set_supp + train_set_que)
else:
	train_set_supp = torch.tensor(train_set_supp)
train_set_que = torch.tensor(train_set_que)
test_set_supp = torch.tensor(test_set_supp)
test_set_que = torch.tensor(test_set_que)
supp_users = torch.tensor(user_supp_list, dtype = torch.long)
edge_IU = []	
for n in range(n_rating):
	edge_UI[n] = torch.tensor(edge_UI[n])
	edge_IU_n = edge_UI[n].transpose(1, 0).contiguous()
	edge_IU.append(edge_IU_n)

def sequence_adjust(seq):
	seq_new = seq
	if len(seq) <= 0:
		seq_new = [np.random.randint(0, n_item) for i in range(HIS_SAMPLE_NUM)]
	if len(seq) > HIS_MAXLEN:
		random.shuffle(seq)
		seq_new = seq[:HIS_MAXLEN]
	return seq_new

def train(model, optimizer, i, supp_or_que):
	model.train()
	optimizer.zero_grad()
	
	if supp_or_que == 'supp':
		train_set_supp_i = train_set_supp[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
		train_set_supp_i_x = train_set_supp_i[:, :2].long().to(device)
		train_set_supp_i_y = train_set_supp_i[:, 2].float().to(device)
		edge_UI_i = [edge_UI[n][train_set_supp_i_x[:, 0]].to(device) for n in range(n_rating)]
		edge_IU_i = [edge_IU[n][train_set_supp_i_x[:, 1]].to(device) for n in range(n_rating)]

		pred_y = model(train_set_supp_i_x, edge_UI_i, edge_IU_i)
		loss_r = torch.sum((train_set_supp_i_y - pred_y) ** 2)
		loss_reg = model.regularization_loss()
		loss = loss_r + LAMBDA_REG * loss_reg
	else:
		train_set_que_i = train_set_que[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
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
		pred_y = model(train_set_i_x, train_set_his_i, train_set_hl_i, edge_UI_i, edge_IU_i)
		loss = torch.sum((train_set_i_y - pred_y) ** 2)
		
	loss.backward()

def test(model, test_set, supp_or_que):
	model.eval()
	loss_r_test_sum, l1_sum, l2_sum, ndcg_sum, num = 0., 0., 0., 0., 0
	test_size = test_set.size(0)
	user_score_dict, user_label_dict = {}, {}
	for k