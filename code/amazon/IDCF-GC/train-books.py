
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
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO, filename='record.log', format='%(message)s')

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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda')

DATASET = 'amazon-books'
LEARNING_RATE = 0.0001
DECAYING_FACTOR = 1.
LAMBDA_REC = 1.
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 500 # 500
EXTRA = True

datadir = '../../../data/books_s20.pkl'
train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list, edge_UI = \
generate_data(datadir=datadir, sample_graph=True)
config = yaml.safe_load(open("./datainfo.yaml", 'r'))
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']

if EXTRA:
	train_set = torch.tensor(train_set_supp)
else:
	train_set = torch.tensor(train_set_que)
test_set = torch.tensor(test_set_que)
supp_users = torch.tensor(user_supp_list, dtype = torch.long)
edge_sparse = torch.tensor(edge_UI)

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

def neg_sampling(train_set_i, num_neg_per = 1):
	size = train_set_i.size(0)
	neg_iid = torch.randint(0, n_item, (num_neg_per * size, )).reshape(-1)
	return torch.stack([train_set_i[:, 0].repeat(num_neg_per), neg_iid, torch.zeros(num_neg_per * size)], dim=1)
	#return train_set_i

def auc_calc(score_label):
    fp1, tp1, fp2, tp2, auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for s in score_label:
        fp2 += (1-s[1]) # noclick
        tp2 += s[1] # click
        auc += (tp2 - tp1) * (fp2 + fp1) / 2
        fp1, tp1 = fp2, tp2
    try:
        return 1 - auc / (tp2 * fp2)
    except:
        return 0.5

def train(model, optimizer, i):
	model.train()
	optimizer.zero_grad()
	
	train_set_que_i = train_set[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
	train_set_que_neg_i = neg_sampling(train_set_que_i)
	train_set_que_i = torch.cat([train_set_que_i, train_set_que_neg_i], dim=0)
	train_set_i_x = train_set_que_i[:, :2].long().to(device)
	train_set_i_y = train_set_que_i[:, 2].float().to(device)
	train_set_his_i = [torch.tensor(
		sequence_adjust( user_his_dic[train_set_que_i[k][0].item()] ),
		dtype = torch.long
		)   for k in range(train_set_que_i.size(0))]
	train_set_hl_i = [train_set_his_i[k].size(0) for k in range(train_set_que_i.size(0))]
	train_set_his_i = torch.nn.utils.rnn.pad_sequence(train_set_his_i, batch_first = True, padding_value = 0.).to(device)
	train_set_hl_i = torch.tensor(train_set_hl_i, dtype=torch.long).to(device)
	if EXTRA:
		pred_y, user_emb_ind, user_emb_trd = model(train_set_i_x, train_set_his_i, train_set_hl_i, mode='EXTRA')
		loss = F.binary_cross_entropy_with_logits(pred_y, train_set_i_y, reduction='sum')
		user_emb_trd_ = user_emb_trd.unsqueeze(0).repeat(user_emb_ind.size(0), 1, 1)
		user_emb_ind_ = user_emb_ind.unsqueeze(1).repeat(1, user_emb_trd.size(0), 1)
		dot_prod = torch.sum(torch.mul(user_emb_trd_, user_emb_ind_), dim=-1)
		loss_con = - torch.mean(
					dot_prod.diagonal() - torch.logsumexp(dot_prod, dim=-1)
				)
		loss += 10.0 * loss_con
	else:
		pred_y = model(train_set_i_x, train_set_his_i, train_set_hl_i)
		loss = F.binary_cross_entropy_with_logits(pred_y, train_set_i_y, reduction='sum')
	loss.backward()
	optimizer.step()
	return loss.item(), 0.

def test(model, test_set):
	model.eval()
	loss_r_test_sum = 0
	score_label = []
	test_size = test_set.size(0)
	for i in range(test_size // BATCH_SIZE_TEST + 1):
		with torch.no_grad():
			test_set_i = test_set[i*BATCH_SIZE_TEST : (i+1)*BATCH_SIZE_TEST]
			test_set_neg_i = neg_sampling(test_set_i)
			test_set_i = torch.cat([test_set_i, test_set_neg_i], dim=0)
			test_set_i_x = test_set_i[:, :2].long().to(device)