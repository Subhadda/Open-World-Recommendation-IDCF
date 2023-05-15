
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import NNMFModel
from utils import *
from datetime import datetime
import torch

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
#fix_seed(1234)

parser = argparse.ArgumentParser(description='PMF')
parser.add_argument('--gpus', default='0', help='gpus')
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
HIS_SAMPLE_NUM = 100
n_epochs = 1 # 500

DATASET = 'douban'
SPLIT_WAY = 'threshold'
THRESHOLD = 30
SUPP_RATIO = 0.9
TRAINING_RATIO = 1
datadir = '../../../data/'
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']

train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic = \
generate_data(datadir=datadir, 
				dataset=DATASET, 
				split_way=SPLIT_WAY,
				supp_ratio=SUPP_RATIO, 
				threshold=THRESHOLD,
				training_ratio=TRAINING_RATIO)

test_set = torch.tensor(test_set_supp + test_set_que)
test_set_supp = torch.tensor(test_set_supp)
test_set_que = torch.tensor(test_set_que)

def test(model, test_x, test_y):
	model.eval()
	user_score_dict, user_label_dict = {}, {}
	for k in user_his_dic.keys():
		user_score_dict[k] = []
		user_label_dict[k] = []
	with torch.no_grad():
		pred_y = model(test_x)
		loss_r = torch.mean((test_y - pred_y) ** 2)
	y_hat, y = pred_y.cpu().numpy(), test_y.cpu().numpy()
	MAE = np.mean(np.abs(y_hat - y))
	RMSE = np.sqrt(np.mean(np.square(y_hat - y)))
	ndcg_sum, num = 0., 0
	for k in range(test_x.size(0)):
		u, s, y = test_x[k, 0].item(), pred_y[k].item(), test_y[k].item()
		user_score_dict[u] += [s]
		user_label_dict[u] += [y]
	for k in user_score_dict.keys():
		if len(user_score_dict[k]) <= 1:
			continue
		ndcg_sum += ndcg_k(user_score_dict[k], user_label_dict[k], len(user_score_dict[k]))
		num += 1
	return loss_r.item(), MAE, RMSE, ndcg_sum, num

def load_model(model, path):
	model.load_model(path+'model.pkl')

model = NNMFModel(n_user, n_item).to(device)
load_model(model, path='./train-douban/')
