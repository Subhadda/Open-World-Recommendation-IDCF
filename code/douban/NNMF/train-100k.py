
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import sys
import argparse
import yaml
from model import NNMFModel
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 0.002 #default 0.001
DECAYING_FACTOR = 0.99 #default 0.95
LAMBDA_REG = 0.1 #default 0.05
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 100000
n_epochs = 50

DATASET = 'ml-100k'
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

train_set = torch.tensor(train_set_supp + train_set_que)
test_set = torch.tensor(test_set_supp + test_set_que)
train_set = train_set[torch.randperm(train_set.size(0))]
val_set = train_set[int(0.95*train_set.size(0)):]
train_set = train_set[:int(0.95*train_set.size(0))]

def train(model, optimizer, train_x, train_y):
	model.train()
	optimizer.zero_grad()
	pred_y = model(train_x)
	loss_r = torch.sum((train_y - pred_y) ** 2)
	loss_reg = model.regularization_loss()
	loss = loss_r + LAMBDA_REG * loss_reg
	loss.backward()
	optimizer.step()
	return loss_r.item(), loss_reg.item()

def test(model, test_x, test_y):
	model.eval()
	with torch.no_grad():
		pred_y = model(test_x)
		loss_r = torch.mean((test_y - pred_y) ** 2)
	y_hat, y = pred_y.cpu().numpy(), test_y.cpu().numpy()
	MAE = np.mean(np.abs(y_hat - y))
	RMSE = np.sqrt(np.mean(np.square(y_hat - y)))

	return loss_r.item(), MAE, RMSE

def save_model(model, path):
	torch.save(model.state_dict(), path+'model.pkl')
