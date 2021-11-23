
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_NN_Model
from utils import *
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

DATASET = 'amazon-beauty'
LEARNING_RATE = 0.001
DECAYING_FACTOR = 0.95
LAMBDA_REC = 1.
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 500 # 500
EXTRA = False

datadir = '../../../data/beauty_s20.pkl'
train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list = generate_data(datadir=datadir)
config = yaml.safe_load(open("./datainfo.yaml", 'r'))
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']

if EXTRA:
	train_set = torch.tensor(train_set_supp)
else:
	train_set = torch.tensor(train_set_que)
test_set = torch.tensor(test_set_que)
supp_users = torch.tensor(user_supp_list, dtype = torch.long)

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
