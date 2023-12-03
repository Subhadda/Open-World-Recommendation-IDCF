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
				training_r