
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import sys
import argparse
import yaml
from model import GCMCModel
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
DECAYING_FACTOR = 0.95 #default 0.95
LAMBDA_REG = 0.1 #default 0.05
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 100000
n_epochs = 50


DATASET = 'ml-100k'
SPLIT_WAY = 'threshold'
THRESHOLD = 30
SUPP_RATIO = 0.9
TRAINING_RATIO = 1