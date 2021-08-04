
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

DATASET = 'amazon-beauty'
LEARNING_RATE = 0.0001 # EXTRA 0.00001