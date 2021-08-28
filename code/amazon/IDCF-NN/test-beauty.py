import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_NN_Model, NNMFModel
from utils import *
from datetime import datetime
import torch
import torch.nn.functional as F

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backen