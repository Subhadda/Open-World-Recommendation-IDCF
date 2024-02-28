import pickle
import random
import yaml
import numpy as np

random.seed(1234)

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', threshold=30, training_ratio=1):
    n_user = config[dataset]['n_user']
    n_item = config[dataset]['n_item']
    with ope