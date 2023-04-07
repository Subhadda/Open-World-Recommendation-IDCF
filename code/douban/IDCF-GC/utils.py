
import pickle
import random
import yaml
import numpy as np

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', split_way='threshold', threshold=50, supp_ratio=None, training_ratio=1):
    n_user = config[dataset]['n_user']
    n_item = config[dataset]['n_item']