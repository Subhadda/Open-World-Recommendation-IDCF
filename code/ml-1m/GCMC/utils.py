import pickle
import random
import yaml
import numpy as np

random.seed(1234)

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', threshold=30, training_ratio=1, sample_graph=False):
    n_user = config[dataset]['n_user']
    n_item = config[dataset]['n_item']
    n_rating = config[dataset]['n_rating']
    with open (datadir+dataset+'.pkl','rb') as f:  
        u = pickle.load(f)
        i = pickle.load(f)
        r = pickle.load(f