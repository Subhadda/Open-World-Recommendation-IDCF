import pickle
import random
import yaml
import numpy as np

random.seed(1234)

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', threshold=30, training_ratio=1):
    n_user = config[dataset]['n_user']
    n_item = config[dataset]['n_item']
    with open (datadir+dataset+'.pkl','rb') as f:  
        u = pickle.load(f)
        i = pickle.load(f)
        r = pickle.load(f)
        train_u = pickle.load(f)
        train_i = pickle.load(f)
        train_r = pickle.load(f)
        test_u = pickle.load(f)
        test_i = pickle.load(f)
        test_r = pickle.load(f)

    index = [i for i in range(len(u))]
    random.shuffle(index)
    train_index, test_index = index[:int(0.9*training_ratio*len(u))], index[int(0.9*len(u)):]

    train_ui_dic = {}
    train_ur_dic = {}

    test_ui_dic = {}
    test_ur_dic = {}

    for user in range(n_user):
        train_ui_dic[user] = []
        train_ur_dic[user] = []
        test_ui_dic[user] = []
        test_ur_dic[user] = []
    
    for index in train_index:
        train_ui_dic[u[index]].append(i[index])
        train_ur_dic[u[index]].append(r[index])

    for index in test_index:
        test_ui_dic[u[index]].append(i[index])
        test_ur_dic[u[index]].append(r[index])

   