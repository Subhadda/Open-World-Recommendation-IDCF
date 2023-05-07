
import pickle
import random
import yaml
import numpy as np

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', split_way='threshold', threshold=50, supp_ratio=None, training_ratio=1):
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

    print(len(train_u)/len(u))

    
    index = [i for i in range(len(u))]
    random.shuffle(index)
    train_index, test_index = index[:int(0.9*len(u))], index[int(0.9*len(u)):]
    

    train_ui_dic = {}
    train_ur_dic = {}

    test_ui_dic = {}
    test_ur_dic = {}

    for user in range(n_user):
        train_ui_dic[user] = []
        train_ur_dic[user] = []
        test_ui_dic[user] = []
        test_ur_dic[user] = []

    
    for k in range(len(train_u)):
        train_ui_dic[train_u[k]].append(train_i[k])
        train_ur_dic[train_u[k]].append(train_r[k])

    for k in range(len(test_u)):
        test_ui_dic[test_u[k]].append(test_i[k])
        test_ur_dic[test_u[k]].append(test_r[k])
