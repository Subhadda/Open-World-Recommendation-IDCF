
import pickle
import random
import yaml
import numpy as np

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', split_way='threshold', threshold=50, supp_ratio=None, training_ratio=1):
    n_user = config[dataset]['n_user']
    n_item = config[dataset]['n_item']
    n_rating = config[dataset]['n_rating']
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

    user_supp_num, user_que_num = 0, 0
    train_set_supp, test_set_supp = [], []
    train_set_que, test_set_que = [], []
    test_set_supp_size, test_set_que_size = 0, 0

    if split_way == 'threshold':
        for u in train_ui_dic.keys():
            num = len(train_ui_dic[u])
            if num >= threshold:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_supp.append([u, i, train_ur_dic[u][index]])
                test_set_supp_u = []