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
        r = pickle.load(f)
        train_u = pickle.load(f)
        train_i = pickle.load(f)
        train_r = pickle.load(f)
        test_u = pickle.load(f)
        test_i = pickle.load(f)
        test_r = pickle.load(f)

    index = [i for i in range(len(u))]
    random.shuffle(index)
    train_index, test_index = index[:int(training_ratio*0.9*len(u))], index[int(0.9*len(u)):]

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

    user_supp_num, user_que_num = 0, 0
    train_set_supp, test_set_supp = [], []
    train_set_que, test_set_que = [], []
    test_set_supp_size, test_set_que_size = 0, 0

    for u in train_ui_dic.keys():
        num = len(train_ui_dic[u])
        if num >= threshold:
            for index, i in enumerate(train_ui_dic[u]):
                train_set_supp.append([u, i, train_ur_dic[u][index]])
            test_set_supp_u = []
            for index, i in enumerate(test_ui_dic[u]):
                test_set_supp.append([u, i, test_ur_dic[u][index]])
            user_supp_num += 1
        else:
            for index, i in enumerate(train_ui_dic[u]):
                train_set_que.append([u, i, train_ur_dic[u][index]])
            test_set_que_u = []
            for index, i in enumera