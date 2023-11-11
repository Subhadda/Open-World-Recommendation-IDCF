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
            for index, i in enumerate(test_ui_dic[u]):
                test_set_que.append([u, i, test_ur_dic[u][index]])
            user_que_num += 1
                
    user_his_dic = {}
    for u in train_ui_dic.keys():
        user_his_dic[u] = train_ui_dic[u]

    train_u, train_i, train_r = [], [], []
    for u in range(n_user):
        ui_list = train_ui_dic[u]
        ur_list = train_ur_dic[u] 
        item_num = len(ui_list)
        if item_num <= 0:
            continue
        train_u += [u for i in range(item_num)]
        train_i += ui_list
        train_r += ur_list
    
    edge_array = np.array([train_u, train_i, train_r], dtype=np.int32)
    edge_UI = []
    
    for i in range(1, n_rating+1):
        if sample_graph:
            edge_i = edge_array[:2, edge_array[2]==i]
            edge_UI.append(edge_i)
        else:
            edge_i = edge_array[:2, edge_array[2]==i]
            edge_UI_i = np.zeros((n_user, n_item), dtype=np.int)
            edge_UI_i[edge_i[0], edge_i[1]] = 1
            edge_UI.append(edge_UI_i)

    

    print("-------Dataset Info--------")
    print("split way [all] with training_ratio {}".format(training_ratio))
    print("support user {}, query user {}".format(user_supp_num, user_que_num))
    print("train set size: support/query {}/{}".format(len(train_set_supp), len(train_set_que)))
    print("test set size: support/query {}/{}".format(len(test_set_supp), len(test_set_que)))

    if sample_graph:
        return train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, edge_array
    else:
        return train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, edge_UI

def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg

def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm