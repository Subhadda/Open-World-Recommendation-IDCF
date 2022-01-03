import pickle
import random
import yaml
import numpy as np

random.seed(1234)
np.random.seed(1234)

def generate_data(datadir):

    try:
        with open(datadir, 'rb') as f:
            ucs_set = pickle.load(f)
            cs_set = pickle.load(f)
            u_his_list = pickle.load(f)
            i_his_list = pickle.load(f)
            ucs_count, cs_count, item_count = pickle.load(f)
    except:
        with open(datadir, 'rb') as f:
            ucs_set = pickle.load(f)
            cs_set = pickle.load(f)
            ucs_count, cs_count, item_count = pickle.load(f)

    user_supp_num, user_que_num = ucs_count, cs_count
    train_set_supp, test_set_supp = [], []
    train_set_que, test_set_que = [], []

    user_supp_list = [u for u in range(ucs_count)]

    def neg_sampling(int_list, neg_num = 1):
        num = len(int_list)
        neg_list = []
        for i in range(num):
            neg_list += [[np.random.randint(0, ucs_count), np.random.randint(0, item_count), 0] for _ in range(neg_num)]
        return neg_list

    for u in range(len(ucs_set)):
        train_set_supp += ucs_set[u][:-10]
        #train_set_supp += neg_sampling(ucs_set[u][:-10])
        test_set_supp += ucs_set[u][-10:]
        #test_set_supp += neg_sampling(ucs_set[u][-10:])

    for u in range(len(cs_set)):
        train_set_que += cs_set[u][:-10]
        #train_set_que += neg_sampling(cs_set[u][:-10])
        test_set_que += cs_set[u][-10:]
        #test_set_que += neg_sampling(cs_set[u][-10:])
        
    user_his_dic = {}
    for u in range(ucs_count):
        tmp = ucs_set[u][:-10]
        user_his_dic[u] = [ tmp[k][1] for k in range(len(tmp))]
    for u in range(0, cs_count):
        tmp = cs_set[u][:-10]
        user_his_dic[u+ucs_count] = [ tmp[k][1] for k in range(len(tmp))]

    print("-------Dataset Info--------")
    print("support user {}, query user {}".format(user_supp_num, user_que_num))
    print("train set size: support/query {}/{}".format(len(train_set_supp), len(train_set_que)))
    print("test set size: support/query {}/{}".format(len(test_set_