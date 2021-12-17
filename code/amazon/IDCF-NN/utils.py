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
            ucs_count, cs_coun