import pickle
import random
import yaml
import numpy as np

random.seed(1234)
np.random.seed(1234)

def generate_data(datadir):

    try:
        with open(datadir, 'rb') as f:
            ucs_set = 