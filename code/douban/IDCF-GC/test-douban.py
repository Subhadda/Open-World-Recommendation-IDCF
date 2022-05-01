import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_GC_Model, GCMCModel
from utils import *
from datetime import datetime
import torch

def fix_seed(seed):
    rando