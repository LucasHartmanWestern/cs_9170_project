import os
import random
import torch
import numpy as np


def generate_random_seeds(num_seeds):
    seeds = []
    for i in range(num_seeds):
        seeds.append(int.from_bytes(os.urandom(4), 'big'))
    return seeds


def make_deterministic(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

