import numpy as np


def arctanh(d):
    return 2 * np.arctanh(1 - d)


def map_to_unit_interval(s, min_s, max_s):
    return (s - min_s) / (max_s - min_s)
     

def cos_to_exp(d, min_s=-1, max_s=1):
    s = 1 - d
    return -np.log(map_to_unit_interval(s, min_s, max_s))

def inverse(d):
    return 1 / d

def log_inverse(d):
    return np.log(1 / d)
