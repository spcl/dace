import numpy as np

def kernel(M, float_n, data):
    return np.cov(np.transpose(data))
