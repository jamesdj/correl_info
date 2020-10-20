import numpy as np

from correl_info.utils import sqrtm_pwd


def bures_dist(A, B):
    A /= np.trace(A)
    B /= np.trace(B)
    A_sqrt = sqrtm_psd(A)
    B_sqrt = sqrtm_psd(B)
    sqrt_prod = np.dot(A_sqrt, B_sqrt)
    return np.sqrt(2 * (1 - np.trace(sqrt_prod)))
