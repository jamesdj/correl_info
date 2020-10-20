import numpy as np
import scipy.linalg

from correl_info.utils import sqrtm_psd


def bures_info(K):
    """
    Based on analytical expression in paper
    """
    n = K.shape[0]
    K = K / np.trace(K) * n
    J = np.ones(K.shape)
    I = np.eye(K.shape[0])
    H = I - (J / n)
    kbar = K.mean()
    hkh = np.dot(H, K).dot(H)
    sqrt_hkh = sqrtm_psd(hkh)
    sq_trace_norm = np.trace(sqrt_hkh) ** 2
    numer = sq_trace_norm
    denom = n * (n - 1)
    info = 1 - np.sqrt(kbar + (numer / denom))
    return info


def bures_lb_matrix_info(rho):
    """
    Port of implementation in Brockmeier repo
    """
    n2 = rho.shape[0]
    keep_idx = np.diag(rho) > 0
    rho = rho[keep_idx][keep_idx]
    n = np.sum(keep_idx)
    if np.trace(rho) > np.spacing(1):
        rho = rho / np.trace(rho) * n
        rho = (rho + rho.T) / 2
        d = np.mean(rho, axis=1)
        p1 = np.mean(d)
        rho_centered = ((rho - d).T - d).T + p1
        rho_centered = (rho_centered + rho_centered.T) / 2
        p_rest = scipy.linalg.eigh(rho_centered, eigvals_only=True)
        p_rest = p_rest[p_rest > 0]
        p2 = 1 / (n - 1) / n * np.sum(np.sqrt(p_rest)) ** 2
        a = p1 / (p1 + p2)
        bcoef = np.sqrt(p1 + p2).clip(-np.inf, 1)
        dist_hell2_2 = 1 - bcoef
        dist_hell2_2 = n / n2 * dist_hell2_2
    else:
        dist_hell2_2 = 0
        a = 1
    return dist_hell2_2, a
