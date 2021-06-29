import jax.numpy as np


def get_sensitivity(D):
    n, d = D.shape
    return d / n


def preserve_statistic(D):
    return np.matmul(D.transpose(), D) / D.shape[0]
