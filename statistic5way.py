import jax.numpy as np


def get_sensitivity(D):
    n, d = D.shape
    return d ** (5 / 2) / n


# 5 way correlations --- i.e. for all quintuplets of columns j,k,l,t,s we compute the value sum_t D[i,j]*D[i,k]*D[i,l]*D[i,t]*D[i,s]
def preserve_statistic(D):
    return np.einsum("ij,ik,il,it,is->jklts", D, D, D, D, D) / D.shape[0]
