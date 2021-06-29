import jax.numpy as np
from jax import jit
from math import sqrt
import string


def get_sensitivity(numqueries):

    def get_sensitity_k(D):
        n, d = D.shape
        return sqrt(numqueries) / n
    
    return get_sensitity_k


# k way correlations --- i.e. for all k-tuple q in queries we compute the value \sum_t (\prod_j D[t,j], j in q)
def preserve_statistic(queries):
    
    letters_idx = string.ascii_lowercase
    row_idx, col_idxs = letters_idx[0], letters_idx[1:]
    k = len(queries[0])

    assert len(queries[0]) <= len(col_idxs)

    # construct string for k-th order statistic with einsum, e.g., for 3-th marginal 'ab,ac,ad->bcd'
    eins_string = ",".join(
        [row_idx + a for a in col_idxs[:k]]) + '->' + col_idxs[:k]

    @jit
    def compute_statistic(D):
        return np.concatenate([
                np.einsum(eins_string, *[D[:, idx_q] for idx_q in q]).flatten() for q in queries
        ]) / D.shape[0]

    return compute_statistic


def preserve_subset_statistic(queries):

    @jit
    def compute_statistic(D):
        return np.concatenate([
                np.prod(D[:, q], 2).sum(0) for q in np.array_split(queries, 10)
            ]) / D.shape[0]

    return compute_statistic
