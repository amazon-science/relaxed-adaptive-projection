# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import jax.numpy as np


def get_sensitivity(D):
    n, d = D.shape
    return d ** (3 / 2) / n


# 3 way correlations --- i.e. for all triples of columns i,j,k, we compute the value sum_t D[t,i]*D[t,j]*D[t,k]
def preserve_statistic(D):
    return np.einsum("ij,ik,il->jkl", D, D, D) / D.shape[0]
