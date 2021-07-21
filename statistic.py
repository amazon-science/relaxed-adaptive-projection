# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import jax.numpy as np


def get_sensitivity(D):
    n, d = D.shape
    return d / n


def preserve_statistic(D):
    return np.matmul(D.transpose(), D) / D.shape[0]
