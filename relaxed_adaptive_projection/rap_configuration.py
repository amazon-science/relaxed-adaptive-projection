# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
from dataclasses import dataclass
from typing import Callable, Any
from jax import numpy as np
from .constants import Norm, ProjectionInterval


@dataclass
class RAPConfiguration:
    # Configuration
    num_points: int
    num_generated_points: int
    num_dimensions: int
    statistic_function: Callable[..., np.DeviceArray]
    preserve_subset_statistic: Callable[..., Callable[[np.DeviceArray], np.DeviceArray]]
    get_queries: Callable[..., Any]
    get_sensitivity: Callable[..., Callable[..., Any]]

    # Logging
    verbose: bool
    silent: bool

    # Hyperparameters
    epochs: int
    iterations: int
    epsilon: float
    norm: Norm
    projection_interval: ProjectionInterval
    optimizer_learning_rate: float
    lambda_l1: float
    k: int
    top_q: int
    epsilon: float
    delta: float
    use_all_queries: bool
    rap_stopping_condition: float
    initialize_binomial: bool

    feats_idx: list
