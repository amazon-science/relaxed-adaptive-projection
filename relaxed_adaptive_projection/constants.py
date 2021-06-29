from enum import Enum
from typing import NamedTuple
from jax import numpy as np


class Norm(Enum):
    L_INFINITY = 'Linfty'
    L_1 = 'L1'
    L_2 = 'L2'
    L_5 = 'L5'
    LOG_EXP = 'LogExp'


class ProjectionInterval(NamedTuple):
    projection_min: float
    projection_max: float


class SyntheticInitializationOptions(Enum):
    RANDOM = 'random'
    RANDOM_INTERVAL = 'random_interval'
    NEAR_ORIGIN = 'near_origin'
    ZEROS = 'zeros'
    RANDOM_BINOMIAL = 'binomial'


norm_mapping = {
    Norm.L_INFINITY: np.inf,
    Norm.L_1: 1,
    Norm.L_2: 2,
    Norm.L_5: 5,
    Norm.LOG_EXP: 5
}