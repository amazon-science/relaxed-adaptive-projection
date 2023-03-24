from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp

from dataloading.dataset import Dataset


@dataclass
class BaseConfiguration:
    # Logging
    verbose: bool = False
    silent: bool = False
    debug: bool = False
    privacy_weight: int = 1


class BaseMechanism(metaclass=ABCMeta):
    def __init__(
        self,
        args: list,
        stats_module: list,
        num_generated_points: int = 1000,
        name="Base",
    ):
        self.args_list = args
        self.stat_module = stats_module
        self.num_generated_points = num_generated_points
        self.algo_name = name

        self.dataset = None
        self.statistics = None
        self.privacy_weights = None
        self.D_prime = None
        self.num_points = 0
        self.data_dimension = None

    def __str__(self):
        return self.algo_name

    def initialize(self, dataset: Dataset, seed: int):
        domain = dataset.domain
        self.dataset = dataset
        self.num_points = len(self.dataset.df)
        self.data_dimension = sum(domain.shape)
        self.statistics = [
            stat.get_statistics(domain, seed) for stat in self.stat_module
        ]

    def get_dprime(self):
        return self.D_prime

    def train(self, epsilon, seed, debug_fn=None):
        if self.dataset is None or self.statistics is None:
            raise Exception("Error must call initialize()")

        self._train(self.args_list, self.num_generated_points, epsilon, seed, debug_fn)

    @abstractmethod
    def _train(
        self, args: list, num_generated_points: int, epsilon: float, seed: int, debug_fn
    ):
        pass

    def _clip_array(self, array):
        return jnp.clip(array, 0, 1)
