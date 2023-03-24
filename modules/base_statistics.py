from abc import ABCMeta
from abc import abstractmethod

import jax.numpy as jnp
import numpy as np

from dataloading.domain import Domain


class Statistics(metaclass=ABCMeta):
    """This class defines the operatioins over the query class"""

    SENTINEL = -10000

    def __init__(self, **kwargs):
        """Define the statistics parameters."""
        pass

    @abstractmethod
    def get_exact_statistics_fn(self, queries_size):
        return 0

    @abstractmethod
    def get_differentiable_statistics_fn(self, queries_size):
        return 0

    def get_num_queries(self):
        raise Exception("Must Implement this method in concrete class")

    def get_sensitivity(self):
        return np.sqrt(2)

    def replace_by_sentinel_value(self, stats):
        return jnp.where(
            jnp.abs(stats) < 10, stats, self.SENTINEL * jnp.ones_like(stats)
        )


class BaseQueryClass(metaclass=ABCMeta):
    """This class defines the query class"""

    def __init__(self, **kwargs):
        """Define the statistics parameters."""
        pass

    @abstractmethod
    def get_statistics(self, domain: Domain, seed=0) -> Statistics:
        pass
