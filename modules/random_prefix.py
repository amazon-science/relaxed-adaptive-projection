import jax.numpy as jnp
import numpy as np
from jax import nn
from jax import random

from dataloading.domain import Domain
from modules.base_statistics import BaseQueryClass
from modules.random_queries_abstract import BaseRandomStatistics
from modules.random_queries_abstract import Statistics


class RandomPrefixQueryClass(BaseQueryClass):
    def __init__(
        self,
        num_random_projections,
        k=2,
        max_number_queries=10000,
        max_number_rows=2000,
    ):
        super(RandomPrefixQueryClass, self).__init__()
        self.num_random_projections = num_random_projections
        self.k = k
        self.max_number_queries = max_number_queries
        self.max_number_rows = max_number_rows

    def get_statistics(self, domain: Domain, seed=0) -> Statistics:
        temp = RandomPrefix(domain, self.num_random_projections, seed=seed, k=self.k)
        temp.max_number_queries = self.max_number_queries
        temp.max_number_rows = self.max_number_rows
        return temp


class RandomPrefix(BaseRandomStatistics):
    """ """

    def __init__(self, domain: Domain, num_random_projections, seed=0, k=2):
        super().__init__(num_random_projections, seed=seed)

        self.idx_map = domain.get_feature_indices_map()
        self._cat = domain.targets
        print(f"self._cat={self._cat}")
        self._cont = domain.get_cont_cols()
        self._continuous_features_idx = jnp.concatenate(
            [jnp.asarray(self.idx_map[con]) for con in self._cont]
        )
        self._categorical_features_idx = [
            jnp.array(self.idx_map[cat]) for cat in self._cat
        ]
        self.num_queries = num_random_projections

        max_cat_size = max([c.shape[0] for c in self._categorical_features_idx])
        temp = [
            jnp.concatenate([cat_idx, -jnp.ones(max_cat_size - cat_idx.shape[0])])
            for cat_idx in self._categorical_features_idx
        ]
        self.marginal_size = np.zeros(shape=(num_random_projections,))
        self._cat_idx_matrix = jnp.vstack(temp).astype(int)
        self.k = k

    def remove_noise(self, stats, queries_idx):
        extra_zeros = jnp.zeros(shape=(queries_idx.shape[0], stats.shape[1]))
        stats = jnp.where(queries_idx.reshape(-1, 1) >= 0, stats, extra_zeros)
        return stats

    def get_random_prefix_query(self, D, key):
        key0, key1, key2 = random.split(key, 3)
        i = random.randint(
            key0, minval=0, maxval=self._cat_idx_matrix.shape[0], shape=(1,)
        )
        numeric_col_idx = random.choice(
            key1, self._continuous_features_idx, (self.k,), replace=False
        )
        ran_thresholds = random.uniform(key2, minval=0, maxval=1, shape=(self.k,))
        cat_proj_columns = self._cat_idx_matrix[i, :].flatten()

        D_zeroes = jnp.hstack([D, jnp.zeros((D.shape[0], 1))])  # add a column of zeros
        projected_conditional_data = D_zeroes[:, cat_proj_columns]

        numerical_thresholds_answers = (
            D[:, numeric_col_idx] - ran_thresholds
        )  # this is n x k

        return numerical_thresholds_answers, projected_conditional_data

    def compute_stat(self, D, key):  # n x d
        proj_features, projected_conditional_data = self.get_random_prefix_query(D, key)

        answer = (proj_features > 0).astype(int)  # (n x k)
        answer_prod = jnp.prod(answer, axis=1).reshape(-1, 1)
        answer_prod_conditional = jnp.multiply(
            answer_prod, projected_conditional_data
        )  # n x 3
        stat = answer_prod_conditional.sum(axis=0)
        return stat

    def compute_diff_stat(self, D, key, sigmoid_param):  # n x d
        proj_features, projected_conditional_data = self.get_random_prefix_query(D, key)

        answer_diff = nn.sigmoid(jnp.multiply(sigmoid_param, proj_features))
        answer_prod = jnp.prod(answer_diff, axis=1).reshape(-1, 1)
        answer_conditional = jnp.multiply(answer_prod, projected_conditional_data)
        stat = answer_conditional.sum(axis=0)
        return stat

    @staticmethod
    def get_random_halfspace(key, dim_oh, dim):
        return (random.normal(key, (dim_oh,))) / jnp.sqrt(dim)
