import jax.numpy as jnp
import numpy as np
from jax import nn
from jax import random

from .base_statistics import BaseQueryClass
from .random_queries_abstract import BaseRandomStatistics
from .random_queries_abstract import Statistics
from dataloading.domain import Domain


class RandomFeaturesLinearQueryClass(BaseQueryClass):
    def __init__(
        self,
        num_random_projections,
        max_number_queries=50000,
        max_number_rows=5000,
    ):
        super(RandomFeaturesLinearQueryClass, self).__init__()
        self.num_random_projections = num_random_projections

        self.max_number_queries = max_number_queries
        self.max_number_rows = max_number_rows

    def get_statistics(self, domain: Domain, seed=0) -> Statistics:
        temp = RandomFeaturesLinear(domain, self.num_random_projections, seed=seed)
        temp.max_number_rows = self.max_number_rows
        return temp


class RandomFeaturesLinear(BaseRandomStatistics):
    """Condition only on binary variables"""

    def __init__(self, domain: Domain, num_random_projections, seed=0):
        super().__init__(num_random_projections, seed=seed)

        self.idx_map = domain.get_feature_indices_map()
        self._cat = domain.get_cat_cols()
        self._cont = domain.get_cont_cols()
        conditional_cols = domain.targets
        self._all_columns = self._cont + self._cat
        oh_dimension = sum(domain.shape)
        self.num_features = len(domain.shape)
        self._continuous_features_idx = jnp.concatenate(
            [jnp.asarray(self.idx_map[con]) for con in self._cont]
        )

        if conditional_cols is None:
            conditional_cols = [col for col in self._cat if 1 < domain[col] <= 2]
        self.conditional_cols_len = len(conditional_cols)
        max_cat_size = max([len(self.idx_map[c]) for c in conditional_cols])
        print(f"conditional_cols={conditional_cols}")
        print(f"max_cat_size={max_cat_size}")
        print(f"oh_dimension={oh_dimension}")

        # Get the index of conditional columns
        def pad_index(arr, pad_size):
            return jnp.concatenate([arr, -jnp.ones(pad_size - arr.shape[0])])

        conditional_index = [
            pad_index(jnp.array(self.idx_map[cat]), pad_size=max_cat_size)
            for cat in conditional_cols
        ]
        self.marginal_size = np.zeros(shape=(num_random_projections,))
        self._cat_idx_matrix = jnp.vstack(conditional_index).astype(int)
        print(f"self._cat_idx_matrix={self._cat_idx_matrix.shape}")

    def get_sensitivity(self):
        return np.sqrt(2)

    def remove_noise(self, stats, queries_idx):
        extra_zeros = jnp.zeros(shape=(queries_idx.shape[0], stats.shape[1]))
        stats = jnp.where(queries_idx.reshape(-1, 1) >= 0, stats, extra_zeros)
        return stats

    def get_random_projection_query(self, D, key):
        key0, key1, key2 = random.split(key, 3)
        i = random.randint(
            key0, minval=0, maxval=self._cat_idx_matrix.shape[0], shape=(1,)
        )
        cat_proj_columns = self._cat_idx_matrix[i, :].flatten()

        con_col_len = self._continuous_features_idx.shape[0]

        h = RandomFeaturesLinear.get_random_halfspace(
            key1, con_col_len, con_col_len
        )  # d x 1
        b = random.normal(key2, shape=(1,))

        D_cont = D[:, self._continuous_features_idx]
        proj_features = jnp.dot(D_cont, h)

        D_zeroes = jnp.hstack([D, jnp.zeros((D.shape[0], 1))])  # add a column of zeros
        projected_conditional_data = D_zeroes[:, cat_proj_columns]

        return proj_features - b, projected_conditional_data

    def compute_stat(self, D, key):  # n x d
        proj_features, projected_conditional_data = self.get_random_projection_query(
            D, key
        )

        answer = (proj_features > 0).astype(int)  # (n x 1)
        answer = jnp.multiply(answer, projected_conditional_data)  # n x 3
        stat = answer.sum(axis=0)

        return stat

    def compute_diff_stat(self, D, key, sigmoid_param):  # n x d
        proj_features, projected_conditional_data = self.get_random_projection_query(
            D, key
        )

        answer_diff = nn.sigmoid(jnp.multiply(sigmoid_param, proj_features))
        answer = jnp.multiply(answer_diff, projected_conditional_data)
        stat = answer.sum(axis=0)
        return stat

    @staticmethod
    def get_random_halfspace(key, dim_oh, dim):
        return (random.normal(key, (dim_oh, 1))) / jnp.sqrt(dim)
