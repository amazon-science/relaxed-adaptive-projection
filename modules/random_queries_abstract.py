import jax.numpy as jnp
from jax import random
from jax import vmap

from .base_statistics import Statistics


class BaseRandomStatistics(Statistics):
    def __init__(
        self,
        num_random_projections,
        seed=0,
        query_splits=False,
        max_number_queries=50000,
        max_number_rows=5000,
    ):
        super().__init__()
        self.query_splits = query_splits
        self.num_queries = num_random_projections
        self.key = random.PRNGKey(seed)
        keys = random.split(self.key, num_random_projections)
        self.keys = jnp.vstack([keys, jnp.zeros(shape=(1, 2), dtype=jnp.uint32)])

        #
        """ These two parameters control the trade-off between run time speed and memory. Larger values increase
        runtime speed but can lead to memory issues. """
        self.max_number_queries = max_number_queries
        self.max_number_rows = max_number_rows

    def compute_stat(self, D, key):
        pass

    def compute_diff_stat(self, D, key, sigmoid_param):  # n x d
        pass

    def get_num_queries(self):
        return self.num_queries

    def get_exact_statistics_fn(self, queries_size):
        """Statistic function for subset of halfspaces."""
        compute_stats_vmap = vmap(self.compute_stat, in_axes=(None, 0))

        def compute_statistics_fn(queries_idx: jnp.ndarray, D):
            assert (
                queries_idx.shape[0] == queries_size
            ), f"{queries_idx.shape} and {queries_size}"

            # Queries index is padded with -1 at the end.
            sub_keys = self.keys[queries_idx]
            D = jnp.asarray(D)
            rows = D.shape[0]

            num_D_split = max(2, int(rows / self.max_number_rows + 0.5))
            print(f"num_D_split={num_D_split}")
            num_query_splits = max(2, int(queries_size / self.max_number_queries + 0.5))
            sub_keys_splits = jnp.array_split(sub_keys, num_query_splits)

            D_splits = jnp.array_split(D, num_D_split)

            stats = jnp.array(
                [
                    jnp.concatenate(
                        [
                            compute_stats_vmap(D_sub, k_split)
                            for k_split in sub_keys_splits
                        ]
                    )
                    for D_sub in D_splits
                ]
            ).sum(axis=0)
            stats_normed = stats / D.shape[0]
            extra_zeros = jnp.zeros(shape=(queries_size, stats_normed.shape[1]))
            stats_normed = jnp.where(
                queries_idx.reshape(-1, 1) >= 0, stats_normed, extra_zeros
            )

            return stats_normed

        return compute_statistics_fn

    def get_differentiable_statistics_fn(self, queries_size):
        compute_diff_stats_vmap = vmap(self.compute_diff_stat, in_axes=(None, 0, None))

        def compute_statistics_fn(queries_idx: jnp.ndarray, D, sigmoid_param):
            assert (
                queries_idx.shape[0] == queries_size
            ), f"{queries_idx.shape} and {queries_size}"

            sub_keys = self.keys[queries_idx]
            D = jnp.asarray(D)
            rows = D.shape[0]
            num_D_split = max(2, int(rows / self.max_number_rows + 0.5))
            D_splits = jnp.array_split(D, num_D_split)

            stats = jnp.array(
                [
                    compute_diff_stats_vmap(D_sub, sub_keys, sigmoid_param)
                    for D_sub in D_splits
                ]
            ).sum(axis=0)
            stats_normed = stats / D.shape[0]
            extra_zeros = jnp.zeros(shape=(queries_size, stats_normed.shape[1]))
            stats_normed = jnp.where(
                queries_idx.reshape(-1, 1) >= 0, stats_normed, extra_zeros
            )
            return stats_normed

        return compute_statistics_fn

    def get_stats_fn_split_D(self, D):
        pass
