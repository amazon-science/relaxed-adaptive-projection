import itertools

import jax.numpy as jnp
import numpy as np
from jax import vmap

from .base_statistics import BaseQueryClass
from .base_statistics import Statistics
from dataloading.domain import Domain


class MarginalQueryClass(BaseQueryClass):
    def __init__(self, K, conditional=True, max_number_rows=5000):
        super(MarginalQueryClass, self).__init__()
        self.K = K
        self.conditional = conditional
        self.max_number_rows = max_number_rows

    def get_statistics(self, domain: Domain, seed=0) -> Statistics:
        temp = MarginalStatistics(
            domain,
            self.K,
            conditional=self.conditional,
            seed=seed,
            max_number_rows=self.max_number_rows,
        )
        temp.max_number_rows = self.max_number_rows
        return temp


class MarginalStatistics(Statistics):
    def __init__(self, domain, k, conditional=True, seed=0, max_number_rows=5000):
        super().__init__()
        cols = domain.get_cat_cols()
        num_cols = domain.get_cont_cols()

        print(f"num_cols={num_cols}")
        self.domain = domain
        self.K = min(k, len(cols))
        self.max_number_rows = max_number_rows

        feature_indices_map = self.domain.get_feature_indices_map()
        queries = []

        max_size = 0
        targets = domain.targets
        print(f"targets={targets}")
        workload = list(itertools.combinations(cols, self.K))
        if conditional:
            features = [feat for feat in cols if feat not in targets]
            workload = list(itertools.combinations(features, self.K))
            # workload = list(itertools.combinations(cols, self.K))

            new_workload = []
            for wk in workload:
                for tar in targets:
                    new_workload.append(wk + (tar,))
            workload = new_workload
        print(f"workload size = {len(workload)}")

        K_temp = self.K + (1 if conditional else 0)
        self.marginal_size = []
        for marginal in workload:
            positions = []
            for col in marginal:
                positions.append(feature_indices_map[col])
            indices = []
            for tup in itertools.product(*positions):
                indices.append(tup)
            # K-way Index vector
            idx = []
            for k in range(K_temp):
                idx.append(jnp.array([q[k] for q in indices]))
            idx = jnp.array(idx)
            self.marginal_size.append(idx.shape[1])
            max_size = max(max_size, idx.shape[1])
            queries.append(idx)
        print(f"marginal max_size = {max_size}")
        self.marginal_size = np.array(self.marginal_size)

        self.mask_matrix = jnp.array(
            [
                jnp.concatenate(
                    [
                        jnp.ones((query_idx.shape[1],)),
                        jnp.zeros((max_size - query_idx.shape[1],)),
                    ]
                )
                for query_idx in queries
            ]
        ).astype(int)
        self.mask_matrix = jnp.vstack(
            [self.mask_matrix, jnp.zeros((1, max_size), dtype=int)]
        )

        queries = jnp.array(
            [
                jnp.hstack(
                    [query_idx, -jnp.ones((K_temp, max_size - query_idx.shape[1]))]
                )
                for query_idx in queries
            ]
        ).astype(int)

        extra_zeroes = -jnp.ones((1, K_temp, max_size), dtype=int)
        self.queries = jnp.vstack([queries, extra_zeroes])
        self.workload = workload
        print(f"self.queries.shape={self.queries.shape}")

    def compute_stat(self, D_zeroes, query_index):
        sub_marginal_queries = self.queries[query_index]
        D_temp = D_zeroes[:, sub_marginal_queries]
        answers = D_temp.prod(axis=1)
        stats = answers.sum(axis=0)
        return stats

    def get_num_queries(self):
        return self.queries.shape[0] - 1

    def get_exact_statistics_fn(self, queries_size):
        compute_stats_vmap = vmap(self.compute_stat, in_axes=(None, 0))

        def compute_statistics_fn(queries_idx: jnp.ndarray, D):
            queries_idx = queries_idx.reshape(-1).astype(int)
            # Queries index is padded with -1 at the end.
            assert (
                queries_idx.shape[0] == queries_size
            ), f"{queries_idx.shape} and {queries_size}"
            D_zeroes = jnp.hstack(
                [D, jnp.zeros((D.shape[0], 1))]
            )  # add a column of zeros
            D_zeroes = jnp.asarray(D_zeroes)
            rows = D_zeroes.shape[0]

            num_D_split = max(2, int(rows / self.max_number_rows + 0.5))
            D_splits = jnp.array_split(D_zeroes, num_D_split)
            stats = jnp.array(
                [compute_stats_vmap(D_sub, queries_idx) for D_sub in D_splits]
            ).sum(axis=0)
            stats_normed = stats / D.shape[0]

            return stats_normed

        return compute_statistics_fn

    def get_differentiable_statistics_fn(self, queries_size):
        stat_fn = self.get_exact_statistics_fn(queries_size=queries_size)

        def compute_statistics_fn(queries_idx: jnp.ndarray, D, sigmoid_param):
            return stat_fn(queries_idx, D)

        return compute_statistics_fn
