import os
import sys

import numpy as np
from jax import jit

sys.path.append(os.path.dirname("./"))

import pandas as pd

from benchmark.get_results import get_synthetic_datasets
from dataloading.data_functions.acs import *
from modules.marginal_queries import MarginalQueryClass
from modules.random_prefix import RandomPrefixQueryClass


def benchmark_stats(dataset_fn, stat_fn, sync_results: pd.DataFrame):
    all_rows = []

    for i, row in sync_results.iterrows():
        algorithm = row["algorithm"]
        dataset_name = row["dataset_name"]
        param = row["param"]
        seed = int(row["seed"])
        time = int(row["time"])

        dataset_container = dataset_fn(seed)
        D = dataset_container.train.get_dataset()

        local_path = row["sync_data_path"]

        sync_data_df = pd.read_csv(local_path)
        D_prime = dataset_container.from_df_to_dataset(sync_data_df).get_dataset()

        errors = stat_fn(D) - stat_fn(D_prime)

        temp_res_row = [
            dataset_name,
            seed,
            algorithm,
            row["epsilon"],
            param,
            np.max(np.abs(errors)),
            np.linalg.norm(errors, ord=1),
            len(domain.attrs),
            time,
        ]
        print(temp_res_row)
        all_rows.append(temp_res_row)
    marginal_tasks_columns = [
        "dataset_name",
        "seed",
        "algorithm",
        "epsilon",
        "param",
        "max",
        "ave",
        "dim",
        "time",
    ]
    all = pd.DataFrame(all_rows, columns=marginal_tasks_columns)
    return all


if __name__ == "__main__":
    """
    Evaluate error on 20K random mixed-marginal queries.
    """

    use_stats = [
        ("marginal", MarginalQueryClass(K=2, max_number_rows=2000)),
        (
            "prefix",
            RandomPrefixQueryClass(
                num_random_projections=20000,
                k=2,
                max_number_rows=2000,
                max_number_queries=20000,
            ),
        ),
    ]

    use_algorithms = ["RAP(Marginal&Halfspace)", "Halfspace"]
    datasets = [
        get_acs(state=state, target=target)
        for state in ["NY", "CA", "TX", "FL", "PA"]
        for target in ["income", "travel", "coverage", "employment", "mobility"]
    ]
    for dataset_fn in datasets:
        dataset_name = str(dataset_fn(0))
        dataset_container = dataset_fn(0)
        D = dataset_container.train.get_dataset()
        domain = dataset_container.train.domain

        for stat_name, query_class in use_stats:
            stat = query_class.get_statistics(domain=domain, seed=123)
            true_stats_fn = jit(stat.get_exact_statistics_fn(stat.get_num_queries()))
            all_idx = np.arange(stat.get_num_queries())

            stat_fn = lambda D: true_stats_fn(all_idx, D)

            for algo in use_algorithms:
                sync_datasets_paths = get_synthetic_datasets(
                    location="results/sync_data",
                    use_dataset_name=[dataset_name],
                    use_algorithms=[algo],
                )
                results = benchmark_stats(
                    dataset_fn,
                    stat_fn,
                    sync_datasets_paths,
                )
                results["ave"] = results["ave"] / stat.get_num_queries()
                results["statistics"] = stat_name
                res_dir = "mix_results"
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, dataset_name)
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, algo)
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, stat_name)
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, "result.csv")
                print(f"Saving {res_dir}")
                results.to_csv(res_dir)
