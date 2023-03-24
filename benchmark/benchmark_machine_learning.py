import os
import sys

from jax import devices
from jax.lib import xla_bridge

sys.path.append(os.path.dirname("./"))

print(xla_bridge.get_backend().platform)
cpu = devices("cpu")[0]

from benchmark.benchmark_multi_ML import benchmark_machine_learning
from benchmark.get_results import get_synthetic_datasets
from dataloading.data_functions.acs import *

if __name__ == "__main__":
    # --------------------------------------------------
    # Choose parameters
    # --------------------------------------------------
    use_models = ["LR"]
    # use_models = ['XGBoost']

    use_algorithms = [
        "RAP(Marginal&Halfspace)",
        # 'RAP',
        "Halfspace",
    ]
    datasets = [
        get_acs(state=state, target=target)
        for state in ["NY", "CA", "TX", "FL", "PA"]
        for target in ["income", "coverage", "employment", "travel", "mobility"]
    ]
    for dataset_fn in datasets:
        dataset_name = str(dataset_fn(0))
        print(dataset_name)
        for algo in use_algorithms:
            print(algo)
            sync_datasets_paths = get_synthetic_datasets(
                location="results/sync_data",
                use_dataset_name=[dataset_name],
                use_algorithms=[algo],
            )
            print(sync_datasets_paths[["algorithm", "dataset_name", "time"]])
            for model_name in use_models:
                print(model_name)

                results = benchmark_machine_learning(
                    dataset_fn, sync_datasets_paths, model_name=model_name
                )

                res_dir = "ml_results"
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, dataset_name)
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, algo)
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, model_name)
                os.makedirs(res_dir, exist_ok=True)
                res_dir = os.path.join(res_dir, "result.csv")
                print(f"Saving {res_dir}")
                results.to_csv(res_dir)
