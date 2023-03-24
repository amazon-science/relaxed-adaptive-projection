import os
import re
from pathlib import Path

import pandas as pd

from dataloading.data_functions.acs import *


def get_synthetic_datasets(
    location="sync_data",
    use_dataset_name: list = None,
    use_algorithms: list = None,
    max_rows=None,
):
    # print(use_dataset_name)

    sync_results = {
        "algorithm": [],
        "dataset_name": [],
        "epsilon": [],
        "param": [],
        "seed": [],
        "time": [],
        "sync_data_path": [],
    }

    print(f"location={location}", use_algorithms)
    for f in Path(location).rglob("*.csv"):
        head_tail = os.path.split(f.name)
        file_loc = head_tail[0]

        algorithm = f.parts[-6]
        dataset_name = f.parts[-5]
        epsilon = float(f.parts[-4])
        params = f.parts[-3]
        seed = int((f.parts[-2]))
        runtime_file_path = f"{location}/{algorithm}/{dataset_name}/{epsilon:.2f}/{params}/{seed}/runtime.txt"
        # print(algorithm, algorithm not in use_algorithms)
        # print(dataset_name, dataset_name not in use_dataset_name)
        if use_algorithms is not None and algorithm not in use_algorithms:
            continue
        if use_dataset_name is not None and dataset_name not in use_dataset_name:
            continue
        sync_results["algorithm"].append(algorithm)
        sync_results["dataset_name"].append(dataset_name)
        sync_results["epsilon"].append(epsilon)
        sync_results["param"].append(params)
        sync_results["seed"].append(seed)
        sync_results["sync_data_path"].append(str(f))

        try:
            with open(runtime_file_path) as fil:
                print(f"reading {runtime_file_path}")
                contents = fil.read()
                runtime = re.findall("[0-9]+", contents)[0]
                sync_results["time"].append(runtime)
        except OSError as e:
            print("Error: %s : %s" % (runtime_file_path, e.strerror))

    sync_results = pd.DataFrame(sync_results)
    return sync_results


def get_results():
    use_algorithms = [
        "RAP(Marginal&Halfspace)",
        # 'PGM',
    ]
    # datasets = [get_classification(d=dim) for dim in [2]]
    # datasets = [get_classification_low_informative(d=dim) for dim in [2, 4, 8, 16, 32, 64]]
    state = "CA"
    datasets = [
        get_acs(state=state, target="income"),
        # get_acs(state=state, target='travel'),
        # get_acs(state=state, target='coverage'),
        # get_acs(state=state, target='employment'),
        # get_acs(state=state, target='mobility'),
        # get_cardiovascular(),
        # get_classification(d=8)
    ]

    sync_results = []
    for dataset_fn in datasets:
        dataset_name = str(dataset_fn(0))
        print(dataset_name)
        for algo in use_algorithms:
            print(algo)
            sync_datasets_paths = get_synthetic_datasets(
                location="../results/sync_data",
                use_dataset_name=[dataset_name],
                use_algorithms=[algo],
            )
            sync_results.append(sync_datasets_paths)

    sync_results_df = pd.concat(sync_results, ignore_index=True)
    print(sync_results_df)
    return sync_results_df


def get_setting():
    use_algorithms = [
        "RAP(Marginal&Halfspace)",
        "PGM",
    ]
    # datasets = [get_classification(d=dim) for dim in [2]]
    # datasets = [get_classification_low_informative(d=dim) for dim in [2, 4, 8, 16, 32, 64]]
    state = "NY"
    datasets = [
        get_acs(state=state, target="income"),
        # get_acs(state=state, target='travel'),
        # get_acs(state=state, target='coverage'),
        # get_acs(state=state, target='employment'),
        # get_acs(state=state, target='mobility'),
        # get_cardiovascular(),
        # get_classification(d=8)
    ]
    return use_algorithms, datasets
