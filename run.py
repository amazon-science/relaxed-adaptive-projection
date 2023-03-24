import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from sklearn.linear_model import LogisticRegression

from benchmark.benchmark_multi_ML import evaluate_machine_learning_task
from mechanisms.mechanism_base import BaseMechanism

RELAXED_SYNC_DATA_PATH = "results/sync_data"


def save_dprime_relaxed_helper(D_prime_relaxed: jnp.array, path_name):
    """
    Save the synthetic data and return path
    """
    D_prime_relaxed = jnp.array(D_prime_relaxed)[:, :]
    jnp.save(path_name, D_prime_relaxed)


def save_synthetic_dataset(
    algorithm_name,
    dataset_name,
    eps,
    params,
    algo_seed,
    D_prime_relaxed,
    synthetic_df: pd.DataFrame,
    runtime: float,
):
    """Save D'"""
    sync_data_path = RELAXED_SYNC_DATA_PATH
    os.makedirs(sync_data_path, exist_ok=True)
    sync_data_path_1 = os.path.join(sync_data_path, algorithm_name)
    os.makedirs(sync_data_path_1, exist_ok=True)
    sync_data_path_2 = os.path.join(sync_data_path_1, dataset_name)
    os.makedirs(sync_data_path_2, exist_ok=True)
    sync_data_path_3 = os.path.join(sync_data_path_2, f"{eps:.2f}")
    os.makedirs(sync_data_path_3, exist_ok=True)
    sync_data_path_4 = os.path.join(sync_data_path_3, f"{params}")
    os.makedirs(sync_data_path_4, exist_ok=True)
    sync_data_path_5 = os.path.join(sync_data_path_4, f"{algo_seed}")
    os.makedirs(sync_data_path_5, exist_ok=True)

    relaxed_file_name = f"relaxed.npy"
    post_file_name = f"synthetic.csv"

    if D_prime_relaxed is not None:
        jnp.save(
            os.path.join(sync_data_path_5, relaxed_file_name),
            jnp.array(D_prime_relaxed)[:, :],
        )
    print(f"Saving synthetic data at {os.path.join(sync_data_path_5, post_file_name)}")
    synthetic_df.to_csv(os.path.join(sync_data_path_5, post_file_name), index=False)

    runtime_file = os.path.join(sync_data_path_5, "runtime.txt")
    print(f"Saving at {runtime_file}")
    f = open(runtime_file, "w")
    f.write(f"{runtime}")
    f.close()


def run_experiment(
    mechanism: BaseMechanism,
    dataset_fn,
    epsilons_list: list,
    algorithm_seed: list,
    params: str,
    oversamples=40,
    save_sycn_data=True,
    run_ml_eval=False,
    get_debug_fn=None,
):
    """Runs RAP++ and saves the relaxed synthetic data as .npy."""

    if algorithm_seed is None:
        algorithm_seed = [0]
    if epsilons_list is None:
        epsilons_list = [1]

    mechanism_name = str(mechanism)

    for algo_seed in algorithm_seed:
        dataset_container = dataset_fn(algo_seed)
        true_dataset_post_df = dataset_container.from_dataset_to_df_fn(
            dataset_container.train
        )
        true_test_dataset_post_df = dataset_container.from_dataset_to_df_fn(
            dataset_container.test
        )
        cat_cols = dataset_container.cat_columns
        num_cols = dataset_container.num_columns
        labels = dataset_container.label_column
        feature_columns = list(set(cat_cols + num_cols) - set(labels))

        """ ML Test """
        # if run_ml_eval:
        #     for label in dataset_container.label_column:
        #         print(f'Original results for {label}:')
        #         evaluate_machine_learning_task(true_dataset_post_df,
        #                          true_test_dataset_post_df,
        #                         feature_columns=feature_columns,
        #                          label_column=label,
        #                          cat_columns=dataset_container.cat_columns,
        #                          endmodel=LogisticRegression(penalty='l1', solver="liblinear")
        #                          )

        debug_fn = get_debug_fn(dataset_container) if get_debug_fn is not None else None

        mechanism.initialize(
            dataset_container.train, algo_seed
        )  # Pass the datase to algorithm so that it can create the statistics.

        for eps in epsilons_list:
            print(
                f"\n\n\nTraining {str(mechanism)} with dataset {str(dataset_container)} and  epsilon={eps}, algo_seed={algo_seed}"
            )

            """ Train RAP """
            stime = time.time()
            mechanism.train(epsilon=eps, seed=algo_seed, debug_fn=debug_fn)
            runtime_seconds = time.time() - stime
            minutes, seconds = divmod(time.time() - stime, 60)
            print(f"elapsed time is {int(minutes)} minutes and {seconds:.0f} seconds.")

            print("Oversampling")
            D_prime_relaxed = mechanism.get_dprime()[:, :]
            D_prime_original_format_df = (
                dataset_container.get_sync_dataset_with_oversample(
                    D_prime_relaxed, seed=algo_seed, oversample_rate=oversamples
                )
            )  # Dataset object
            print("Done")

            if save_sycn_data:
                save_synthetic_dataset(
                    mechanism_name,
                    str(dataset_container),
                    eps,
                    params,
                    algo_seed,
                    D_prime_relaxed,
                    D_prime_original_format_df,
                    runtime_seconds,
                )

            """ ML Test """
            if run_ml_eval:
                for label in dataset_container.label_column:
                    print(f"Synthetic results for {label}:")
                    evaluate_machine_learning_task(
                        D_prime_original_format_df,
                        true_test_dataset_post_df,
                        feature_columns=feature_columns,
                        label_column=label,
                        cat_columns=dataset_container.cat_columns,
                        endmodel=LogisticRegression(penalty="l1", solver="liblinear"),
                    )
