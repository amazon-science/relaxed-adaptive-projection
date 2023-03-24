import os
import sys

import pandas as pd
from jax import devices
from jax.lib import xla_bridge

sys.path.append(os.path.dirname("./"))

print(xla_bridge.get_backend().platform)
cpu = devices("cpu")[0]
# from src.examples.util import get_dataset
import numpy as np
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from benchmark.get_results import get_synthetic_datasets
from dataloading.data_functions.acs import *


def evaluate_machine_learning_task(
    train_data,
    test_data,
    feature_columns: list,
    label_column: str,
    cat_columns: list,
    seed=0,
    endmodel=xgboost.XGBClassifier(),
):
    """
    Originally by Shuai Tang and modified by Giuseppe Vietri to work for any mixed-type dataset.

    @param train_data:
    @param test_data:
    @param label_column:
    @param cat_columns:
    @param cont_columns:
    @return:
    """
    # gridsearch_params = [
    #     (max_depth, subsample)
    #     for max_depth in range(5, 12)
    #     # for min_child_weight in range(5,8)
    #     for subsample in np.arange(0.2, 1.0, 0.2)
    # ]
    for col in cat_columns:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)

    combined_data = train_data.append(test_data, ignore_index=True)
    #
    assert label_column not in feature_columns

    X_train = train_data.copy()
    X_test = test_data.copy()
    y_train = train_data[[label_column]]
    y_test = test_data[[label_column]]
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    stored_binarizers = []

    for col in cat_columns:
        lb = LabelBinarizer()
        lb_fitted = lb.fit(combined_data[col].astype(int))
        stored_binarizers.append(lb_fitted)

    def replace_with_binarized_legit(dataframe, column_names, stored_binarizers):
        newDf = dataframe.copy()
        for idx, column_name in enumerate(column_names):
            if column_name not in newDf.columns:
                continue

            lb = stored_binarizers[idx]
            lb_results = lb.transform(newDf[column_name])
            if len(lb.classes_) <= 1:
                print(f"replace_with_binarized_legit: Error with label {column_name}")
                continue
            columns = lb.classes_ if len(lb.classes_) > 2 else [f"is {lb.classes_[1]}"]
            binarized_cols = pd.DataFrame(lb_results, columns=columns)

            newDf.drop(columns=column_name, inplace=True)
            binarized_cols.index = newDf.index
            #
            newDf = pd.concat([newDf, binarized_cols], axis=1)
        return newDf

    x_cat_cols = []
    for cat in cat_columns:
        if cat in X_train.columns:
            x_cat_cols.append(cat)

    X_train = replace_with_binarized_legit(X_train, x_cat_cols, stored_binarizers)
    X_test = replace_with_binarized_legit(X_test, x_cat_cols, stored_binarizers)

    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    endmodel = endmodel.fit(X_train, y_train)
    y_predict = endmodel.predict(np.array(X_test))

    print(classification_report(y_test, y_predict))
    results = classification_report(y_test, y_predict, output_dict=True)
    results["seed"] = seed
    return results


def benchmark_machine_learning(
    dataset_fn,
    sync_results: pd.DataFrame,
    model_name,
    max_rows=None,
):
    endmodel = None
    if model_name == "LR":
        endmodel = LogisticRegression(penalty="l1", solver="liblinear")
    elif model_name == "XGBoost":
        endmodel = xgboost.XGBClassifier(max_depth=7, subsample=0.8, eta=0.1, alpha=0.1)
    elif model_name == "kNN":
        endmodel = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    elif model_name == "MLP":
        endmodel = MLPClassifier(random_state=1, max_iter=300)

    all_rows = []

    original_res_cache = {}

    for i, row in sync_results.iterrows():
        algorithm = row["algorithm"]
        dataset_name = row["dataset_name"]
        param = row["param"]
        seed = int(row["seed"])
        time = int(row["time"])
        dataset_container = dataset_fn(seed)
        domain = dataset_container.train.domain
        assert str(dataset_container) == dataset_name
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

        for label in labels:
            # Evaluate original
            if (dataset_name, seed, label) not in original_res_cache:
                print(f"Train with Original({label}):")
                original_ml_res = evaluate_machine_learning_task(
                    true_dataset_post_df,
                    true_test_dataset_post_df,
                    feature_columns,
                    label,
                    cat_cols,
                    endmodel=endmodel,
                )
                original_res_cache[(dataset_name, seed, label)] = original_ml_res
            else:
                original_ml_res = original_res_cache[(dataset_name, seed, label)]

            local_path = row["sync_data_path"]
            print(f"Reading file {local_path}...")
            sync_data_df = pd.read_csv(local_path)
            if max_rows is not None:
                sync_data_df = sync_data_df.sample(max_rows)

            print(f"Train with Synthetic data({label}):")
            try:
                ml_res = evaluate_machine_learning_task(
                    sync_data_df,
                    true_test_dataset_post_df,
                    feature_columns,
                    label,
                    cat_cols,
                    endmodel=endmodel,
                )
            except:
                print("Failed:", local_path)

            temp_res_row = [
                dataset_name,
                seed,
                algorithm,
                row["epsilon"],
                label,
                param,
                model_name,
                original_ml_res["accuracy"],
                original_ml_res["macro avg"]["f1-score"],
                original_ml_res["weighted avg"]["f1-score"],
                ml_res["accuracy"],
                ml_res["macro avg"]["f1-score"],
                ml_res["weighted avg"]["f1-score"],
                "Sync",
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
        "label",
        "param",
        "model",
        "original accuracy",
        "original (macro) f1",
        "original (weighted avg) f1",
        "accuracy",
        "(macro) f1",
        "(weighted avg) f1",
        "Type",
        "dim",
        "time",
    ]
    all_ml_tasks = pd.DataFrame(all_rows, columns=marginal_tasks_columns)
    return all_ml_tasks


if __name__ == "__main__":
    # --------------------------------------------------
    # Choose parameters
    # --------------------------------------------------
    use_models = ["LR"]
    # use_models = ['XGBoost']
    use_algorithms = [
        "RAP(Marginal&Halfspace)",
        # "RAP",
        "Halfspace",
    ]
    datasets = [get_acs_all(state=state) for state in ["NY", "CA", "TX", "FL", "PA"]]
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
