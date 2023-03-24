import json

import jax
import numpy as np
import pandas as pd
from jax import jit
from jax import numpy as jnp
from jax import random
from jax import vmap

from dataloading.dataset import Dataset
from dataloading.domain import Domain
from dataloading.transformer import Transformer


def ohe_to_categorical(D, feats_idx):
    return jnp.vstack(
        jnp.argwhere(D[:, feat] == 1)[:, 1] if len(feat) > 1 else D[:, feat].T
        for feat in feats_idx
    ).T


def get_dataset_from_oh(D, domain):
    feats_csum = np.array([0] + list(domain.shape)).cumsum()
    feats_idx = [
        list(range(feats_csum[i], feats_csum[i + 1]))
        for i in range(len(feats_csum) - 1)
    ]
    cat_data = ohe_to_categorical(D, feats_idx)
    df = pd.DataFrame(data=cat_data, columns=list(domain.attrs))
    return Dataset(df, domain=domain)


def get_dataset(
    raw_data_name,
    raw_data_location="../../../data_raw",
    mean_std_norm=False,
    bin_size=None,
    normalize=False,
):
    raw_data_path = f"{raw_data_location}/{raw_data_name}.csv"
    column_path = f"{raw_data_location}/{raw_data_name}-columns.json"
    return get_dataset_help(
        raw_data_path, column_path, bin_size=bin_size, normalize=normalize
    )


def get_dataset_help(raw_data_path, column_path, bin_size=None, normalize=False):
    # raw_data_path = f'{raw_data_location}/{raw_data_name}.csv'
    with open(column_path, "rb") as f:
        raw_data_column_types = json.load(f)

    # Read train and test data
    raw_data_df = pd.read_csv(raw_data_path)

    transformer = Transformer(
        raw_data_column_types["categorical"],
        raw_data_column_types["continuous"],
        bin_size=bin_size,
        normalize=normalize,
    )
    transformer.fit(raw_data_df)
    dataset = transformer.transform(raw_data_df)
    post_df = transformer.inverse_transform(dataset)

    post_fn = lambda dataset: transformer.inverse_transform(dataset)
    return dataset, post_df, post_fn, transformer


import matplotlib.pyplot as plt


def test_get_dataset():
    data, post, post_from_matrix_fn, transformer = get_dataset(
        "classification2d", raw_data_location="../../data_raw", mean_std_norm=True
    )
    data_df = data.df
    # post_data_df = transformer.inverse_transform(data_df)
    post_data_df = post_from_matrix_fn(data.get_dataset())

    data_df.plot(kind="hist", subplots=True, sharex=True, sharey=True, title="data_df")
    post_data_df.plot(
        kind="hist", subplots=True, sharex=True, sharey=True, title=f"post_data_df"
    )
    plt.show()

    print("pre")
    for col in transformer.continuous_columns:
        col_data = data_df[col]
        print(
            f"{col:<50}:"
            f"{col_data.mean():<20.5f}"
            f"{col_data.std():<20.5f}"
            f"{col_data.quantile(0.01):<20.5f}"
            f"{col_data.quantile(0.99):<20.5f}"
            f"{col_data.unique().shape[0]:<50}"
        )
    print("post")
    for col in transformer.continuous_columns:
        col_data = post_data_df[col]
        print(
            f"{col:<50}:"
            f"{col_data.mean():<20.5f}"
            f"{col_data.std():<20.5f}"
            f"{col_data.quantile(0.01):<20.5f}"
            f"{col_data.quantile(0.99):<20.5f}"
            f"{col_data.unique().shape[0]:<50}"
        )


def get_upsample_fn(oversample):
    def get_upsample_row(key, prob):
        cat_size = prob.shape[0]  # number of categories. 1 if continuous

        if cat_size > 1:
            categorical_labels = jnp.arange(prob.shape[0])
            categories_sizes = (oversample * prob.squeeze() + 0.5).astype(
                int
            )  # round to nearest integer
            idx_permuted = random.permutation(key, jnp.arange(prob.shape[0]))
            categorical_labels = categorical_labels[idx_permuted]
            categories_sizes = categories_sizes[idx_permuted]
            all_labels = jnp.repeat(
                categorical_labels, categories_sizes, total_repeat_length=oversample
            )
            all_labels = random.permutation(key, all_labels)
            return all_labels.astype(int)
        else:
            temp = prob[0] * jnp.ones(oversample).astype(float)
            return temp

    return get_upsample_row


def get_upsampled_dataset_from_relaxed(D, domain: Domain, oversample=1, seed=0):
    feats_idx = domain.get_feats_idx()
    n, d = D.shape
    key = jax.random.PRNGKey(seed)
    key_cols = jax.random.split(key, len(feats_idx))

    all_columns = []
    for i, (feat, key_col) in enumerate(zip(feats_idx, key_cols)):
        get_upsample_row = get_upsample_fn(oversample)
        get_upsample_row_vmap = jit(vmap(get_upsample_row))
        sub_keys = jax.random.split(key_col, n)
        col = get_upsample_row_vmap(sub_keys, D[:, feat]).reshape(-1, 1)
        all_columns.append(col)

    new_D = np.hstack(all_columns)
    df = pd.DataFrame(data=new_D, columns=list(domain.attrs))
    dataset = Dataset(df, domain=domain)
    return dataset


if __name__ == "__main__":
    test_get_dataset()
