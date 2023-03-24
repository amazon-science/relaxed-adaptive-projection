import json

import numpy as np
import pandas as pd

from dataloading.domain import Domain


def ohe_to_categorical(D, feats_idx):
    print(f"D = {D.shape}")
    print(feats_idx)
    return np.vstack(
        [
            np.argwhere(D[:, feat] == 1)[:, 1] if len(feat) > 1 else D[:, feat].T
            for feat in feats_idx
        ]
    ).T


class Dataset:
    def __init__(self, df, domain):
        """Create a Dataset object.
        This object generates a one-hot encoded data matrix.

        :param df: a pandas dataframe.
        :param domain: a domain object
        """
        assert set(domain.attrs) <= set(
            df.columns
        ), "data must contain domain attributes"
        self.domain = domain
        self.df = df.loc[:, domain.attrs]

    @staticmethod
    def synthetic(domain, N):
        """Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [
            np.random.randint(low=0, high=n, size=N) if n > 1 else np.random.rand(N)
            for n in domain.shape
        ]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load_by_name(default_dir, name):
        """Load data into a dataset object using name"""
        data_path = f"{default_dir}/{name}.csv"
        domain_path = f"{default_dir}/{name}-domain.json"
        # TODO: download data if not found in default path
        return Dataset.load(data_path=data_path, domain_path=domain_path)

    @staticmethod
    def load(data_path, domain_path):
        """Load data into a dataset object
        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(data_path)
        domfile = open(domain_path)
        config = json.load(domfile)
        domfile.close()
        domain = Domain(config.keys(), config.values())

        return Dataset(df, domain)

    def project(self, cols):
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    def datavector(self, flatten=True, weights=None):
        """return the database in vector-of-counts form"""
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=weights)[0]
        return ans.flatten() if flatten else ans

    def get_quantile_projection_intervals(self, lower=0.05, upper=0.95):
        feats_domain = {key: self.domain[key] for key in self.domain}
        projection_intervals = {}
        for col, size_bin in feats_domain.items():
            if size_bin > 1:
                projection_intervals[col] = (0, 1)
            else:
                # numeric column
                projection_min = np.quantile(
                    self.df[col].values, np.array([lower]), interpolation="higher"
                )
                projection_max = np.quantile(
                    self.df[col].values, np.array([upper]), interpolation="lower"
                )
                projection_intervals[col] = (projection_min, projection_max + 1e-8)
        return projection_intervals

    def get_quantile_projection_intervals_by_column(self, column_quantiles):
        feats_domain = {key: self.domain[key] for key in self.domain}
        projection_intervals = {}
        for col, size_bin in feats_domain.items():
            if size_bin > 1:
                projection_intervals[col] = (0, 1)
            else:
                if col in column_quantiles:
                    lower, upper = column_quantiles[col]
                else:
                    lower, upper = (0.0, 1)

                # numeric column
                projection_min = np.quantile(
                    self.df[col].values, np.array([lower]), interpolation="higher"
                )
                projection_max = np.quantile(
                    self.df[col].values, np.array([upper]), interpolation="lower"
                )
                projection_intervals[col] = (projection_min, projection_max + 1e-8)
        return projection_intervals

    @staticmethod
    def get_quantile(values, lower=0.05, upper=0.95):
        projection_min = np.quantile(values, np.array([lower]), interpolation="higher")
        projection_max = np.quantile(values, np.array([upper]), interpolation="lower")
        return projection_min, projection_max

    def scale_numeric_features(self, in_interval=None, out_interval=None):
        feats_domain = {key: self.domain[key] for key in self.domain}
        for col, size_bin in feats_domain.items():
            if size_bin == 1:
                in_min, in_max = in_interval[col] if in_interval is not None else (0, 1)
                out_min, out_max = (
                    out_interval[col] if out_interval is not None else (0, 1)
                )

                self.df[col] = (self.df[col] - in_min) / (in_max - in_min)
                self.df[col] = self.df[col] * (out_max - out_min) + out_min

    def inverse_quantile_projection(self, quantile_projections):
        feats_domain = {key: self.domain[key] for key in self.domain}
        for col, size_bin in feats_domain.items():
            if size_bin == 1:
                projection_min, projection_max = quantile_projections[col]
                self.df[col] = (self.df[col] - projection_min) / (
                    projection_max - projection_min
                )

    def get_oh_dataset(self, quantiles=False):
        """
        returns the one-hot encoding of the categorical features.
        """
        feats_domain = {key: self.domain[key] for key in self.domain}

        # get binning of attributes
        bins_size_array = [
            (
                size_bin,
                np.digitize(self.df[col], range(size_bin + 1), right=True)
                if size_bin > 1
                else self.df[col].values,
            )
            for col, size_bin in feats_domain.items()
        ]

        # perform one-hot-encoding of all features and stack them into a numpy matrix
        bin_dataset = np.hstack(
            [
                np.eye(size_bin)[bin_array]
                if size_bin > 1
                else bin_array.reshape(-1, 1)
                for size_bin, bin_array in bins_size_array
            ]
        )
        return bin_dataset

    def get_dataset(self):
        return self.get_oh_dataset()

    @staticmethod
    def preprocess(data_df, schema_path):
        data_df = data_df.copy()
        data_df = data_df.loc[:, ~data_df.columns.str.contains("^Unnamed")]

        schema = json.load(open(schema_path))

        def my_index(a_list, a_value):
            # Maps categorical values to ints. If the category does not exist in a_list then assign a random label.
            try:
                return a_list.index(a_value)
            except:
                return np.random.randint(0, len(a_list))  # TODO:

        attrs = []
        shape = []
        for col in schema:
            attrs.append(col)
            if schema[col] == "continuous":
                shape.append(1)
            elif schema[col] == "data":
                shape.append(1)
            else:
                shape.append(len(schema[col]))
                data_df[col] = data_df[col].astype("str")
                data_df.loc[:, col] = data_df.loc[:, col].apply(lambda x: x.strip())
                data_df.loc[:, col] = data_df.loc[:, col].apply(
                    lambda x: my_index(schema[col], x)
                )

        domain = Domain(attrs, shape)
        dataset = Dataset(data_df, domain)
        quantile_interval = dataset.get_quantile_projection_intervals(lower=0, upper=1)
        dataset.scale_numeric_features(in_interval=quantile_interval)
        return dataset, quantile_interval

    @staticmethod
    def postprocess(dataset, schema_path, quantile_interval):
        data_df = dataset.df.copy()
        domain = dataset.domain
        schema = json.load(open(schema_path))
        for col in schema:
            if schema[col] != "continuous":
                data_df[col] = data_df[col].astype(int)
                data_df.loc[:, col] = data_df.loc[:, col].apply(
                    lambda i: schema[col][i]
                )
            else:
                data_df[col] = data_df[col].astype(float)
        dataset = Dataset(data_df, domain)
        dataset.scale_numeric_features(out_interval=quantile_interval)

        return dataset

    def split(self, p, seed=0):
        np.random.seed(seed)
        msk = np.random.rand(len(self.df)) < p
        train = self.df[msk]
        test = self.df[~msk]
        return Dataset(train, self.domain), Dataset(test, self.domain)

    def get_sample(self, k, seed=0):
        np.random.seed(seed)
        sample_data = self.df.sample(n=k)
        return Dataset(sample_data, self.domain)


if __name__ == "__main__":
    raw_data_path = "../../data_raw/adult.csv"
    schema_path = "../../data_raw/adult-schema.json"
    col_name = "workclass"

    data_df = pd.read_csv(raw_data_path)

    data, quantile_interval = Dataset.preprocess(data_df, schema_path)
    data2 = Dataset.postprocess(data, schema_path, quantile_interval)
