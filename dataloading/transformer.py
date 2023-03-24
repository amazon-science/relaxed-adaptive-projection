import numpy as np
import pandas as pd

from dataloading.dataset import Dataset
from dataloading.domain import Domain

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"


class Transformer:
    """Continuous and ordinal columns are normalized to [0, 1].
    Discrete columns are converted to a one-hot vector.
    """

    def __init__(
        self, categorical_columns, continuous_columns, bin_size=None, normalize=False
    ):
        self.meta = None
        self.output_dim = None
        self.bin_size = bin_size
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.normalize = normalize

    @staticmethod
    def get_metadata(
        data,
        categorical_columns=tuple(),
        continuous_columns=tuple(),
        ordinal_columns=tuple(),
        normalize=False,
    ):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append(
                    {
                        "name": index,
                        "type": CATEGORICAL,
                        "size": len(mapper),
                        "i2s": mapper,
                    }
                )
            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append(
                    {"name": index, "type": ORDINAL, "size": len(mapper), "i2s": mapper}
                )
            elif index in continuous_columns:
                meta.append(
                    {
                        "name": index,
                        "type": CONTINUOUS,
                        "min": column.min() if normalize else 0,
                        "max": column.max() if normalize else 1,
                    }
                )

        return meta

    def fit(self, data):
        self.meta = self.get_metadata(
            data,
            self.categorical_columns,
            self.continuous_columns,
            normalize=self.normalize,
        )
        self.output_dim = 0
        for info in self.meta:
            if info["type"] in [CONTINUOUS, ORDINAL]:
                self.output_dim += 1
            else:
                self.output_dim += info["size"]

    def transform(self, data: pd.DataFrame, target: list) -> Dataset:
        data_t = []
        attrs = []
        shape = []
        original_shape = []
        for id_, info in enumerate(self.meta):
            col_name = info["name"]
            col = data[col_name].to_numpy()
            attrs.append(col_name)
            if info["type"] == CONTINUOUS:
                if self.bin_size is not None:
                    col_binned = np.array(
                        pd.cut(
                            col,
                            bins=self.bin_size,
                            labels=np.arange(self.bin_size),
                            retbins=False,
                        )
                    )
                    col_binned = col_binned.reshape([-1, 1])
                    data_t.append(col_binned)
                    shape.append(self.bin_size)
                else:
                    col_normed = (col - info["min"]) / (
                        info["max"] - info["min"] + 1e-9
                    )
                    data_t.append(col_normed.reshape([-1, 1]))
                    shape.append(1)
                original_shape.append(1)
            else:
                mapper = dict([(item, id) for id, item in enumerate(info["i2s"])])
                mapped = data[col_name].apply(lambda x: mapper[x]).values.astype(int)
                data_t.append(mapped.reshape([-1, 1]).astype(int))
                shape.append(info["size"])
                original_shape.append(info["size"])

        domain = Domain(attrs, shape, target)
        df = pd.DataFrame(np.concatenate(data_t, axis=1), columns=domain.attrs)
        for id_, info in enumerate(self.meta):
            col_name = info["name"]
            if info["type"] == CATEGORICAL:
                df[col_name] = df[col_name].astype(int)

        self.original_domain = Domain(attrs, original_shape, target)
        return Dataset(df, domain)

    def inverse_transform(self, data: Dataset) -> pd.DataFrame:
        """Takes as input a Dataset and return a DataFrame"""
        data_t = pd.DataFrame()

        data = data.df.copy()
        res = []
        column_names = []
        for id_, info in enumerate(self.meta):
            col_name = info["name"]
            column_names.append(col_name)
            if info["type"] == CONTINUOUS:
                current = data.iloc[:, 0]
                data = data.iloc[:, 1:]
                min_val = info["min"]
                max_val = info["max"]
                if self.bin_size is not None:
                    lower = current.values / self.bin_size
                    upper = (current.values + 1) / self.bin_size
                    current = np.random.uniform(lower, upper)
                    current = current * (max_val - min_val) + min_val
                    res.append(current)
                else:
                    current = (current * (max_val - min_val) + min_val).values
                    res.append(current)
            else:
                current = data.iloc[:, 0].astype(int)
                data = data.iloc[:, 1:]
                col_data = np.array(list(map(info["i2s"].__getitem__, current)))
                res.append(col_data)

        data_t = pd.DataFrame(np.column_stack(res), columns=column_names)
        for id_, info in enumerate(self.meta):
            col_name = info["name"]
            if info["type"] == CATEGORICAL:
                data_t[col_name] = data_t[col_name].astype(int)

        return data_t


if __name__ == "__main__":
    cat = ("animal",)
    con = ("age",)
    trans = Transformer(cat, con, bin_size=None)

    data = pd.DataFrame(
        [
            ["cat", 19],
            ["cat", 11],
            ["dog", 15.1],
            ["dog", 14.5],
            ["dog", 13.1],
            ["dog", 11.2],
            ["dog", 15.6],
            ["whale", 17],
            ["whale", 23],
        ],
        columns=cat + con,
    )
    trans.fit(
        data,
    )
