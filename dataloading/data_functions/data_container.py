from dataloading.dataloading_util import get_upsampled_dataset_from_relaxed
from dataloading.dataset import Dataset


class DatasetContainer:
    def __init__(
        self,
        dataset_name,
        train,
        test,
        from_df_to_dataset,
        from_dataset_to_df_fn,
        cat_columns,
        num_columns,
        label: list,
    ):
        (
            self.dataset_name,
            self.train,
            self.test,
            self.from_df_to_dataset,
            self.from_dataset_to_df_fn,
        ) = (dataset_name, train, test, from_df_to_dataset, from_dataset_to_df_fn)
        self.domain = self.train.domain
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.label_column = label

    def __str__(self):
        return self.dataset_name

    def get_sync_dataset_with_oversample(
        self, D_relaxed, oversample_rate=40, seed=0
    ) -> Dataset:
        """
        This function takes as input a relaxed dataset matrix. Then creates a DataFrame in the original
         datas format.
        """
        D_prime_post_dataset = get_upsampled_dataset_from_relaxed(
            D_relaxed, self.domain, oversample=oversample_rate, seed=seed
        )

        post_df = self.from_dataset_to_df_fn(D_prime_post_dataset)
        return post_df
