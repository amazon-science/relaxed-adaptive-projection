from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, read_file, filepath, use_subset, n, d):
        self.read_file = read_file
        self.filepath = filepath
        self.use_subset = use_subset
        self.n = n
        self.d = d

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def preprocess(self, dataset):
        pass

    @abstractmethod
    def postprocess(self, dataset):
        pass
