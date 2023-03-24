import itertools
from functools import reduce

import numpy as np


class Domain:
    def __init__(self, attrs, shape, targets: list):
        """Construct a Domain object

        :param attrs: a list or tuple of attribute names
        :param shape: a list or tuple of domain sizes for each attribute
        """
        assert len(attrs) == len(shape), "dimensions must be equal"
        self.attrs = tuple(attrs)
        self.shape = tuple(shape)
        self.config = dict(zip(attrs, shape))
        self.targets = targets

    @staticmethod
    def fromdict(config, targets: list):
        """Construct a Domain object from a dictionary of { attr : size } values"""
        return Domain(config.keys(), config.values(), targets)

    def get_feats_idx(self):
        feats_csum = np.array([0] + list(self.shape)).cumsum()
        feats_idx = [
            list(range(feats_csum[i], feats_csum[i + 1]))
            for i in range(len(feats_csum) - 1)
        ]
        return feats_idx

    def get_feature_indices_map(self):
        feature_indices_map = {}
        col_map = {}
        for i, col in enumerate(self.attrs):
            col_map[col] = i

        cur = 0
        for j, sz in enumerate(self.shape):
            col = self.attrs[j]
            feature_indices_map[col] = list(range(cur, cur + sz))
            cur += sz
        return feature_indices_map

    def project(self, attrs):
        """project the domain onto a subset of attributes

        :param attrs: the attributes to project onto
        :return: the projected Domain object
        """
        # return the projected domain
        if type(attrs) is str:
            attrs = [attrs]
        shape = tuple(self.config[a] for a in attrs)
        return Domain(attrs, shape, self.targets if self.targets in attrs else None)

    def project_cat(self):
        """projects categorical attributes"""
        cat_attrs = []
        for att in self.config:
            sz = self.config[att]
            if sz > 1:
                cat_attrs.append(att)
        return self.project(cat_attrs)

    def project_numerical(self):
        """projects numerical attributes"""
        cont_attrs = []
        for att in self.config:
            sz = self.config[att]
            if sz == 1:
                cont_attrs.append(att)
        return self.project(cont_attrs)

    def marginalize(self, attrs):
        """marginalize out some attributes from the domain (opposite of project)

        :param attrs: the attributes to marginalize out
        :return: the marginalized Domain object
        """
        proj = [a for a in self.attrs if not a in attrs]
        return self.project(proj)

    def axes(self, attrs):
        """return the axes tuple for the given attributes

        :param attrs: the attributes
        :return: a tuple with the corresponding axes
        """
        return tuple(self.attrs.index(a) for a in attrs)

    def transpose(self, attrs):
        """reorder the attributes in the domain object"""
        return self.project(attrs)

    def invert(self, attrs):
        """returns the attributes in the domain not in the list"""
        return [a for a in self.attrs if a not in attrs]

    def merge(self, other):
        """merge this domain object with another

        :param other: another Domain object
        :return: a new domain object covering the full domain

        Example:
        >>> D1 = Domain(['a','b'], [10,20])
        >>> D2 = Domain(['b','c'], [20,30])
        >>> D1.merge(D2)
        Domain(['a','b','c'], [10,20,30])
        """
        extra = other.marginalize(self.attrs)
        return Domain(
            self.attrs + extra.attrs,
            self.shape + extra.shape,
            self.targets + other.targets,
        )

    def contains(self, other):
        """
        returns True if this domain is a superset of another domain.
        """
        return set(other.attrs) <= set(self.attrs)

    def size(self, attrs=None):
        """return the total size of the domain"""
        if attrs == None:
            return reduce(lambda x, y: x * y, self.shape, 1)
        return self.project(attrs).size()

    def sort(self, how="size"):
        """return a new domain object, sorted by attribute size or attribute name"""
        if how == "size":
            attrs = sorted(self.attrs, key=self.size)
        elif how == "name":
            attrs = sorted(self.attrs)
        return self.project(attrs)

    def canonical(self, attrs):
        """return the canonical ordering of the attributes"""
        return tuple(a for a in self.attrs if a in attrs)

    def __contains__(self, attr):
        return attr in self.attrs

    def __getitem__(self, a):
        """return the size of an individual attribute
        :param a: the attribute
        """
        return self.config[a]

    def __iter__(self):
        """iterator for the attributes in the domain"""
        return self.attrs.__iter__()

    def __len__(self):
        return len(self.attrs)

    def __eq__(self, other):
        return self.attrs == other.attrs and self.shape == other.shape

    def __repr__(self):
        inner = ", ".join(["%s: %d" % x for x in zip(self.attrs, self.shape)])
        return "Domain(%s)" % inner

    def __str__(self):
        return self.__repr__()

    def get_cat_cols(self):
        """returns the names of all categorical columns"""
        cat_cols = [col for col in self.attrs if self.size(col) > 1]
        return cat_cols

    def get_cont_cols(self):
        """returns the names of all continuous columns"""
        cont_cols = [col for col in self.attrs if self.size(col) == 1]
        return cont_cols

    def categorical_columns_mask(self):
        """
        returns a boolean vector of size d, where the i-th entry is True
        if it's a categorical feature and False otherwise
        """
        cat_cols = [self.size(col) > 1 for col in self.attrs]
        return np.array(cat_cols)

    def random_kway_mixed(self, num_kways=-1, k=1, seed=0):
        """
        Get k-way combinations using both categorical and continuous features

        @param domain: shape of the data
        @param num_kways: is the number of feature combinations, if -1 then return all k-way combinations.
        @param k: k-way combinations
        @param seed:  random seed
        @return: a list of  k-way tuples of size num_kways from data features
        """
        prng = np.random.RandomState(seed)
        mixed_vars = [v for v in self.attrs]
        proj = [p for p in itertools.combinations(mixed_vars, k)]
        if num_kways >= 0 and len(proj) > num_kways:
            proj = [proj[i] for i in prng.choice(len(proj), num_kways, replace=False)]

        return proj

    def random_k_way_categorical(self, num_kways=-1, k=3, seed=0):
        """
        Get k-way combinations using only categorical features

        @param domain: shape of the data
        @param num_kways: is the number of categorical feature combinations, if -1 then return all k-way combinations.
        @param k: k-way combinations
        @param seed:  random seed
        @return: a list of  k-way tuples of size num_kways from data features
        """
        prng = np.random.RandomState(seed)

        # cat_vars = [v for v in self.attrs if self[v] > 1]
        cat_vars = self.get_cat_cols()
        proj = [p for p in itertools.combinations(cat_vars, k)]
        if num_kways >= 0 and len(proj) > num_kways:
            proj = [proj[i] for i in prng.choice(len(proj), num_kways, replace=False)]

        return proj

    def random_k_way_continuous(self, num_kways=-1, k=1, seed=0):
        """
        Get k-way combinations using only continuous features

        @param domain: shape of the data
        @param num_kways: is the number of continuous feature combinations, if -1 then return all k-way combinations.
        @param k: k-way combinations
        @param seed:  random seed
        @return: a list of  k-way tuples of size num_kways from data features
        """
        prng = np.random.RandomState(seed)
        continuous_attrs = self.get_cont_cols()
        proj = [p for p in itertools.combinations(continuous_attrs, k)]
        if num_kways >= 0 and len(proj) > num_kways:
            proj = [proj[i] for i in prng.choice(len(proj), num_kways, replace=False)]
        return proj
