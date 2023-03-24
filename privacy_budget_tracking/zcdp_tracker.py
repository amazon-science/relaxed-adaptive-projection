import math

import numpy as np
from jax import jit
from jax import random
from jax import vmap

from .privacy_budget_exhausted_error import PrivacyBudgetExhaustedError


class zCDPTracker:
    """
    Class to track privacy budget usage using zCDP or zero-Concentrated Differential Privacy
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        epochs: int,
        queries_per_epoch: int,
        num_points: int,
    ):
        """
        :param epsilon:
        :param delta: failure probability
        :param epochs: number of epochs
        :param queries_per_epoch: the number of queries that are selected every epoch. Can be between 1 and size of
        query set |Q|
        :param num_points: the number of points in the dataset
        """

        self.epsilon = epsilon
        self.delta = delta
        self.epochs = epochs
        self.queries_per_epoch = queries_per_epoch
        self.num_points = num_points
        self.budget = self.get_budget_zcdp(epsilon, delta)

        self.sensitivity = 1 / num_points

        self.epochs_elapsed = 0
        self.budget_used = 0

    def reset(self) -> None:
        self.budget_used = 0

    def increment_epochs(self) -> None:
        self.epochs_elapsed += 1

    def enforce_budget(self) -> None:
        if self.budget < self.budget_used:
            raise PrivacyBudgetExhaustedError()

    def budget_per_epoch(self, rebalance: bool = False) -> float:
        """
        Calculate the budget expenditure per epoch, potentially rebalancing if using Sparse/AboveThreshold
        :param rebalance: whether to rebalance the budget expenditure
        :return: privacy budget to expend per epoch/schedule
        """
        if rebalance:
            return (self.budget - self.budget_used) / (
                self.epochs - self.epochs_elapsed
            )
        else:
            return self.budget / self.epochs

    def get_budget_used(self):
        return self.budget_used

    @staticmethod
    def convert_epsilon_to_zcdp(epsilon: float) -> float:
        """
        Convert (epsilon, 0)-DP to zCDP based on Proposition 1.4 in https://arxiv.org/pdf/1605.02065.pdf
        :param epsilon:
        :return: rho
        """
        rho = (epsilon**2) / 2
        return rho

    @staticmethod
    def convert_zcdp_to_eps_delta(rho: float, delta: float) -> float:
        """
        Convert (rho, delta)-zCDP to (epsilon, delta)-DP based on Proposition 1.3 in
        https://arxiv.org/pdf/1605.02065.pdf
        :param rho:
        :param delta:
        :return: epsilon
        """
        epsilon = rho + 2 * math.sqrt(rho * math.log(1 / delta))
        epsilon2 = rho + 2 * math.sqrt(rho * math.log(math.sqrt(math.pi * rho) / delta))
        return min(epsilon, epsilon2)

    def get_gaussian_sd_for_budget(self, rho: float) -> float:
        """
        Return standard deviation for gaussian noise based on rho. Formula is: rho = sens^2 / (2 sigma^2)
        Refer to Proposition 1.6 in https://arxiv.org/pdf/1605.02065.pdf
        :param rho:
        :return: sigma, standard deviation for gaussian noise

        # scale for answering single query.
        sensitivity = 1/n
        for q queries
        sensitivity = sqrt{q}/n
        """
        sigma = math.sqrt((self.sensitivity**2) / (2 * rho))
        return sigma

    def get_laplace_b_for_budget(self, rho: float) -> float:
        """
        Compute parameter b for generating Laplacian noise.
        Using standard epsilon-DP analysis from e.g. https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        and backwards conversion from Prop 1.4 in https://arxiv.org/pdf/1605.02065.pdf
        :param rho:
        :return: lap_b, parameter for generating laplacian noise
        """
        epsilon = math.sqrt(2 * rho)
        lap_b = self.sensitivity / epsilon
        return lap_b

    def __charge_above_threshold(self, lap_b: float, calls: int = 1) -> None:
        """
        Charge privacy budget for computing error values on queries that pass the error threshold
        Noise is from Lap(sensitivity / epsilon)
        :param lap_b: parameter used to generate laplace noise
        :param calls:
        :return:
        """
        epsilon = self.sensitivity / lap_b
        rho = self.convert_epsilon_to_zcdp(epsilon)
        self.budget_used += rho * calls

    def __charge_gaussian_mech(self, rho: float, calls: int = 1) -> None:
        """
        Charge privacy budget using the gaussian mechanism
        rho = sens^2 / (2 sigma^2)
        Proposition 1.6 in https://arxiv.org/pdf/1605.02065.pdf
        :param rho:
        :param calls:
        :return:
        """
        self.budget_used += rho * calls

    def __charge_report_noisy_max(self, rho: float, calls: int = 1) -> None:
        """
        Using Laplace noise, report noisy max
        :param rho: budget used per call
        :param calls:
        :return:
        """
        self.budget_used += rho * calls

    def get_gumbel_scale(self, rho: float, q: int) -> float:
        """
        Given the total budget to spend and the number of indices to return,
        return the scale/beta parameter for the gumbel distribution.

        The scale is determined by a few published results.
        1. The iterative (or peeling) exponential mechanism is epsilon-bounded
        range and hence 2*epsilon-DP (Lemma 4.1 of https://arxiv.org/pdf/1905.04273.pdf)
        2. The iterative exponential mechanism is equivalent to the iterative
        Gumbel mechanism (Lemma 4.2 of https://arxiv.org/pdf/1905.04273.pdf)
        3. The iterative Gumbel mechanism (repeated k-times) is equivalent to
        the one-shot Gumbel mechanism (Corollary 4.1 of https://arxiv.org/pdf/1905.04273.pdf)
        4. Thus, the one-shot Gumbel mechanism is 2*epsilon-DP, inverting this
        to find the scale we find that scale = sqrt(k)/(n * sqrt(2 * rho))
        :param rho: zCDP budget
        :param q: number of queries to select
        :return: beta, parameter for generating Gumbel noise
        """

        return self.sensitivity * math.sqrt(q / (2 * rho))

    def select_noisy_q(
        self,
        query_errs: np.array,
        answered_queries: np.array,
        q: int,
        query_select_budget: float,
    ) -> np.array:
        """
        Given the errors of a set of queries, find the q noisiest queries that have not been answered yet
        :param query_errs: Errors corresponding to each query
        :param answered_queries: Already answered queries
        :param q: Number of queries to select
        :param query_select_budget: The privacy budget we can spend on this
        :return: the privately chosen q worst queries we haven't answered yet
        """

        gumbel_scale = self.get_gumbel_scale(query_select_budget, q)

        self.__charge_report_noisy_max(query_select_budget)
        noisy_query_errors = query_errs + np.random.gumbel(
            loc=0, scale=gumbel_scale, size=query_errs.size
        )

        noisy_query_errors[answered_queries.astype(int)] = -np.inf
        return np.argpartition(noisy_query_errors, -q)[-q:]

    @staticmethod
    def __compute_rho(
        epsilon: float, delta: float, compute_negative: bool = False
    ) -> float:
        """
        Computes the rho value used to express zCDP privacy loss
        :param epsilon:
        :param delta:
        :param compute_negative: If true, computes the negative rho
        :return: rho
        """
        return (
            ((-1) ** compute_negative)
            * 2
            * math.sqrt(epsilon * math.log(1 / delta) + math.log(1 / delta) ** 2)
            + epsilon
            + 2 * math.log(1 / delta)
        )

    @staticmethod
    def get_budget_zcdp(epsilon: float, delta: float) -> float:
        """
        Given a target (epsilon, delta)-DP budget, computes the budget in zCDP units
        Refer to Proposition 1.3 in https://arxiv.org/pdf/1605.02065.pdf
        :param epsilon:
        :param delta:
        :return: privacy budget in zCDP units
        """
        if delta > 0:
            rho_0 = zCDPTracker.__compute_rho(
                epsilon=epsilon, delta=delta, compute_negative=True
            )
            rho_1 = zCDPTracker.__compute_rho(epsilon=epsilon, delta=delta)
            return rho_0 if rho_0 > 0 else rho_1
        else:
            return zCDPTracker.convert_epsilon_to_zcdp(epsilon)

    def get_budget_epoch_zcdp(self, epsilon: float, delta: float) -> float:
        """
        Given a target (epsilon, delta)-DP budget, computes the per epoch budget in zCDP units
        Refer to Proposition 1.3 in https://arxiv.org/pdf/1605.02065.pdf
        :param epsilon:
        :param delta:
        :return: privacy budget in zCDP units
        """
        rho_0 = (
            zCDPTracker.__compute_rho(
                epsilon=epsilon, delta=delta, compute_negative=True
            )
            / self.epochs
        )
        rho_1 = zCDPTracker.__compute_rho(epsilon=epsilon, delta=delta) / self.epochs
        return rho_0 if rho_0 > 0 else rho_1

    def gaussian_mechanism(self, key, statistic_fn, dataset, query_answer_budget):
        gaussian_sd = self.get_gaussian_sd_for_budget(query_answer_budget)
        computed_statistic = statistic_fn(dataset)
        self.__charge_gaussian_mech(
            query_answer_budget, calls=computed_statistic.shape[0]
        )

        rand_noise = zCDPTracker.generate_random_noise(key, computed_statistic.shape)
        return computed_statistic + (gaussian_sd * rand_noise)

    def gaussian_mechanism_v2(self, key, computed_statistic, query_answer_budget):
        gaussian_sd = self.get_gaussian_sd_for_budget(query_answer_budget)
        self.__charge_gaussian_mech(
            query_answer_budget, calls=computed_statistic.shape[0]
        )

        rand_noise = zCDPTracker.generate_random_noise(key, computed_statistic.shape)
        return computed_statistic + (gaussian_sd * rand_noise)

    @staticmethod
    def generate_random_noise(key, shape, sampling_function=random.normal):
        # This function samples from a probability distribution using sampling
        # function, with PRNG key and returns it with specified size. The sampling
        # function MUST take as arguments key and shape
        # By default, this function will use the normal distribution
        # Read: better random numbers generation with subkeys
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#JAX-PRNG

        compiled_sampler = jit(sampling_function, static_argnums=1)

        if len(shape) < 2:
            # no need to split the key for low-dimensional statistic
            return compiled_sampler(key, shape)

        # generate subkeys
        subkeys = random.split(key, shape[0])
        return vmap(compiled_sampler, in_axes=(0, None), out_axes=0)(subkeys, shape[1:])
