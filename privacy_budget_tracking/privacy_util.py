import math

import numpy as np
from jax import jit
from jax import random
from jax import vmap


def get_gumbel_scale(rho: float, sensitivity: float) -> float:
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
    return sensitivity * math.sqrt(1 / (2 * rho))


# @jit
def select_noisy_q(
    query_errs: np.array,
    answered_queries: np.array,
    q: int,
    query_select_budget: float,
    sensitivity: float,
) -> np.array:
    """
    Given the errors of a set of queries, find the q noisiest queries that have not been answered yet
    :param query_errs: Errors corresponding to each query
    :param answered_queries: Already answered queries
    :param q: Number of queries to select
    :param query_select_budget: The privacy budget we can spend on this
    :return: the privately chosen q worst queries we haven't answered yet
    """

    gumbel_scale = get_gumbel_scale(query_select_budget, sensitivity=sensitivity)

    noisy_query_errors = query_errs + np.random.gumbel(
        loc=0, scale=gumbel_scale, size=query_errs.size
    )

    noisy_query_errors[answered_queries.astype(int)] = -np.inf
    return np.argpartition(noisy_query_errors, -q)[-q:]


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


def get_zcdp_parameter(epsilon: float, delta: float) -> float:
    """
    Given a target (epsilon, delta)-DP budget, computes the per epoch budget in zCDP units
    Refer to Proposition 1.3 in https://arxiv.org/pdf/1605.02065.pdf
    :param epsilon:
    :param delta:
    :return: privacy budget in zCDP units
    """
    rho_0 = __compute_rho(epsilon=epsilon, delta=delta, compute_negative=True)
    rho_1 = __compute_rho(epsilon=epsilon, delta=delta)
    return rho_0 if rho_0 > 0 else rho_1


def gaussian_mechanism(
    key, computed_statistic, query_select_budget: float, sensitivity: float
):
    gaussian_sd = math.sqrt((sensitivity**2) / (2 * query_select_budget))
    rand_noise = generate_random_noise(key, computed_statistic.shape)

    return computed_statistic + (gaussian_sd * rand_noise)


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
