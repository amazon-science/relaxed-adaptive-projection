import jax.numpy as np
from jax import random, jit, nn, vmap, partial
import numpy as onp
from scipy.stats import norm
import math
import datasets
from jax.ops import index_update

data_sources = {
    "toy_binary": datasets.toy_binary.ToyBinary,
    "adult": datasets.adult.Adult,
    "loans": datasets.loans.Loans,
}


def init_D_prime(selection, n_prime, d, D=False, interval=None):
    """
    selection: text
    n_prime: int, number of samples for Dprime
    d: int, number of features in Dprime
    D: true data, only needed if near_origin is selected
    """
    if selection == "random":
        Dprime = 2 * (onp.random.random((n_prime, d)) - 0.5)
    elif selection == "rand_interval":
        a, b = interval
        Dprime = (b - a) * onp.random.random((n_prime, d)) + a
    elif selection == "near_origin":
        Dprime = D + 0.05 * onp.random.randn(n_prime, d)
    else:
        raise ValueError(
            "Supported selections are 'random', 'randomunit', and 'near_origin'"
        )
    return Dprime


# Takes as input a GDP parameter mu and a target parameter delta < 1, and returns eps such that a mu-GDP
# algorithm is (eps,delta)-DP.
def GDP_to_DP(mu, delta):
    def deltaval(eps):
        return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(
            -eps / mu - mu / 2
        )

    if delta <= 0:  # No finite epsilon value is possible
        return math.inf
    lower = 0.0
    upper = 500.0
    while (
        abs(deltaval((lower + upper) / 2) - delta) >= delta / 10
        and (lower + upper / 2) > 0
    ):  # binary search to get within (1+-1/10)*delta
        # print("Delta: ", deltaval((lower+upper)/2), "Eps value: ", (lower+upper)/2)
        if deltaval((lower + upper) / 2) < delta:
            upper = (lower + upper) / 2
        else:
            lower = (lower + upper) / 2
    return (lower + upper) / 2


def DP_to_GDP(epsilon, delta):
    """
    Given a target (epsilon-delta), returns mu such that a mu-GDP algorithm is (eps,delta)-DP
    """
    if epsilon <= 0:  # No finite mu value is possible
        return math.inf

    lower = 10 ** -6
    upper = 50.0

    mid = (lower + upper) / 2
    while (
        abs(GDP_to_DP(mu=mid, delta=delta) - epsilon) >= (epsilon * 10 ** -2)
        and mid > 0
    ):
        target_mu = GDP_to_DP(mu=mid, delta=delta)
        if target_mu < epsilon:
            lower = mid
        else:
            upper = mid

        mid = (lower + upper) / 2
        print("Mu-GDP value", mid)

    return mid


def l2_loss_fn(Dprime, target_statistics, statistic_fn):
    # np.linalg.norm returns the Frobenius norm for matrix input or L2 norm of the vector when ord is not provided
    return np.linalg.norm(statistic_fn(Dprime) - target_statistics)


def jit_loss_fn(statistic_fn, norm=None, lambda_l1=0):

    if norm == "L2":
        ord_norm = 2
    elif norm == "Linfty":
        ord_norm = np.inf
    else:
        ord_norm = 5

    @jit
    def compute_loss_fn(Dprime, target_statistics):
        if norm == "LogExp":
            return np.log(
                np.exp(statistic_fn(Dprime) - target_statistics).sum()
            ) + lambda_l1 * np.linalg.norm(Dprime, 1)
        else:
            return np.linalg.norm(
                statistic_fn(Dprime) - target_statistics, ord=ord_norm
            ) + lambda_l1 * np.linalg.norm(Dprime, 1)

    return compute_loss_fn


@jit
def gumbel_softmax_sample(logits, key, temp=1):
    """
    Draw a sample from the Gumbel-Softmax distribution
    """

    y = logits + random.gumbel(key=key, shape=logits.shape)
    return nn.softmax(y / temp, axis=-1)


@jit
def gumbel_softmax(logits, key, temperature=1.0):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    """

    logits = nn.log_softmax(logits, axis=-1)

    y = gumbel_softmax_sample(logits, key=key, temp=temperature)

    # if hard:
    #   shape = y.shape
    #   ind = y.argmax(-1)
    #   y_hard = index_update(np.zeros_like(y), [np.arange(shape[0]), ind], 1)
    #   y = y_hard

    return y


@jit
def project_gumbel_softmax(D, feats_idx, key, temperature=1.0):

    # D = np.log(D)

    return np.hstack(gumbel_softmax(D[:, q], key, temperature) for q in feats_idx)


@jit
def sparsemax(logits):
    """forward pass for sparsemax
    this will process a 2d-array $logits, where axis 1 (each row) is assumed to be
    the logits-vector.
    """

    # sort logits
    z_sorted = np.sort(logits, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, logits.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = logits.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(logits)
    tau_sum = z_cumsum[np.arange(0, logits.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, logits - tau_z)


@jit
def sparsemax_project(D, feats_idx):

    return np.hstack(sparsemax(D[:, q]) for q in feats_idx)


def ohe_to_categorical(D, feats_idx):
    return np.vstack(np.argwhere(D[:, feat] == 1)[:, 1] for feat in feats_idx).T


def randomized_rounding(D, feats_idx, key, oversample=1):

    return np.hstack(
        np.vstack(
            nn.one_hot(
                random.choice(
                    key, a=len(probs), shape=(oversample, 1), p=probs
                ).squeeze(),
                len(probs),
            )
            for probs in D[:, feat]
        )
        for feat in feats_idx
    )


def compute_sigma(k, sens, epsilon, delta):
    if delta == 0:
        return (2 * k * sens) / epsilon
    else:
        return (np.sqrt(32 * (2 * k) * np.log(delta ** -1)) * sens) / epsilon


def numeric_sparse(queries, skip_idxs, k, T, epsilon, sens, delta=0):

    count = 0
    idxs = onp.array([], onp.int)
    pos = 0

    sigma = compute_sigma(k, epsilon, sens, delta)

    T_hat = T + onp.random.laplace(loc=0, scale=sigma)

    while pos < len(queries) and len(idxs) < k:

        if pos in skip_idxs:
            pos += 1
            continue

        v_i = onp.random.laplace(loc=0, scale=2 * sigma)

        if queries[pos] + v_i >= T_hat:
            idxs = onp.append(idxs, pos)
            count += 1
            T_hat = T + onp.random.laplace(loc=0, scale=sigma)

        pos += 1

    return idxs
