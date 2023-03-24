from jax import jit
from jax import nn
from jax import numpy as jnp
from jax import random


@jit
def sparsemax(logits):
    """forward pass for sparsemax
    this will process a 2d-array $logits, where axis 1 (each row) is assumed to be
    the logits-vector.
    """

    # sort logits
    z_sorted = jnp.sort(logits, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = jnp.cumsum(z_sorted, axis=1)
    k = jnp.arange(1, logits.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = logits.shape[1] - jnp.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(logits)
    tau_sum = z_cumsum[jnp.arange(0, logits.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return jnp.maximum(0, logits - tau_z)


@jit
def softmax_project(D, feats_idx):
    return jnp.hstack(
        nn.softmax(D[:, q], axis=1) if len(q) > 1 else D[:, q] for q in feats_idx
    )


@jit
def sparsemax_project(D, feats_idx):
    return jnp.hstack(
        sparsemax(D[:, q]) if len(q) > 1 else D[:, q] for q in feats_idx  # after
    )


@jit
def clip_continuous(D, feats_idx, vmin, vmax):
    return jnp.hstack(
        D[:, q] if len(q) > 1 else jnp.clip(D[:, q], vmin, vmax) for q in feats_idx
    )


def initialize_synthetic_dataset(
    key: jnp.ndarray, num_generated_points, data_dimension
):
    shape = (num_generated_points, data_dimension)
    random_initial = random.uniform(key=key, shape=shape)
    return random_initial
