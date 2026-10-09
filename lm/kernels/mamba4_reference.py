"""Deliberately sequential dense-solve conformance oracle; never production."""

import jax.numpy as jnp


def selective_dense_reference(keys, values, queries, beta, decay, floor):
    """Sequential fixed-floor recurrence with token gates and dense solves.

    S_t = decay_t S_(t-1) + beta_t k k^T, C_t likewise, A_t = S_t + diag(floor).
    The floor is never discounted, so this oracle is independent of chunking,
    lane layout and the custom solve derivative used in production.
    """
    batch, time, heads, dim = keys.shape
    floor = jnp.broadcast_to(jnp.asarray(floor), (heads, dim))
    evidence = jnp.zeros((batch, heads, dim, dim))
    cross = jnp.zeros((batch, heads, values.shape[-1], dim))
    reads, variances = [], []
    for t in range(time):
        k, v, b, gate = keys[:, t], values[:, t], beta[:, t], decay[:, t]
        evidence = (
            gate[..., None, None] * evidence
            + b[..., None, None] * k[..., :, None] * k[..., None, :]
        )
        cross = (
            gate[..., None, None] * cross
            + b[..., None, None] * v[..., :, None] * k[..., None, :]
        )
        precision = evidence + floor[None, :, :, None] * jnp.eye(dim)
        solved = jnp.linalg.solve(precision, queries[:, t, ..., None])[..., 0]
        reads.append(jnp.einsum("bhpd,bhd->bhp", cross, solved))
        variances.append(jnp.sum(queries[:, t] * solved, axis=-1))
    return jnp.stack(reads, axis=1), jnp.stack(variances, axis=1), evidence, cross
