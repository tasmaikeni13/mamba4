"""Deliberately sequential dense-solve conformance oracle; never production."""

import jax
import jax.numpy as jnp


def dense_reference(keys, values, queries, beta, decay, epsilon, floor):
    """Direct sequential dense precision, separate from chunk/Cholesky code."""
    batch, time, heads, dim = keys.shape
    lam = jnp.broadcast_to(jnp.asarray(decay), (batch, time, heads))
    eps = jnp.broadcast_to(jnp.asarray(epsilon), (heads,))
    eye = jnp.eye(dim)
    prior_diagonal = (
        eps[:, None] * jnp.asarray(decay)[:, None] ** -jnp.arange(dim)
        if floor == "cyclic"
        else eps[:, None] * jnp.ones((heads, dim))
    )
    prior = jnp.broadcast_to(
        prior_diagonal[..., :, None] * eye, (batch, heads, dim, dim)
    )
    a, c = prior, jnp.zeros((batch, heads, values.shape[-1], dim))
    ys, variances = [], []
    for t in range(time):
        k, v, b, gate = keys[:, t], values[:, t], beta[:, t], lam[:, t]
        a = (
            gate[..., None, None] * a
            + b[..., None, None] * k[..., :, None] * k[..., None, :]
        )
        c = (
            gate[..., None, None] * c
            + b[..., None, None] * v[..., :, None] * k[..., None, :]
        )
        if floor == "fixed":
            injection = (1 - gate)[..., None, None] * eps[None, :, None, None] * eye
        else:
            amplitude = eps * (jnp.asarray(decay) ** (-(dim - 1)) - jnp.asarray(decay))
            coordinate = jax.nn.one_hot(t % dim, dim)
            injection = amplitude[None, :, None, None] * coordinate[:, None] * eye
        a = a + injection
        prior = gate[..., None, None] * prior + injection
        solved = jnp.linalg.solve(a, queries[:, t, ..., None])[..., 0]
        ys.append(jnp.einsum("bhpd,bhd->bhp", c, solved))
        variances.append(jnp.sum(queries[:, t] * solved, axis=-1))
    return jnp.stack(ys, axis=1), jnp.stack(variances, axis=1), a, c, prior


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
