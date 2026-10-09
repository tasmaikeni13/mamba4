"""Fused Gaussian sufficient-statistic scans and independent protected QR banks.

The prefill path uses chunk summaries and batched Cholesky solves. It has
O(T d**3) factor work, unlike the separate quadratic cyclic decode update.
All evidence, factors, and solves are float32 even under bf16 model projections.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular


class GaussianState(NamedTuple):
    precision: jax.Array
    cross: jax.Array
    prior: jax.Array
    step: jax.Array


class GaussianResult(NamedTuple):
    output: jax.Array
    variance: jax.Array
    state: GaussianState


def _gates(x, shape):
    """Broadcast scalar, per-head, or full token gates to (batch,time,heads)."""
    x = jnp.asarray(x, jnp.float32)
    if x.ndim == 2 and x.shape == (shape[0], shape[2]):
        x = x[:, None, :]
    return jnp.broadcast_to(x, shape)


def _affine_combine(earlier, later):
    l0, x0 = earlier
    l1, x1 = later
    return l1 * l0, x1 + l1 * x0


def _chunked_prefix(decay, increments, initial, chunk_size):
    """Two-level associative prefix; summaries carry chronology across chunks."""
    batch, time, heads, rows, dim = increments.shape
    padded = ((time + chunk_size - 1) // chunk_size) * chunk_size
    count = padded // chunk_size
    pad = padded - time
    lam = jnp.pad(decay, ((0, 0), (0, pad), (0, 0)), constant_values=1)
    inc = jnp.pad(increments, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)))
    lam = lam.reshape(batch, count, chunk_size, heads, 1, 1)
    inc = inc.reshape(batch, count, chunk_size, heads, rows, dim)
    local_lam, local_inc = lax.associative_scan(_affine_combine, (lam, inc), axis=2)
    block_lam, block_inc = lax.associative_scan(
        _affine_combine, (local_lam[:, :, -1], local_inc[:, :, -1]), axis=1
    )
    end_states = block_inc + block_lam * initial[:, None]
    starts = jnp.concatenate((initial[:, None], end_states[:, :-1]), axis=1)
    prefixes = local_inc + local_lam * starts[:, :, None]
    return prefixes.reshape(batch, padded, heads, rows, dim)[:, :time]


def gaussian_memory_prefix(
    keys,
    values,
    queries,
    beta,
    decay,
    epsilon,
    *,
    floor="cyclic",
    chunk_size=64,
    initial=None,
):
    """Inclusive causal ridge reads, plus conditional latent query variance.

    keys/queries: [B,T,H,D], values: [B,T,H,P]; gates: scalar, [H], or
    [B,T,H]. Cyclic decay must be scalar or per-head and constant over time.
    Arbitrary token gates explicitly select floor='fixed'; that path refactors
    the full precision and retains the fixed epsilon-I prior on every step.
    Epsilon and constant cyclic decay remain differentiable parameters.
    """
    keys, values, queries = (
        jnp.asarray(a, jnp.float32) for a in (keys, values, queries)
    )
    batch, time, heads, dim = keys.shape
    if queries.shape != keys.shape or values.shape[:3] != keys.shape[:3]:
        raise ValueError("Mamba4 keys, values and queries have incompatible shapes")
    if floor not in ("cyclic", "fixed"):
        raise ValueError("floor must be 'cyclic' or 'fixed'")
    raw_decay = jnp.asarray(decay, jnp.float32)
    if floor == "cyclic" and raw_decay.ndim > 1:
        raise ValueError(
            "Variable cyclic gates require the explicit fixed-floor fallback"
        )
    shape = (batch, time, heads)
    lam, weight = _gates(raw_decay, shape), _gates(beta, shape)
    eps = jnp.broadcast_to(jnp.asarray(epsilon, jnp.float32), (heads,))
    eye = jnp.eye(dim, dtype=jnp.float32)
    if initial is None:
        if floor == "cyclic":
            constant_lam = jnp.broadcast_to(raw_decay, (heads,))
            diagonal = eps[:, None] * constant_lam[:, None] ** (-jnp.arange(dim))
        else:
            diagonal = eps[:, None] * jnp.ones((heads, dim))
        prior = jnp.broadcast_to(diagonal[:, :, None] * eye, (batch, heads, dim, dim))
        state = GaussianState(
            prior,
            jnp.zeros((batch, heads, values.shape[-1], dim)),
            prior,
            jnp.array(0, jnp.int32),
        )
    else:
        state = initial
    write = weight[..., None, None] * keys[..., :, None] * keys[..., None, :]
    cross = weight[..., None, None] * values[..., :, None] * keys[..., None, :]
    if floor == "fixed":
        injection = (1 - lam)[..., None, None] * eps[None, None, :, None, None] * eye
    else:
        constant_lam = jnp.broadcast_to(raw_decay, (heads,))
        amplitude = eps * (constant_lam ** (-(dim - 1)) - constant_lam)
        coordinates = (jnp.arange(time) + state.step) % dim
        diagonal = jax.nn.one_hot(coordinates, dim, dtype=jnp.float32)
        injection = (
            amplitude[None, None, :, None, None]
            * diagonal[None, :, None, :, None]
            * eye
        )
    increments = jnp.concatenate((write + injection, cross), axis=-2)
    initial_matrix = jnp.concatenate((state.precision, state.cross), axis=-2)
    prefixes = _chunked_prefix(lam, increments, initial_matrix, chunk_size)
    precision, cross = prefixes[..., :dim, :], prefixes[..., dim:, :]
    # Symmetrize roundoff only; no numerical jitter that would change the prior.
    precision = (precision + jnp.swapaxes(precision, -1, -2)) * 0.5
    factor = jnp.linalg.cholesky(precision)
    y = solve_triangular(factor, queries[..., None], lower=True)
    y = solve_triangular(jnp.swapaxes(factor, -1, -2), y, lower=False)[..., 0]
    output = jnp.einsum("bthpd,bthd->bthp", cross, y, precision=lax.Precision.HIGHEST)
    variance = jnp.einsum("bthd,bthd->bth", queries, y, precision=lax.Precision.HIGHEST)
    total_decay = jnp.prod(lam, axis=1)
    if floor == "fixed":
        final_prior = (
            total_decay[..., None, None] * state.prior
            + (1 - total_decay)[..., None, None] * eps[None, :, None, None] * eye
        )
    else:
        ages = jnp.arange(time - 1, -1, -1)
        weights = constant_lam[None, :] ** ages[:, None]
        injected_diagonal = jnp.einsum(
            "th,td,h->hd", weights, diagonal, amplitude, precision=lax.Precision.HIGHEST
        )
        final_prior = (
            total_decay[..., None, None] * state.prior
            + injected_diagonal[None, :, :, None] * eye
        )
    final = GaussianState(
        precision[:, -1], cross[:, -1], final_prior, state.step + time
    )
    return GaussianResult(output, variance, final)


def chol_rank1(lower, vector):
    """Batched stable positive rank-one Cholesky update; O(d**2) work."""
    dim = vector.shape[-1]

    def body(i, carry):
        factor, x = carry
        old_diag = factor[..., i, i]
        radius = jnp.hypot(old_diag, x[..., i])
        cosine, sine = radius / old_diag, x[..., i] / old_diag
        column = (factor[..., :, i] + sine[..., None] * x) / cosine[..., None]
        active = jnp.arange(dim) > i
        column = jnp.where(active, column, factor[..., :, i])
        column = column.at[..., i].set(radius)
        factor = factor.at[..., :, i].set(column)
        new_x = cosine[..., None] * x - sine[..., None] * column
        x = jnp.where(active, new_x, x)
        return factor, x

    return lax.fori_loop(0, dim, body, (lower, vector))[0]


def cyclic_decode(state, lower, key, value, query, beta, decay, epsilon):
    """One constant-gate cyclic write/read with two quadratic factor updates."""
    dim = key.shape[-1]
    lam = jnp.broadcast_to(jnp.asarray(decay, jnp.float32), key.shape[:-1])
    eps = jnp.broadcast_to(jnp.asarray(epsilon, jnp.float32), key.shape[:-1])
    weight = jnp.broadcast_to(jnp.asarray(beta, jnp.float32), key.shape[:-1])
    amplitude = eps * (lam ** (-(dim - 1)) - lam)
    phantom = jax.nn.one_hot(state.step % dim, dim) * jnp.sqrt(amplitude)[..., None]
    lower = chol_rank1(
        lower * jnp.sqrt(lam)[..., None, None], key * jnp.sqrt(weight)[..., None]
    )
    lower = chol_rank1(lower, phantom)
    prior = (
        lam[..., None, None] * state.prior
        + phantom[..., :, None] * phantom[..., None, :]
    )
    precision = (
        lam[..., None, None] * state.precision
        + weight[..., None, None] * key[..., :, None] * key[..., None, :]
        + phantom[..., :, None] * phantom[..., None, :]
    )
    cross = (
        lam[..., None, None] * state.cross
        + weight[..., None, None] * value[..., :, None] * key[..., None, :]
    )
    y = solve_triangular(lower, query[..., None], lower=True)
    y = solve_triangular(jnp.swapaxes(lower, -1, -2), y, lower=False)[..., 0]
    output = jnp.einsum("...pd,...d->...p", cross, y, precision=lax.Precision.HIGHEST)
    variance = jnp.sum(query * y, axis=-1)
    return GaussianResult(
        output, variance, GaussianState(precision, cross, prior, state.step + 1)
    ), lower


class ProtectedBank(NamedTuple):
    """Padded thin QR cache with isolated values, retained IDs, and eligibility."""

    keys: jax.Array
    values: jax.Array
    basis: jax.Array
    triangular: jax.Array
    ids: jax.Array
    valid: jax.Array


class ProtectedCascadeState(NamedTuple):
    banks: ProtectedBank
    counts: jax.Array
    merges: jax.Array


class ProtectedResult(NamedTuple):
    output: jax.Array
    state: ProtectedCascadeState


def _empty_bank(leading, dim, value_dim, budget):
    return ProtectedBank(
        jnp.zeros((*leading, dim, budget), jnp.float32),
        jnp.zeros((*leading, value_dim, budget), jnp.float32),
        jnp.zeros((*leading, dim, budget), jnp.float32),
        jnp.broadcast_to(
            jnp.eye(budget, dtype=jnp.float32), (*leading, budget, budget)
        ),
        jnp.full((*leading, budget), -1, jnp.int32),
        jnp.zeros((*leading, budget), jnp.bool_),
    )


def _insert_anchor(bank, key, value, item_id, tolerance=1e-5):
    """Chronological twice-reorthogonalized QR selection, differentiable in-rank.

    Discrete acceptance/reselection is a piecewise-constant policy; no invented
    gradient through integer IDs or rank decisions is supplied.
    """
    budget = bank.valid.shape[-1]
    count = jnp.sum(bank.valid, axis=-1)
    coefficients = jnp.einsum(
        "...da,...d->...a", bank.basis, key, precision=lax.Precision.HIGHEST
    )
    residual = key - jnp.einsum(
        "...da,...a->...d", bank.basis, coefficients, precision=lax.Precision.HIGHEST
    )
    correction = jnp.einsum(
        "...da,...d->...a", bank.basis, residual, precision=lax.Precision.HIGHEST
    )
    residual -= jnp.einsum(
        "...da,...a->...d", bank.basis, correction, precision=lax.Precision.HIGHEST
    )
    coefficients += correction
    residual_squared = jnp.sum(residual * residual, axis=-1)
    norm = jnp.sqrt(jnp.maximum(residual_squared, tolerance * tolerance))
    accepted = (count < budget) & (residual_squared > tolerance * tolerance)
    slot = jax.nn.one_hot(jnp.minimum(count, budget - 1), budget)
    mask = slot * accepted[..., None]
    q = residual / jnp.maximum(norm[..., None], tolerance)
    r = coefficients + slot * norm[..., None]
    keys = bank.keys * (1 - mask[..., None, :]) + key[..., :, None] * mask[..., None, :]
    values = (
        bank.values * (1 - mask[..., None, :])
        + value[..., :, None] * mask[..., None, :]
    )
    basis = bank.basis * (1 - mask[..., None, :]) + q[..., :, None] * mask[..., None, :]
    triangular = (
        bank.triangular * (1 - mask[..., None, :])
        + r[..., :, None] * mask[..., None, :]
    )
    ids = jnp.where(
        mask.astype(bool), jnp.broadcast_to(item_id, count.shape)[..., None], bank.ids
    )
    valid = bank.valid | mask.astype(bool)
    return ProtectedBank(keys, values, basis, triangular, ids, valid)


def select_protected(keys, values, ids, budget, *, tolerance=1e-5):
    """Select <=budget independent chronological anchors; keys [...,N,D]."""
    if not 1 <= budget <= keys.shape[-1]:
        raise ValueError("Protected anchor budget must be between 1 and key_dim")
    bank = _empty_bank(keys.shape[:-2], keys.shape[-1], values.shape[-1], budget)
    # Inputs are chronological; caller sorts merged candidate IDs when necessary.
    xs = (
        jnp.moveaxis(keys, -2, 0),
        jnp.moveaxis(values, -2, 0),
        jnp.moveaxis(ids, -1, 0),
    )

    def step(carry, x):
        key, value, item_id = x
        candidate = _insert_anchor(carry, key, value, item_id, tolerance)
        candidate = jax.tree.map(
            lambda a, b: jnp.where((item_id >= 0)[..., None, None], a, b)
            if a.ndim == item_id.ndim + 2
            else jnp.where((item_id >= 0)[..., None], a, b),
            candidate,
            carry,
        )
        return candidate, None

    return lax.scan(step, bank, xs)[0]


def protected_read(bank, queries):
    """Exact retained-bank linear read, without posterior-variance semantics.

    queries [...,D] has the same leading dimensions as the bank. Background
    Gaussian evidence never enters this QR cache, including after a merge.
    """
    coefficients = jnp.einsum(
        "...da,...d->...a", bank.basis, queries, precision=lax.Precision.HIGHEST
    )
    coefficients = solve_triangular(
        bank.triangular, coefficients[..., None], lower=False
    )[..., 0]
    return jnp.einsum(
        "...pa,...a->...p", bank.values, coefficients, precision=lax.Precision.HIGHEST
    )


def protected_route_by_id(state, queries, query_ids):
    """Route by an explicit retained ID; missing IDs return zero and False.

    This contract costs an all-bank ID search. It does not claim that learned
    query geometry can infer every correct ID, especially under conflicts.
    Query leading shape is [B,H]; cascade bank axes are [level,slot,B,H].
    """
    banks = state.banks
    matches = jnp.any(
        (banks.ids == query_ids[None, None, ..., None]) & banks.valid, axis=-1
    )
    active = jnp.arange(2)[None, :] < state.counts[:, None]
    matches &= active[..., None, None]
    reads = protected_read(
        banks, jnp.broadcast_to(queries, (*banks.basis.shape[:-2], queries.shape[-1]))
    )
    output = jnp.sum(reads * matches[..., None], axis=(0, 1))
    return output, jnp.any(matches, axis=(0, 1))


def _merge_banks(left, right, budget):
    keys = jnp.concatenate(
        (jnp.swapaxes(left.keys, -1, -2), jnp.swapaxes(right.keys, -1, -2)), axis=-2
    )
    values = jnp.concatenate(
        (jnp.swapaxes(left.values, -1, -2), jnp.swapaxes(right.values, -1, -2)), axis=-2
    )
    ids = jnp.concatenate((left.ids, right.ids), axis=-1)
    # Invalid padding goes last, retained IDs remain in chronological order.
    order = jnp.argsort(
        jnp.where(ids >= 0, ids, jnp.iinfo(jnp.int32).max), axis=-1, stable=True
    )
    keys = jnp.take_along_axis(keys, order[..., None], axis=-2)
    values = jnp.take_along_axis(values, order[..., None], axis=-2)
    ids = jnp.take_along_axis(ids, order, axis=-1)
    return select_protected(keys, values, ids, budget)


def _cascade_append(state, bank, budget):
    """Redundant counter: third bank merges two oldest and retains newest."""
    carry_bank, pending = bank, jnp.array(True)
    banks, counts, merges = state
    for level in range(counts.shape[0]):
        old0 = jax.tree.map(lambda x: x[level, 0], banks)
        old1 = jax.tree.map(lambda x: x[level, 1], banks)
        full = counts[level] == 2
        merged = lax.cond(
            pending & full,
            lambda _: _merge_banks(old0, old1, budget),
            lambda _: carry_bank,
            operand=None,
        )
        target = jnp.where(full, 0, counts[level])
        banks = jax.tree.map(
            lambda x, new: x.at[level, target].set(
                jnp.where(pending, new, x[level, target])
            ),
            banks,
            carry_bank,
        )
        counts = counts.at[level].set(
            jnp.where(pending, jnp.where(full, 1, counts[level] + 1), counts[level])
        )
        merges += (pending & full).astype(jnp.int32)
        pending = pending & full
        carry_bank = merged
    return ProtectedCascadeState(banks, counts, merges)


def protected_cascade_memory(
    keys,
    values,
    queries,
    *,
    budget=4,
    block_size=64,
    route_strength=4.0,
    tolerance=1e-5,
):
    """Causal local QR plus all-bank reads from a redundant protected cascade.

    Every query reads all occupied earlier banks and its current partial block.
    Softmax routing uses span geometry and is trainable but is not an exact-ID
    guarantee. Exactness is exposed separately by protected_route_by_id.
    This counts QR selection, merge reselection, IDs, and all-bank solves.
    """
    keys, values, queries = (
        jnp.asarray(a, jnp.float32) for a in (keys, values, queries)
    )
    batch, time, heads, dim = keys.shape
    value_dim = values.shape[-1]
    if not 1 <= budget <= dim:
        raise ValueError("Protected anchor budget must be between 1 and key_dim")
    nblocks = (time + block_size - 1) // block_size
    padded = nblocks * block_size
    levels = nblocks.bit_length()
    state = ProtectedCascadeState(
        _empty_bank((levels, 2, batch, heads), dim, value_dim, budget),
        jnp.zeros((levels,), jnp.int32),
        jnp.array(0, jnp.int32),
    )

    def blocks(x):
        x = jnp.pad(x, ((0, 0), (0, padded - time), (0, 0), (0, 0)))
        return x.reshape(batch, nblocks, block_size, heads, x.shape[-1]).transpose(
            1, 2, 0, 3, 4
        )

    ids = jnp.arange(padded).reshape(nblocks, block_size)
    ids = jnp.where(ids < time, ids, -1)
    strength = jnp.broadcast_to(jnp.asarray(route_strength, jnp.float32), (heads,))

    def block_step(cascade, xs):
        key_block, value_block, query_block, id_block = xs
        local = _empty_bank((batch, heads), dim, value_dim, budget)

        def anchor_step(bank, x):
            key, value, item_id = x
            new_bank = _insert_anchor(bank, key, value, item_id, tolerance)
            new_bank = jax.tree.map(
                lambda new, old: jnp.where(item_id >= 0, new, old), new_bank, bank
            )
            return new_bank, new_bank

        local, local_prefixes = lax.scan(
            anchor_step, local, (key_block, value_block, id_block)
        )
        bank_count = levels * 2
        old_banks = jax.tree.map(
            lambda x: x.reshape(bank_count, *x.shape[2:]), cascade.banks
        )
        # Bank axis is explicit; all-bank work remains visible to XLA and ledger.
        queries_old = jnp.broadcast_to(
            query_block[:, None], (block_size, bank_count, batch, heads, dim)
        )
        expanded_old = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None], (block_size, *x.shape)), old_banks
        )
        old_reads = protected_read(expanded_old, queries_old)
        local_reads = protected_read(local_prefixes, query_block)
        old_geometry = jnp.sum(
            jnp.einsum(
                "tmbhda,tbhd->tmbha",
                expanded_old.basis,
                query_block,
                precision=lax.Precision.HIGHEST,
            )
            ** 2,
            axis=-1,
        )
        local_geometry = jnp.sum(
            jnp.einsum(
                "tbhda,tbhd->tbha",
                local_prefixes.basis,
                query_block,
                precision=lax.Precision.HIGHEST,
            )
            ** 2,
            axis=-1,
        )
        geometry = jnp.concatenate((old_geometry, local_geometry[:, None]), axis=1)
        norm = jnp.maximum(jnp.sum(query_block**2, axis=-1), 1e-6)
        scores = geometry / norm[:, None] * strength[None, None, None, :]
        old_valid = jnp.any(old_banks.valid, axis=-1)
        occupied = (jnp.arange(2)[None, :] < cascade.counts[:, None]).reshape(
            bank_count
        )
        old_valid &= occupied[:, None, None]
        valid = jnp.concatenate(
            (
                jnp.broadcast_to(old_valid[None], (block_size, *old_valid.shape)),
                jnp.any(local_prefixes.valid, axis=-1)[:, None],
            ),
            axis=1,
        )
        scores = jnp.where(valid, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=1) * valid
        reads = jnp.concatenate((old_reads, local_reads[:, None]), axis=1)
        output = jnp.sum(weights[..., None] * reads, axis=1)
        return _cascade_append(cascade, local, budget), output

    state, outputs = lax.scan(
        jax.checkpoint(block_step),
        state,
        (blocks(keys), blocks(values), blocks(queries), ids),
    )
    output = outputs.transpose(2, 0, 1, 3, 4).reshape(batch, padded, heads, value_dim)[
        :, :time
    ]
    return ProtectedResult(output, state)


def gaussian_initial_state(
    batch, heads, key_dim, value_dim, epsilon, decay, *, floor="cyclic"
):
    """Initialize the exact prior and cached decode factor with learned values."""
    eps = jnp.broadcast_to(jnp.asarray(epsilon, jnp.float32), (heads,))
    lam = jnp.broadcast_to(jnp.asarray(decay, jnp.float32), (heads,))
    exponents = -jnp.arange(key_dim) if floor == "cyclic" else jnp.zeros((key_dim,))
    diagonal = eps[:, None] * lam[:, None] ** exponents
    prior = jnp.broadcast_to(
        diagonal[..., :, None] * jnp.eye(key_dim), (batch, heads, key_dim, key_dim)
    )
    state = GaussianState(
        prior,
        jnp.zeros((batch, heads, value_dim, key_dim)),
        prior,
        jnp.array(0, jnp.int32),
    )
    lower = jnp.broadcast_to(
        jnp.sqrt(diagonal)[..., :, None] * jnp.eye(key_dim), prior.shape
    )
    return state, lower


class ProtectedDecodeState(NamedTuple):
    cascade: ProtectedCascadeState
    local: ProtectedBank
    step: jax.Array


def protected_initial_state(
    batch, heads, key_dim, value_dim, budget, max_seq_len, block_size
):
    levels = ((max_seq_len + block_size - 1) // block_size).bit_length()
    cascade = ProtectedCascadeState(
        _empty_bank((levels, 2, batch, heads), key_dim, value_dim, budget),
        jnp.zeros((levels,), jnp.int32),
        jnp.array(0, jnp.int32),
    )
    return ProtectedDecodeState(
        cascade,
        _empty_bank((batch, heads), key_dim, value_dim, budget),
        jnp.array(0, jnp.int32),
    )


def protected_decode(
    state, key, value, query, *, budget, block_size, route_strength=4.0
):
    """Cached current QR and all-bank geometry reads; same routing as prefill."""
    local = _insert_anchor(state.local, key, value, state.step)
    banks = state.cascade.banks
    old_reads = protected_read(
        banks, jnp.broadcast_to(query, (*banks.basis.shape[:-2], query.shape[-1]))
    )
    local_reads = protected_read(local, query)
    geometry = jnp.sum(
        jnp.einsum(
            "lsbhda,bhd->lsbha", banks.basis, query, precision=lax.Precision.HIGHEST
        )
        ** 2,
        axis=-1,
    )
    geometry = geometry.reshape(-1, *geometry.shape[2:])
    current = jnp.sum(
        jnp.einsum("bhda,bhd->bha", local.basis, query, precision=lax.Precision.HIGHEST)
        ** 2,
        axis=-1,
    )
    geometry = jnp.concatenate((geometry, current[None]), axis=0)
    valid = jnp.any(banks.valid, axis=-1)
    active = jnp.arange(2)[None, :] < state.cascade.counts[:, None]
    valid &= active[..., None, None]
    valid = jnp.concatenate(
        (valid.reshape(-1, *valid.shape[2:]), jnp.any(local.valid, axis=-1)[None]),
        axis=0,
    )
    norm = jnp.maximum(jnp.sum(query**2, axis=-1), 1e-6)
    scores = geometry / norm[None] * jnp.asarray(route_strength)[None, None]
    weights = jax.nn.softmax(jnp.where(valid, scores, -1e9), axis=0) * valid
    reads = jnp.concatenate(
        (old_reads.reshape(-1, *old_reads.shape[2:]), local_reads[None]), axis=0
    )
    output = jnp.sum(weights[..., None] * reads, axis=0)
    completed = (state.step + 1) % block_size == 0
    cascade = lax.cond(
        completed,
        lambda s: _cascade_append(s, local, budget),
        lambda s: s,
        state.cascade,
    )
    empty = _empty_bank(key.shape[:-1], key.shape[-1], value.shape[-1], budget)
    local = jax.tree.map(
        lambda zero, keep: jnp.where(completed, zero, keep), empty, local
    )
    return output, ProtectedDecodeState(cascade, local, state.step + 1)


def gaussian_memory(
    keys,
    values,
    queries,
    beta,
    decay,
    epsilon,
    *,
    floor="cyclic",
    chunk_size=64,
    initial=None,
):
    """Rematerialized chunk prefill with exact Gaussian evidence and solves.

    Every chunk materializes only its local prefixes. The outer scan retains
    boundary sufficient statistics and rematerializes factor/read work in
    backward; it never needs a full-sequence cross-statistic activation.
    Factorization remains per token, with O(T*d**3) factor work.
    """
    keys, values, queries = (
        jnp.asarray(a, jnp.float32) for a in (keys, values, queries)
    )
    batch, time, heads, dim = keys.shape
    if queries.shape != keys.shape or values.shape[:3] != keys.shape[:3]:
        raise ValueError("Mamba4 keys, values and queries have incompatible shapes")
    if floor not in ("cyclic", "fixed"):
        raise ValueError("floor must be 'cyclic' or 'fixed'")
    raw_decay = jnp.asarray(decay, jnp.float32)
    if floor == "cyclic" and raw_decay.ndim > 1:
        raise ValueError(
            "Variable cyclic gates require the explicit fixed-floor fallback"
        )
    shape = (batch, time, heads)
    lam, weight = _gates(raw_decay, shape), _gates(beta, shape)
    eps = jnp.broadcast_to(jnp.asarray(epsilon, jnp.float32), (heads,))
    eye = jnp.eye(dim, dtype=jnp.float32)
    constant_lam = jnp.broadcast_to(raw_decay, (heads,)) if floor == "cyclic" else None
    state = (
        gaussian_initial_state(
            batch,
            heads,
            dim,
            values.shape[-1],
            eps,
            constant_lam if floor == "cyclic" else 1.0,
            floor=floor,
        )[0]
        if initial is None
        else initial
    )
    padded = ((time + chunk_size - 1) // chunk_size) * chunk_size
    count = padded // chunk_size

    def pack(x, pad_value=0):
        padding = [(0, 0), (0, padded - time)] + [(0, 0)] * (x.ndim - 2)
        x = jnp.pad(x, padding, constant_values=pad_value)
        x = x.reshape(batch, count, chunk_size, *x.shape[2:])
        return jnp.moveaxis(x, 1, 0)

    times = jnp.arange(padded).reshape(count, chunk_size)
    amplitude = (
        eps * (constant_lam ** (-(dim - 1)) - constant_lam)
        if floor == "cyclic"
        else None
    )
    initial_matrix = jnp.concatenate((state.precision, state.cross), axis=-2)

    def block_step(carry, xs):
        key, value, query, beta_block, lam_block, positions = xs
        write = beta_block[..., None, None] * key[..., :, None] * key[..., None, :]
        cross = beta_block[..., None, None] * value[..., :, None] * key[..., None, :]
        if floor == "fixed":
            injection = (
                (1 - lam_block)[..., None, None] * eps[None, None, :, None, None] * eye
            )
        else:
            diagonal = jax.nn.one_hot(
                (positions + state.step) % dim, dim, dtype=jnp.float32
            )
            diagonal *= (positions < time)[:, None]
            injection = (
                amplitude[None, None, :, None, None]
                * diagonal[None, :, None, :, None]
                * eye
            )
        increments = jnp.concatenate((write + injection, cross), axis=-2)
        local_lam, local_inc = lax.associative_scan(
            _affine_combine, (lam_block[..., None, None], increments), axis=1
        )
        prefixes = local_inc + local_lam * carry[:, None]
        precision, cross = prefixes[..., :dim, :], prefixes[..., dim:, :]
        precision = (precision + jnp.swapaxes(precision, -1, -2)) * 0.5
        with jax.default_matmul_precision("highest"):
            factor = jnp.linalg.cholesky(precision)
            solved = solve_triangular(factor, query[..., None], lower=True)
            solved = solve_triangular(
                jnp.swapaxes(factor, -1, -2), solved, lower=False
            )[..., 0]
        output = jnp.einsum(
            "bthpd,bthd->bthp", cross, solved, precision=lax.Precision.HIGHEST
        )
        variance = jnp.einsum(
            "bthd,bthd->bth", query, solved, precision=lax.Precision.HIGHEST
        )
        final = jnp.concatenate((precision[:, -1], cross[:, -1]), axis=-2)
        return final, (output, variance)

    final, (output, variance) = lax.scan(
        jax.checkpoint(block_step),
        initial_matrix,
        (pack(keys), pack(values), pack(queries), pack(weight), pack(lam, 1), times),
    )
    output = jnp.moveaxis(output, 0, 1).reshape(batch, padded, heads, values.shape[-1])[
        :, :time
    ]
    variance = jnp.moveaxis(variance, 0, 1).reshape(batch, padded, heads)[:, :time]
    total_decay = jnp.prod(lam, axis=1)
    if floor == "fixed":
        prior = (
            total_decay[..., None, None] * state.prior
            + (1 - total_decay)[..., None, None] * eps[None, :, None, None] * eye
        )
    else:
        ages = jnp.arange(time - 1, -1, -1)
        weights = constant_lam[None, :] ** ages[:, None]
        diagonal = jax.nn.one_hot(
            (jnp.arange(time) + state.step) % dim, dim, dtype=jnp.float32
        )
        injected = jnp.einsum(
            "th,td,h->hd", weights, diagonal, amplitude, precision=lax.Precision.HIGHEST
        )
        prior = (
            total_decay[..., None, None] * state.prior
            + injected[None, :, :, None] * eye
        )
    final_state = GaussianState(
        final[..., :dim, :], final[..., dim:, :], prior, state.step + time
    )
    return GaussianResult(output, variance, final_state)


def gaussian_diagnostics(state):
    """Small-state conditioning probes; explicit monitoring cost, not every token."""
    with jax.default_matmul_precision("highest"):
        eigenvalues = jnp.linalg.eigvalsh(
            (state.precision + jnp.swapaxes(state.precision, -1, -2)) * 0.5
        )
    minimum, maximum = eigenvalues[..., 0], eigenvalues[..., -1]
    return {
        "precision_min_eigenvalue": jnp.min(minimum),
        "precision_max_eigenvalue": jnp.max(maximum),
        "precision_max_condition": jnp.max(maximum / jnp.maximum(minimum, 1e-30)),
        "state_all_finite": jnp.all(jnp.isfinite(state.precision))
        & jnp.all(jnp.isfinite(state.cross))
        & jnp.all(jnp.isfinite(state.prior)),
        "cross_max_abs": jnp.max(jnp.abs(state.cross)),
        "prior_min_diagonal": jnp.min(jnp.diagonal(state.prior, axis1=-2, axis2=-1)),
        "prior_max_diagonal": jnp.max(jnp.diagonal(state.prior, axis1=-2, axis2=-1)),
    }


def protected_diagnostics(state):
    """Count active resources and QR rank margins; no learned-routing certainty."""
    if isinstance(state, ProtectedDecodeState):
        cascade, local = state.cascade, state.local
    else:
        cascade, local = state, None
    banks = cascade.banks
    active = jnp.arange(2)[None, :] < cascade.counts[:, None]
    valid = banks.valid & active[..., None, None, None]
    diagonal = jnp.diagonal(banks.triangular, axis1=-2, axis2=-1)
    min_diag = jnp.min(jnp.where(valid, diagonal, jnp.inf))
    per_head_anchors = jnp.sum(valid, axis=(0, 1, 4))
    anchors = jnp.sum(valid)
    finite = (
        jnp.all(jnp.isfinite(banks.keys))
        & jnp.all(jnp.isfinite(banks.basis))
        & jnp.all(jnp.isfinite(banks.triangular))
        & jnp.all(jnp.isfinite(banks.values))
    )
    if local is not None:
        finite &= (
            jnp.all(jnp.isfinite(local.keys))
            & jnp.all(jnp.isfinite(local.basis))
            & jnp.all(jnp.isfinite(local.triangular))
            & jnp.all(jnp.isfinite(local.values))
        )
        min_diag = jnp.minimum(
            min_diag,
            jnp.min(
                jnp.where(
                    local.valid,
                    jnp.diagonal(local.triangular, axis1=-2, axis2=-1),
                    jnp.inf,
                )
            ),
        )
        per_head_anchors += jnp.sum(local.valid, axis=-1)
        anchors += jnp.sum(local.valid)
    return {
        "protected_active_banks": jnp.sum(cascade.counts),
        "protected_retained_anchors": anchors,
        "protected_retained_anchors_min": jnp.min(per_head_anchors),
        "protected_retained_anchors_max": jnp.max(per_head_anchors),
        "protected_min_qr_diagonal": min_diag,
        "protected_merges": cascade.merges,
        "protected_all_finite": finite,
    }
