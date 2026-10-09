"""Mamba-3 exponential-trapezoidal SSD, including complex phases and MIMO.

The equations and parameter conventions follow state-spaces/mamba at commit
e9594ce1c732d97440f0332fdc43170a2294dbfa. The production implementation uses
batched dot products within each chunk and an XLA-fused scan over chunk states;
it never materializes a recurrent state for every token. All accumulators and
decay/phase calculations use float32, with model-precision matrix operands.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


OFFICIAL_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def rotate(x, angles, pairwise=True):
    """Rotate the last dimension; angle prefix omits any nonrotary coordinates."""
    half = x.shape[-1] // 2
    if x.shape[-1] % 2 or angles.shape[-1] > half:
        raise ValueError("Mamba-3 state dimension must be even and contain angles")
    angles = jnp.pad(
        angles, [(0, 0)] * (angles.ndim - 1) + [(0, half - angles.shape[-1])]
    )
    x32 = x.astype(jnp.float32)
    if pairwise:
        pairs = x32.reshape(*x.shape[:-1], half, 2)
        left, right = pairs[..., 0], pairs[..., 1]
    else:
        left, right = jnp.split(x32, 2, axis=-1)
    cosine, sine = jnp.cos(angles), jnp.sin(angles)
    left, right = left * cosine - right * sine, left * sine + right * cosine
    if pairwise:
        output = jnp.stack((left, right), axis=-1).reshape(x.shape)
    else:
        output = jnp.concatenate((left, right), axis=-1)
    return output.astype(x.dtype)


def _rotary_frame(q, k, dt, angles, q_bias, k_bias, pairwise):
    if q_bias is not None:
        q = (q.astype(jnp.float32) + q_bias).astype(q.dtype)
    if k_bias is not None:
        k = (k.astype(jnp.float32) + k_bias).astype(k.dtype)
    phase = jnp.cumsum(angles.astype(jnp.float32) * dt[..., None], axis=1)
    # Rangewise B/C rotations share a per-head phase; the integrated phase has
    # no modulo, which preserves the correct derivative through every timestep.
    return (
        rotate(q, phase[..., None, :], pairwise),
        rotate(k, phase[..., None, :], pairwise),
        phase,
    )


@partial(jax.jit, static_argnames=("chunk_size", "pairwise"))
def mamba3_chunked_serial(
    q,
    k,
    v,
    adt,
    dt,
    trap_logits,
    angles,
    *,
    chunk_size=64,
    q_bias=None,
    k_bias=None,
    pairwise=True,
):
    """Fused causal SSD returning [B,T,H,R,P] outputs and final [B,H,N,P] state.

    Q/K: [B,T,H,R,N]; V: [B,T,H,R,P]. ADT/DT/trap: [B,T,H];
    angle rates: [B,T,H,A]. R=1 gives SISO; R>1 contracts every write rank
    into one shared state and reads every rank from that state (true MIMO).

    alpha=exp(ADT), beta=alpha*DT*(1-sigmoid(trap)),
    gamma=DT*sigmoid(trap), H_t=alpha*H_(t-1)+beta*K_(t-1)V_(t-1)^T
    +gamma*K_t V_t^T. B/C are in the cumulative rotary frame.
    """
    batch, length, heads, rank, state_dim = q.shape
    if k.shape != q.shape or v.shape[:4] != q.shape[:4]:
        raise ValueError("Mamba-3 Q/K/V dimensions are incompatible")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    q, k, phase = _rotary_frame(q, k, dt, angles, q_bias, k_bias, pairwise)
    del phase
    trap = jax.nn.sigmoid(trap_logits.astype(jnp.float32))
    gamma = dt.astype(jnp.float32) * trap
    previous_weight = dt.astype(jnp.float32) * (1 - trap)
    # Offdiagonal source j has gamma_j + beta_(j+1)/alpha_(j+1).
    # Diagonal uses gamma_j only, hence future trap/dt can never leak into y_j.
    scale = gamma + jnp.concatenate(
        (previous_weight[:, 1:], jnp.zeros_like(previous_weight[:, :1])), axis=1
    )
    padding = (-length) % chunk_size

    def chunks(array):
        array = jnp.pad(array, [(0, 0), (0, padding)] + [(0, 0)] * (array.ndim - 2))
        array = array.reshape(batch, -1, chunk_size, *array.shape[2:])
        return jnp.swapaxes(array, 0, 1)

    data = tuple(chunks(array) for array in (q, k, v, adt, gamma, scale))
    initial = jnp.zeros((batch, heads, state_dim, v.shape[-1]), dtype=jnp.float32)
    position = jnp.arange(chunk_size)
    strictly_lower = position[:, None] > position[None, :]

    def chunk_step(state, inputs):
        qc, kc, vc, log_decay, current_weight, source_scale = inputs
        prefix = jnp.cumsum(log_decay.astype(jnp.float32), axis=1)
        decay = jnp.exp(jnp.minimum(prefix[:, :, None, :] - prefix[:, None, :, :], 0.0))
        # Rank contraction of a write and rank expansion of a read happens in
        # the same QK dot; ranks do not create independent SSM states.
        score = jnp.einsum(
            "bthrn,bshun->bhtsru", qc, kc, preferred_element_type=jnp.float32
        )
        score *= jnp.transpose(decay, (0, 3, 1, 2))[..., None, None]
        score *= jnp.transpose(source_scale, (0, 2, 1))[:, :, None, :, None, None]
        score = jnp.where(strictly_lower[None, None, :, :, None, None], score, 0)
        within = jnp.einsum(
            "bhtsru,bshup->bthrp",
            score.astype(vc.dtype),
            vc,
            preferred_element_type=jnp.float32,
        )
        diagonal = jnp.einsum(
            "bthrn,bthun->bthru", qc, kc, preferred_element_type=jnp.float32
        )
        diagonal = jnp.einsum(
            "bthru,bthup->bthrp",
            diagonal.astype(vc.dtype),
            vc,
            preferred_element_type=jnp.float32,
        )
        diagonal *= current_weight[..., None, None]
        inherited = jnp.einsum(
            "bthrn,bhnp->bthrp",
            qc,
            state.astype(qc.dtype),
            preferred_element_type=jnp.float32,
        )
        inherited *= jnp.exp(prefix)[..., None, None]
        output = inherited + within + diagonal
        tail_decay = jnp.exp(prefix[:, -1:, :] - prefix)
        weighted_v = (
            vc.astype(jnp.float32) * (source_scale * tail_decay)[..., None, None]
        ).astype(vc.dtype)
        update = jnp.einsum(
            "bthrn,bthrp->bhnp", kc, weighted_v, preferred_element_type=jnp.float32
        )
        state = state * jnp.exp(prefix[:, -1])[..., None, None] + update
        return state, output.astype(v.dtype)

    final, output = jax.lax.scan(jax.checkpoint(chunk_step), initial, data)
    output = jnp.swapaxes(output, 0, 1).reshape(batch, -1, heads, rank, v.shape[-1])
    return output[:, :length], final


@partial(jax.jit, static_argnames=("chunk_size", "pairwise"))
def mamba3_chunked(
    q,
    k,
    v,
    adt,
    dt,
    trap_logits,
    angles,
    *,
    chunk_size=64,
    q_bias=None,
    k_bias=None,
    pairwise=True,
):
    """Parallel SSD chunks with an associative scan of boundary summaries.

    All within-chunk matrix products are evaluated in one batch. Only one
    [N,P] summary per chunk is scanned, preserving the same recurrence and
    final state as the serial diagnostic path without serial tiny GEMMs.
    """
    batch, length, heads, rank, state_dim = q.shape
    if k.shape != q.shape or v.shape[:4] != q.shape[:4]:
        raise ValueError("Mamba-3 Q/K/V dimensions are incompatible")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    q, k, _ = _rotary_frame(q, k, dt, angles, q_bias, k_bias, pairwise)
    trap = jax.nn.sigmoid(trap_logits.astype(jnp.float32))
    gamma = dt.astype(jnp.float32) * trap
    previous_weight = dt.astype(jnp.float32) * (1 - trap)
    scale = gamma + jnp.concatenate(
        (previous_weight[:, 1:], jnp.zeros_like(previous_weight[:, :1])), axis=1
    )
    padding = (-length) % chunk_size

    def chunks(array):
        array = jnp.pad(array, [(0, 0), (0, padding)] + [(0, 0)] * (array.ndim - 2))
        return array.reshape(batch, -1, chunk_size, *array.shape[2:])

    qc, kc, vc, log_decay, current_weight, source_scale = (
        chunks(array) for array in (q, k, v, adt, gamma, scale)
    )
    prefix = jnp.cumsum(log_decay.astype(jnp.float32), axis=2)
    decay = jnp.exp(
        jnp.minimum(prefix[:, :, :, None, :] - prefix[:, :, None, :, :], 0.0)
    )
    score = jnp.einsum(
        "bcthrn,bcshun->bchtsru", qc, kc, preferred_element_type=jnp.float32
    )
    score *= jnp.transpose(decay, (0, 1, 4, 2, 3))[..., None, None]
    score *= jnp.transpose(source_scale, (0, 1, 3, 2))[:, :, :, None, :, None, None]
    position = jnp.arange(chunk_size)
    lower = position[:, None] > position[None, :]
    score = jnp.where(lower[None, None, None, :, :, None, None], score, 0)
    within = jnp.einsum(
        "bchtsru,bcshup->bcthrp",
        score.astype(vc.dtype),
        vc,
        preferred_element_type=jnp.float32,
    )
    diagonal = jnp.einsum(
        "bcthrn,bcthun->bcthru", qc, kc, preferred_element_type=jnp.float32
    )
    diagonal = jnp.einsum(
        "bcthru,bcthup->bcthrp",
        diagonal.astype(vc.dtype),
        vc,
        preferred_element_type=jnp.float32,
    )
    diagonal *= current_weight[..., None, None]
    tail_decay = jnp.exp(prefix[:, :, -1:, :] - prefix)
    weighted_v = (
        vc.astype(jnp.float32) * (source_scale * tail_decay)[..., None, None]
    ).astype(vc.dtype)
    update = jnp.einsum(
        "bcthrn,bcthrp->bchnp", kc, weighted_v, preferred_element_type=jnp.float32
    )
    chunk_decay = jnp.exp(prefix[:, :, -1])[..., None, None]

    def combine(earlier, later):
        early_decay, early_state = earlier
        late_decay, late_state = later
        return late_decay * early_decay, late_state + late_decay * early_state

    _, ends = jax.lax.associative_scan(combine, (chunk_decay, update), axis=1)
    starts = jnp.concatenate((jnp.zeros_like(ends[:, :1]), ends[:, :-1]), axis=1)
    inherited = jnp.einsum(
        "bcthrn,bchnp->bcthrp",
        qc,
        starts.astype(qc.dtype),
        preferred_element_type=jnp.float32,
    )
    inherited *= jnp.exp(prefix)[..., None, None]
    output = (within + diagonal + inherited).astype(v.dtype)
    return output.reshape(batch, -1, heads, rank, v.shape[-1])[:, :length], ends[:, -1]


def mamba3_sequential(
    q, k, v, adt, dt, trap_logits, angles, *, q_bias=None, k_bias=None, pairwise=True
):
    """Differentiable, tokenwise recurrence used only for diagnostic conformance."""
    q, k, _ = _rotary_frame(q, k, dt, angles, q_bias, k_bias, pairwise)
    initial = jnp.zeros((q.shape[0], q.shape[2], q.shape[-1], v.shape[-1]), jnp.float32)
    gamma = dt.astype(jnp.float32) * jax.nn.sigmoid(trap_logits.astype(jnp.float32))
    beta = dt.astype(jnp.float32) - gamma

    def step(carry, inputs):
        state, previous = carry
        qi, ki, vi, ai, gi, bi = inputs
        alpha = jnp.exp(ai.astype(jnp.float32))[..., None, None]
        current = jnp.einsum(
            "bhrn,bhrp->bhnp", ki.astype(jnp.float32), vi.astype(jnp.float32)
        )
        state = (
            alpha * (state + bi[..., None, None] * previous)
            + gi[..., None, None] * current
        )
        output = jnp.einsum("bhrn,bhnp->bhrp", qi.astype(jnp.float32), state)
        return (state, current), output.astype(v.dtype)

    inputs = tuple(jnp.swapaxes(array, 0, 1) for array in (q, k, v, adt, gamma, beta))
    (state, _), output = jax.lax.scan(step, (initial, initial), inputs)
    return jnp.swapaxes(output, 0, 1), state


def numpy_reference(
    q, k, v, adt, dt, trap_logits, angles, *, q_bias=None, k_bias=None, pairwise=True
):
    """Independent float64 reference in the *local* complex state frame.

    It rotates the previous state at each step instead of using cumulative
    rotations on Q/K, so it detects rotary sign and phase-integration mistakes.
    """
    q, k, v = (np.asarray(array, dtype=np.float64) for array in (q, k, v))
    if q_bias is not None:
        q = q + np.asarray(q_bias)
    if k_bias is not None:
        k = k + np.asarray(k_bias)
    batch, length, heads, _, state_dim = q.shape
    state = np.zeros((batch, heads, state_dim, v.shape[-1]))
    previous = np.zeros_like(state)
    output = np.empty_like(v)

    def local_rotation(array, increment):
        half = state_dim // 2
        increment = np.pad(
            increment,
            [(0, 0)] * (increment.ndim - 1) + [(0, half - increment.shape[-1])],
        )
        # Move state coordinates to the last axis before rotating real pairs.
        array = np.swapaxes(array, -1, -2)
        if pairwise:
            pairs = array.reshape(*array.shape[:-1], half, 2)
            left, right = pairs[..., 0], pairs[..., 1]
        else:
            left, right = np.split(array, 2, axis=-1)
        cosine, sine = np.cos(increment)[..., None, :], np.sin(increment)[..., None, :]
        left, right = left * cosine - right * sine, left * sine + right * cosine
        result = (
            np.stack((left, right), axis=-1).reshape(array.shape)
            if pairwise
            else np.concatenate((left, right), axis=-1)
        )
        return np.swapaxes(result, -1, -2)

    for index in range(length):
        delta = np.asarray(angles[:, index]) * np.asarray(dt[:, index])[..., None]
        alpha = np.exp(np.asarray(adt[:, index]))[..., None, None]
        trap = 1 / (1 + np.exp(-np.asarray(trap_logits[:, index])))
        current = np.einsum("bhrn,bhrp->bhnp", k[:, index], v[:, index])
        state = alpha * local_rotation(state, -delta)
        state += (
            (np.asarray(dt[:, index]) * (1 - trap))[..., None, None]
            * alpha
            * local_rotation(previous, -delta)
        )
        state += (np.asarray(dt[:, index]) * trap)[..., None, None] * current
        output[:, index] = np.einsum("bhrn,bhnp->bhrp", q[:, index], state)
        previous = current
    return output
