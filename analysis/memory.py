"""Reference algebra, deliberately independent of future TPU training code.

Rows of ``keys`` are observations; rows of ``values`` are corresponding values.
The state C has shape (value_dim, key_dim). No explicit inverse is used for reads.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular

Array = NDArray[np.floating]


def unit_rows(x: Array) -> Array:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize zero keys")
    return x / norms


def effective_weights(beta: Array, decay: Array) -> Array:
    """beta[i] times product of all subsequent decay factors, without division."""
    if beta.shape != decay.shape:
        raise ValueError("Evidence and decay lengths must agree")
    weights = np.empty_like(beta, dtype=np.float64)
    tail = 1.0
    for i in range(len(beta) - 1, -1, -1):
        weights[i] = beta[i] * tail
        tail *= decay[i]
    return weights


@dataclass
class EvidenceState:
    s: Array
    c: Array

    @classmethod
    def zeros(cls, key_dim: int, value_dim: int) -> "EvidenceState":
        return cls(np.zeros((key_dim, key_dim)), np.zeros((value_dim, key_dim)))

    @classmethod
    def from_batch(
        cls, keys: Array, values: Array, weights: Array | None = None
    ) -> "EvidenceState":
        if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
            raise ValueError("Keys and values must be matrices with equal row count")
        w = np.ones(len(keys)) if weights is None else np.asarray(weights)
        if w.shape != (len(keys),) or np.any(w < 0):
            raise ValueError("Evidence weights must be nonnegative, one per write")
        return cls(keys.T @ (w[:, None] * keys), values.T @ (w[:, None] * keys))

    def write(self, key: Array, value: Array, beta=1.0, decay=1.0) -> None:
        if beta < 0 or not 0 <= decay <= 1:
            raise ValueError("Expected beta >= 0 and decay in [0, 1]")
        self.s = decay * self.s + beta * np.outer(key, key)
        self.c = decay * self.c + beta * np.outer(value, key)

    def merge(self, other: "EvidenceState") -> "EvidenceState":
        """Add evidence only; the caller adds its prior exactly once at read time."""
        return EvidenceState(self.s + other.s, self.c + other.c)

    def read(self, queries: Array, ridge: float) -> Array:
        if ridge <= 0:
            raise ValueError("SPD read requires positive ridge; use interpolation")
        a = self.s + ridge * np.eye(len(self.s))
        factor = cho_factor(a, lower=True)
        q = np.atleast_2d(queries)
        return (self.c @ cho_solve(factor, q.T)).T

    def confidence(self, queries: Array, ridge: float) -> Array:
        if ridge <= 0:
            raise ValueError("Confidence requires a positive prior precision")
        q = np.atleast_2d(queries)
        x = cho_solve(cho_factor(self.s + ridge * np.eye(len(self.s))), q.T).T
        return np.einsum("ij,ij->i", q, x)


@dataclass
class AffineSummary:
    decay: float
    s: Array
    c: Array

    def after(self, earlier: "AffineSummary") -> "AffineSummary":
        return AffineSummary(
            self.decay * earlier.decay,
            self.s + self.decay * earlier.s,
            self.c + self.decay * earlier.c,
        )


def tree_prefix(elements: list[AffineSummary]) -> list[AffineSummary]:
    """Work-efficient balanced scan; O(n) combines, O(log n) dependency depth.

    This serial NumPy evaluator checks the tree algorithm's arithmetic; it does
    not measure parallel speed. Output is inclusive, in chronological order.
    """
    if not elements:
        return []
    identity = AffineSummary(1.0, np.zeros_like(elements[0].s), 0 * elements[0].c)

    def build(lo, hi):
        if hi - lo == 1:
            return (lo, hi, elements[lo], None, None)
        mid = (lo + hi) // 2
        left, right = build(lo, mid), build(mid, hi)
        return (lo, hi, right[2].after(left[2]), left, right)

    out = [identity] * len(elements)

    def visit(node, preceding):
        lo, hi, summary, left, right = node
        if hi - lo == 1:
            out[lo] = summary.after(preceding)
            return
        visit(left, preceding)
        visit(right, left[2].after(preceding))

    visit(build(0, len(elements)), identity)
    return out


def interpolation(keys: Array, values: Array, queries: Array) -> Array:
    """Minimum-norm exact read within capacity; all write keys must be independent.

    QR avoids squaring the key condition number, unlike the normal equations.
    Queries outside the key span get the minimum-norm extension, without a
    calibrated posterior confidence. This mode does not claim a positive floor.
    """
    if len(keys) > keys.shape[1] or np.linalg.matrix_rank(keys) < len(keys):
        raise ValueError("Protected interpolation requires independent keys")
    basis, triangular = np.linalg.qr(keys.T, mode="reduced")
    coeff = solve_triangular(triangular, basis.T @ np.atleast_2d(queries).T)
    return (values.T @ coeff).T


def softmax_read(
    keys: Array,
    values: Array,
    queries: Array,
    temperature: float,
    precision: Array | None = None,
) -> Array:
    scores = temperature * np.atleast_2d(queries) @ keys.T
    if precision is not None:
        if np.any(precision <= 0):
            raise ValueError("Precision attention expects positive precision")
        scores += np.log(precision)[None, :]
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ values


def hebbian_read(keys: Array, values: Array, queries: Array) -> Array:
    return np.atleast_2d(queries) @ keys.T @ values


def diagonal_read(keys: Array, values: Array, queries: Array, ridge: float) -> Array:
    state = EvidenceState.from_batch(keys, values)
    return (state.c @ (np.atleast_2d(queries) / (state.s.diagonal() + ridge)).T).T


def delta_read(keys: Array, values: Array, queries: Array, step: float) -> Array:
    m = np.zeros((values.shape[1], keys.shape[1]))
    for k, v in zip(keys, values):
        m += step * np.outer(v - m @ k, k)
    return (m @ np.atleast_2d(queries).T).T


def rotate_pairs(x: Array, angle: Array | float) -> Array:
    """Apply the real representation of a complex diagonal phase to row pairs."""
    if x.shape[0] % 2:
        raise ValueError("Complex state dimension must be even")
    pair = x.reshape(x.shape[0] // 2, 2, *x.shape[1:])
    shape = (len(pair),) + (1,) * (x.ndim - 1)
    theta = np.broadcast_to(np.asarray(angle), (len(pair),)).reshape(shape)
    cosine, sine = np.cos(theta), np.sin(theta)
    out = np.empty_like(pair)
    out[:, 0] = cosine * pair[:, 0] - sine * pair[:, 1]
    out[:, 1] = sine * pair[:, 0] + cosine * pair[:, 1]
    return out.reshape(x.shape)


def mamba3_core(
    b: Array,
    x: Array,
    read_keys: Array,
    dt: Array,
    real_a: Array,
    angles: Array,
    trap: Array,
    initial: Array | None = None,
) -> tuple[Array, Array]:
    """Mamba-3 real-pair exponential-trapezoidal MIMO recurrence.

    B: (time, state_dim, rank), X: (time, value_dim, rank).
    H <- alpha R(-angle)(H) + (1-trap) dt alpha R(-angle)(Bprev Xprev^T)
                      + trap dt B X^T.
    Read keys: (queries, state_dim, rank); output (queries, value_dim, rank).
    ``angles`` are already dt-integrated (paper theta times dt). The past input
    is zero at the first step. No MLP, learned encoder, output gate or training
    is represented; this is an operator baseline, not a trained language model.
    """
    n, dim, rank = b.shape
    if x.shape[0] != n or x.shape[2] != rank or read_keys.shape[1:] != (dim, rank):
        raise ValueError("Incompatible MIMO shapes")
    h = np.zeros((dim, x.shape[1])) if initial is None else initial.copy()
    previous = np.zeros_like(h)
    for i in range(n):
        alpha = np.exp(dt[i] * real_a[i])
        current = b[i] @ x[i].T
        h = (
            alpha * rotate_pairs(h, -angles[i])
            + (1 - trap[i]) * dt[i] * alpha * rotate_pairs(previous, -angles[i])
            + trap[i] * dt[i] * current
        )
        previous = current
    return np.einsum("qdr,dv->qvr", read_keys, h), h


def mamba3_read(
    keys: Array,
    values: Array,
    queries: Array,
    trap=1.0,
    decay=1.0,
    angle=0.0,
    rank=1,
) -> Array:
    """Tied lifts isolate recurrence differences at equal encoded observations."""
    n = len(keys)
    scale = np.ones(rank) / np.sqrt(rank)
    b = keys[:, :, None] * scale
    x = values[:, :, None] * scale
    reads = np.atleast_2d(queries)[:, :, None] * scale
    y, _ = mamba3_core(
        b,
        x,
        reads,
        np.ones(n),
        np.full(n, np.log(decay)),
        np.full((n, keys.shape[1] // 2), angle),
        np.full(n, trap),
    )
    return np.einsum("qvr,r->qv", y, scale)


def chol_rank1(lower: Array, vector: Array) -> Array:
    """Stable textbook rank-one Cholesky update, O(d^2); no downdate."""
    factor, x = lower.copy(), vector.copy()
    for i in range(len(x)):
        r = np.hypot(factor[i, i], x[i])
        cosine = r / factor[i, i]
        sine = x[i] / factor[i, i]
        factor[i, i] = r
        factor[i + 1 :, i] = (factor[i + 1 :, i] + sine * x[i + 1 :]) / cosine
        x[i + 1 :] = cosine * x[i + 1 :] - sine * factor[i + 1 :, i]
    return factor


@dataclass
class CyclicFloorMemory:
    """O(d^2) decode with a guaranteed anisotropic, rather than fixed, floor.

    For constant lambda, stationary diagonal prior F has min eps and max
    eps * lambda**(-(d-1)). Initialize at the exact cycle to avoid burn-in.
    The affine state recurrence is scannable, including the scheduled injection.
    At lambda=1, use a fixed prior and no phantom write.
    """

    state: EvidenceState
    lower: Array
    prior: Array
    ridge: float
    decay: float
    step: int = 0

    @classmethod
    def create(cls, key_dim, value_dim, ridge, decay):
        if ridge <= 0 or not 0 < decay <= 1:
            raise ValueError("Expected positive ridge and decay in (0,1]")
        prior = ridge * decay ** (-np.arange(key_dim, dtype=float))
        return cls(
            EvidenceState.zeros(key_dim, value_dim),
            np.diag(np.sqrt(prior)),
            prior,
            ridge,
            decay,
        )

    def write(self, key, value, beta=1.0):
        dim = len(self.prior)
        index = self.step % dim
        amplitude = self.ridge * (self.decay ** (-(dim - 1)) - self.decay)
        self.state.write(key, value, beta, self.decay)
        self.lower *= np.sqrt(self.decay)
        self.lower = chol_rank1(self.lower, np.sqrt(beta) * key)
        self.prior *= self.decay
        self.prior[index] += amplitude
        if amplitude > 0:
            phantom = np.zeros(dim)
            phantom[index] = np.sqrt(amplitude)
            self.lower = chol_rank1(self.lower, phantom)
        self.step += 1

    def read(self, queries):
        return (
            self.state.c @ cho_solve((self.lower, True), np.atleast_2d(queries).T)
        ).T


@dataclass
class ProtectedBlock:
    """Distinct exact anchors plus separate, additive background evidence."""

    span: int
    keys: Array
    values: Array
    ids: NDArray[np.integer]
    background: EvidenceState
    basis: Array = field(init=False, repr=False)
    triangular: Array = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.keys) > self.keys.shape[1] or np.linalg.matrix_rank(
            self.keys
        ) < len(self.keys):
            raise ValueError("Protected block anchors must be independent")
        self.basis, self.triangular = np.linalg.qr(self.keys.T, mode="reduced")

    def read(self, queries):
        coefficients = solve_triangular(
            self.triangular, self.basis.T @ np.atleast_2d(queries).T
        )
        return (self.values.T @ coefficients).T


def select_anchors(keys: Array, values: Array, ids, budget, tolerance=1e-8):
    """Deterministic chronological Gram-Schmidt selection; reject dependence.

    Selection is a research policy, not a Bayes-optimal eviction theorem. Later
    phases must evaluate learned/recency/residual policies on held-out tasks.
    """
    chosen, basis = [], []
    for i in np.argsort(ids, kind="stable"):
        residual = keys[i].copy()
        for q in basis:
            residual -= q * (q @ residual)
        norm = np.linalg.norm(residual)
        if norm > tolerance:
            chosen.append(i)
            basis.append(residual / norm)
        if len(chosen) == budget:
            break
    return keys[chosen], values[chosen], np.asarray(ids)[chosen]


class ProtectedCascade:
    """Redundant binary counter: retain 1-2 blocks at every occupied level.

    On a third block, merge the two oldest; reselect <=d independent anchors.
    All discarded writes remain in separate ridge evidence. Exactness concerns
    retained anchors only, and routing IDs must be counted in the state budget.
    """

    def __init__(self, key_dim, value_dim, anchor_budget=None):
        self.dim = key_dim
        self.value_dim = value_dim
        self.budget = key_dim if anchor_budget is None else anchor_budget
        if not 1 <= self.budget <= key_dim:
            raise ValueError("Anchor budget must be in [1,key_dim]")
        self.levels: list[list[ProtectedBlock]] = []
        self.merges = 0

    def append(self, keys, values, ids):
        k, v, kept = select_anchors(keys, values, ids, self.budget)
        block = ProtectedBlock(
            len(keys), k, v, kept, EvidenceState.from_batch(keys, values)
        )
        level = 0
        while True:
            if level == len(self.levels):
                self.levels.append([])
            self.levels[level].append(block)
            if len(self.levels[level]) <= 2:
                break
            left, right = self.levels[level][:2]
            del self.levels[level][:2]
            k, v, kept = select_anchors(
                np.concatenate([left.keys, right.keys]),
                np.concatenate([left.values, right.values]),
                np.concatenate([left.ids, right.ids]),
                self.budget,
            )
            block = ProtectedBlock(
                left.span + right.span,
                k,
                v,
                kept,
                left.background.merge(right.background),
            )
            self.merges += 1
            level += 1

    @property
    def blocks(self):
        return [block for level in self.levels for block in level]
