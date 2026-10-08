"""Analytic read derivatives and scalar-loss reverse recurrence."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from analysis.memory import EvidenceState


def read_vjp(state, query, upstream, ridge):
    """VJP for L=upstream^T C (S+ridge I)^-1 query, treating S symmetric."""
    factor = cho_factor(state.s + ridge * np.eye(len(state.s)))
    y = cho_solve(factor, query)
    adjoint = cho_solve(factor, state.c.T @ upstream)
    dc = np.outer(upstream, y)
    ds = -0.5 * (np.outer(adjoint, y) + np.outer(y, adjoint))
    return ds, dc, adjoint


def terminal_stream_vjp(keys, values, beta, decay, query, upstream, ridge):
    state = EvidenceState.zeros(keys.shape[1], values.shape[1])
    before = []
    for k, v, b, lam in zip(keys, values, beta, decay):
        before.append(EvidenceState(state.s.copy(), state.c.copy()))
        state.write(k, v, b, lam)
    gs, gc, gq = read_vjp(state, query, upstream, ridge)
    gk, gv = np.zeros_like(keys), np.zeros_like(values)
    gb, gl = np.zeros_like(beta), np.zeros_like(decay)
    for i in range(len(keys) - 1, -1, -1):
        k, v = keys[i], values[i]
        gb[i] = k @ gs @ k + v @ gc @ k
        gl[i] = np.sum(gs * before[i].s) + np.sum(gc * before[i].c)
        gk[i] = beta[i] * ((gs + gs.T) @ k + gc.T @ v)
        gv[i] = beta[i] * gc @ k
        gs *= decay[i]
        gc *= decay[i]
    return {"keys": gk, "values": gv, "beta": gb, "decay": gl, "query": gq}
