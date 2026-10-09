"""Blocked TPU timings of isolated memory operators at production layer shape.

Launched only through ``scripts.pod`` on all four hosts. Each host times one
pmap over its four local chips; there is no cross-host collective. These are
kernel diagnostics for choosing an implementation, not training throughput.
"""

import argparse
import faulthandler
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np

from lm.kernels import mamba4 as kernels
from lm.kernels.mamba3 import mamba3_chunked
from lm.runtime import atomic_json, initialize


LANES = kernels.spd_solve


def xla_solve(precision, rhs, backend=None):
    """XLA batched Cholesky with automatic differentiation (comparison only)."""
    del backend
    with jax.default_matmul_precision("highest"):
        factor = jnp.linalg.cholesky(precision)
        solved = solve_triangular(factor, rhs[..., None], lower=True)
        return solve_triangular(jnp.swapaxes(factor, -1, -2), solved, lower=False)[
            ..., 0
        ]


def selective_inputs(seed, batch, time, heads, dim, value_dim):
    """Host-side NumPy inputs, so no extra device programs precede the timing."""
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(batch, time, heads, dim)).astype(np.float32)
    q = rng.normal(size=(batch, time, heads, dim)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    v = rng.normal(size=(batch, time, heads, value_dim)).astype(np.float32)
    beta = 1 / (1 + np.exp(-rng.normal(size=(batch, time, heads))))
    log_decay = -np.log1p(np.exp(rng.normal(size=(batch, time, heads)) - 3))
    floor = 0.25 + np.log1p(np.exp(rng.normal(size=(heads, dim))))
    return tuple(np.asarray(a, np.float32) for a in (k, v, q, beta, log_decay, floor))


def timed(function, arguments, repeats, label):
    from jax.experimental import multihost_utils

    started = time.monotonic()
    compiled = function.lower(*arguments).compile()
    compile_seconds = time.monotonic() - started
    # Every host must reach the same launch before timing collective programs.
    multihost_utils.sync_global_devices(f"compiled-{label}")
    samples = []
    for _ in range(repeats + 1):
        started = time.monotonic()
        jax.block_until_ready(compiled(*arguments))
        samples.append(time.monotonic() - started)
    return {
        "compile_seconds": compile_seconds,
        "first_seconds": samples[0],
        "steady_seconds": samples[1:],
        "median_seconds": float(np.median(samples[1:])),
    }


def selective_case(dim, solver, batch, time, heads, value_dim, chunk, repeats):
    # The kernel resolves these module globals at trace time.
    if isinstance(solver, str):
        kernels.SOLVE_BACKEND = solver
        kernels.spd_solve = LANES
    else:
        kernels.spd_solve = solver

    def loss(k, v, q, beta, log_decay, floor):
        result = kernels.selective_gaussian_memory(
            k, v, q, beta, log_decay, floor, chunk_size=chunk
        )
        return jnp.sum(result.output**2) + jnp.sum(result.variance)

    def step(*arguments):
        return jax.grad(loss, argnums=tuple(range(6)))(*arguments)

    inputs = selective_inputs(dim, batch, time, heads, dim, value_dim)
    inputs = jax.device_put_replicated(inputs, jax.local_devices())
    forward = jax.pmap(lambda *a: loss(*a))
    backward = jax.pmap(step)
    return {
        "forward": timed(forward, inputs, repeats, f"selective-{dim}-forward"),
        "forward_backward": timed(backward, inputs, repeats, f"selective-{dim}-step"),
    }


def mamba3_case(batch, time, heads, state, value_dim, chunk, repeats):
    key = jax.random.split(jax.random.key(3), 6)
    q = jax.random.normal(key[0], (batch, time, heads, 1, state), jnp.bfloat16)
    k = jax.random.normal(key[1], (batch, time, heads, 1, state), jnp.bfloat16)
    v = jax.random.normal(key[2], (batch, time, heads, 1, value_dim), jnp.bfloat16)
    dt = jax.nn.softplus(jax.random.normal(key[3], (batch, time, heads)) - 3)
    adt = -dt
    trap = jax.random.normal(key[4], (batch, time, heads))
    angles = jax.random.normal(key[5], (batch, time, heads, state // 4))
    inputs = jax.device_put_replicated(
        (q, k, v, adt, dt, trap, angles), jax.local_devices()
    )

    def loss(*arguments):
        output, _ = mamba3_chunked(*arguments, chunk_size=chunk)
        return jnp.sum(output.astype(jnp.float32) ** 2)

    forward = jax.pmap(loss)
    backward = jax.pmap(jax.grad(loss, argnums=tuple(range(7))))
    return {
        "forward": timed(forward, inputs, repeats, "mamba3-forward"),
        "forward_backward": timed(backward, inputs, repeats, "mamba3-step"),
    }


BLOCK_BASE = dict(
    architecture="mamba4",
    vocab_size=50257,
    d_model=512,
    n_layers=1,
    d_ff=0,
    d_state=96,
    expand=2,
    head_dim=64,
    key_dim=16,
    chunk_size=64,
    dtype="bfloat16",
    param_dtype="float32",
    max_seq_len=1024,
    remat=False,
    memory_mixer="selective",
    layer_pattern="M",
    protected_anchor_budget=0,
    conv_kernel=4,
)


def block_case(overrides, repeats, label):
    """Time one residual block exactly as a model layer uses it."""
    from lm.config import ModelConfig
    from lm.models.mamba3 import Mamba3Block
    from lm.models.mamba4 import SelectiveMemoryBlock
    from flax import linen as nn

    overrides = dict(overrides)
    kernels.spd_solve = LANES
    rematerialize = overrides.pop("block_remat", False)
    config = ModelConfig(**{**BLOCK_BASE, **overrides})
    kind = Mamba3Block if config.layer_pattern == "S" else SelectiveMemoryBlock
    kind = nn.remat(kind, static_argnums=(2,)) if rematerialize else kind
    block = kind(config)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 1024, config.d_model)).astype(np.float32) * 0.5
    params = block.init(jax.random.key(0), x[:1, :64], False)["params"]

    def loss(params, x):
        y = block.apply({"params": params}, x.astype(jnp.bfloat16), False)
        return jnp.sum(jnp.square(y.astype(jnp.float32)))

    inputs = jax.device_put_replicated((params, x), jax.local_devices())
    forward = jax.pmap(loss)
    backward = jax.pmap(jax.grad(loss, argnums=(0, 1)))
    return {
        "forward": timed(forward, inputs, repeats, f"{label}-forward"),
        "forward_backward": timed(backward, inputs, repeats, f"{label}-step"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dims", default="16,32")
    parser.add_argument("--solvers", default="loop,unrolled,xla")
    parser.add_argument("--skip-mamba3", action="store_true")
    parser.add_argument("--dump", default="")
    parser.add_argument("--libtpu-args", default="")
    parser.add_argument(
        "--blocks", default="", help="JSON object: label -> config overrides"
    )
    options = parser.parse_args()
    if options.libtpu_args:
        import os

        os.environ["LIBTPU_INIT_ARGS"] = options.libtpu_args
    if options.dump:
        # Backends initialize lazily, so this reaches XLA before compilation.
        import os

        os.environ["XLA_FLAGS"] = f"--xla_dump_to={options.dump} --xla_dump_hlo_as_text"
    hardware = initialize(True)
    batch, time_steps, heads, value_dim, chunk = 8, 1024, 16, 64, 64
    result = {
        "hardware": hardware,
        "shape": {
            "batch_per_device": batch,
            "time": time_steps,
            "heads": heads,
            "value_dim": value_dim,
            "chunk": chunk,
        },
        "cases": {},
    }
    for label, overrides in (
        json.loads(options.blocks) if options.blocks else {}
    ).items():
        try:
            result["cases"][label] = block_case(overrides, options.repeats, label)
        except Exception as error:  # Record a failed variant, keep the others.
            result["cases"][label] = {"error": repr(error)[:2000]}
        print(label, json.dumps(result["cases"][label])[:400], flush=True)
    solvers = {"loop": "loop", "unrolled": "unrolled", "xla": xla_solve}
    for dim in (int(value) for value in options.dims.split(",") if value):
        for name in options.solvers.split(","):
            solver = solvers[name]
            label = f"selective_d{dim}_{name}"
            try:
                result["cases"][label] = selective_case(
                    dim,
                    solver,
                    batch,
                    time_steps,
                    heads,
                    value_dim,
                    chunk,
                    options.repeats,
                )
            except Exception as error:  # Record a failed variant, keep the others.
                result["cases"][label] = {"error": repr(error)[:2000]}
            print(label, json.dumps(result["cases"][label])[:400], flush=True)
    if options.skip_mamba3:
        atomic_json(Path(options.output) / f"host-{jax.process_index()}.json", result)
        return
    result["cases"]["mamba3_n96"] = mamba3_case(
        batch, time_steps, heads, 96, value_dim, chunk, options.repeats
    )
    print("mamba3", json.dumps(result["cases"]["mamba3_n96"])[:400], flush=True)
    atomic_json(Path(options.output) / f"host-{jax.process_index()}.json", result)


if __name__ == "__main__":
    faulthandler.enable()
    main()
