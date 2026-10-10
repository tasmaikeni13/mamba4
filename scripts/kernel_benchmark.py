"""Time kernels and single blocks on one local TPU chip per host.

Run via scripts.pod on every host; each host measures its own first chip
with synthetic inputs and no collectives, so the four records are
independent repeats. Every timing blocks on device results and excludes
compilation. Forward and forward+backward (value_and_grad of a fixed random
projection of the outputs) are reported separately.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from lm.config import ModelConfig
from lm.runtime import atomic_json, initialize, source_provenance


def measure(function, *args, repeats=10):
    compiled = jax.jit(function).lower(*args).compile()
    started = time.perf_counter()
    jax.block_until_ready(compiled(*args))
    first = time.perf_counter() - started
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - started)
    return {"median_seconds": float(np.median(times)), "first_seconds": first}


def projected_loss(function):
    """Scalar loss: fixed pseudo-random weights times every float output."""

    def loss(*args):
        total = 0.0
        for index, leaf in enumerate(jax.tree.leaves(function(*args))):
            weights = jnp.cos(jnp.arange(leaf.size, dtype=jnp.float32) * (index + 1.7))
            total += jnp.sum(leaf.astype(jnp.float32).reshape(-1) * weights)
        return total

    return loss


def both(name, function, args, argnums, results):
    device = jax.local_devices()[0]
    args = jax.device_put(args, device)
    entry = {}
    try:
        entry["forward"] = measure(function, *args)
        grad = jax.value_and_grad(projected_loss(function), argnums=argnums)
        entry["forward_backward"] = measure(grad, *args)
    except Exception as error:  # record and continue with the next variant
        entry["error"] = f"{type(error).__name__}: {str(error)[:4000]}"
    results[name] = entry
    print(name, json.dumps(entry)[:600], flush=True)


def ssd_inputs(batch, length, heads, state, width, seed=0):
    rng = np.random.default_rng(seed)
    shape = (batch, length, heads)
    q = rng.normal(size=(*shape, 1, state)) * 0.3
    k = rng.normal(size=(*shape, 1, state)) * 0.3
    v = rng.normal(size=(*shape, 1, width))
    dt = rng.uniform(0.001, 0.1, size=shape)
    adt = -rng.uniform(0.5, 4, size=shape) * dt
    trap = rng.normal(size=shape)
    angles = rng.normal(size=(*shape, state // 4))
    bias = rng.normal(size=(heads, 1, state)) * 0.1
    as_dtype = lambda x, d: jnp.asarray(x, d)  # noqa: E731
    return (
        as_dtype(q, jnp.bfloat16),
        as_dtype(k, jnp.bfloat16),
        as_dtype(v, jnp.bfloat16),
        as_dtype(adt, jnp.float32),
        as_dtype(dt, jnp.float32),
        as_dtype(trap, jnp.float32),
        as_dtype(angles, jnp.float32),
        as_dtype(bias, jnp.float32),
        as_dtype(bias, jnp.float32),
    )


def ssd_benchmarks(results, shapes):
    from lm.kernels.flashmamba import mamba3_flash
    from lm.kernels.mamba3 import mamba3_chunked
    from lm.kernels.mamba3_fast import mamba3_fast

    def flash(heads):
        return lambda *a: mamba3_flash(
            *a[:7], q_bias=a[7], k_bias=a[8], heads_per_step=heads
        )

    kernels = {
        "chunked": lambda *a: mamba3_chunked(*a[:7], q_bias=a[7], k_bias=a[8]),
        "fast": lambda *a: mamba3_fast(*a[:7], q_bias=a[7], k_bias=a[8]),
        "flash-h1": flash(1),
        "flash-h2": flash(2),
        "flash-h4": flash(4),
        "flash-h8": flash(8),
    }
    for label, shape in shapes.items():
        args = ssd_inputs(*shape)
        for name, kernel in kernels.items():
            both(f"ssd/{label}/{name}", kernel, args, tuple(range(9)), results)


def part_benchmarks(results, batch, length, heads=24, state=128, width=64):
    """Pieces of one Mamba-3 layer's scan and one memory layer's read."""
    from lm.kernels import flashmamba
    from lm.kernels.mamba3 import _rotary_frame
    from lm.kernels.mamba4_fused import batched_solve

    q, k, v, adt, dt, trap, angles, qb, kb = ssd_inputs(
        batch, length, heads, state, width
    )

    def rotary(q, k, dt, angles, qb, kb):
        rotated_q, rotated_k, _ = _rotary_frame(q, k, dt, angles, qb, kb, True)
        return rotated_q, rotated_k

    both("part/rotary-frame", rotary, (q, k, dt, angles, qb, kb), (0, 1, 2, 3), results)
    both(
        "part/rotary-pairs",
        flashmamba._rotary_frame_pairs,
        (q, k, dt, angles, qb, kb),
        (0, 1, 2, 3),
        results,
    )
    both(
        "part/mamba3-flash",
        lambda *a: flashmamba.mamba3_flash(*a[:7], q_bias=a[7], k_bias=a[8]),
        (q, k, v, adt, dt, trap, angles, qb, kb),
        tuple(range(9)),
        results,
    )
    gamma = dt * 0.5
    core = (q[..., 0, :], k[..., 0, :], v[..., 0, :], adt, gamma, gamma)
    both(
        "part/flash-scan",
        lambda *a: flashmamba.ssd(*a, 128, False, 1),
        core,
        tuple(range(6)),
        results,
    )
    rng = np.random.default_rng(1)
    sequences, chunk, dim = batch * (length // 64) * 12, 64, 64
    g = rng.normal(size=(sequences, dim, dim)) / 8
    s0 = jnp.asarray(g @ np.swapaxes(g, 1, 2), jnp.float32)
    keys = rng.normal(size=(sequences, chunk, dim))
    keys = jnp.asarray(keys / np.linalg.norm(keys, axis=-1, keepdims=True), jnp.float32)
    beta = jnp.asarray(rng.uniform(0.1, 1, size=(sequences, chunk)), jnp.float32)
    prefix = jnp.asarray(
        np.cumsum(-rng.uniform(0, 0.05, size=(sequences, chunk)), axis=-1), jnp.float32
    )
    floor = jnp.ones((sequences, dim), jnp.float32)
    rhs = jnp.asarray(rng.normal(size=(sequences, chunk, dim)), jnp.float32)
    device = jax.local_devices()[0]
    arrays = jax.device_put((s0, keys, beta, prefix, floor, rhs), device)
    from lm.kernels.mamba4_fused import factored_solve

    def store(*a):
        return batched_solve(*a, store_factors=True)

    pieces = {
        "part/memory-solve": (lambda *a: batched_solve(*a), arrays),
        "part/memory-solve-store": (store, arrays),
    }
    for name, (function, inputs) in pieces.items():
        try:
            results[name] = {"forward": measure(function, *inputs)}
        except Exception as error:
            results[name] = {"error": f"{type(error).__name__}: {str(error)[:3000]}"}
        print(name, json.dumps(results[name])[:600], flush=True)
    try:
        _, factors = jax.jit(store)(*arrays)
        results["part/memory-factored"] = {
            "forward": measure(lambda f, r: factored_solve(f, r), factors, arrays[-1])
        }
    except Exception as error:
        results["part/memory-factored"] = {
            "error": f"{type(error).__name__}: {str(error)[:3000]}"
        }
    print(
        "part/memory-factored",
        json.dumps(results["part/memory-factored"])[:600],
        flush=True,
    )


def memory_inputs(batch, length, heads, dim, width, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(batch, length, heads, dim))
    q = rng.normal(size=k.shape)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    v = rng.normal(size=(batch, length, heads, width))
    beta = rng.uniform(0.05, 1.0, size=k.shape[:-1])
    log_decay = np.log(rng.uniform(0.9, 0.999, size=k.shape[:-1]))
    floor = rng.uniform(0.25, 1.5, size=(heads, dim))
    return tuple(jnp.asarray(x, jnp.float32) for x in (k, v, q, beta, log_decay, floor))


def memory_benchmarks(results, shapes):
    from lm.kernels.mamba4 import selective_gaussian_memory

    for label, shape in shapes.items():
        args = memory_inputs(*shape)
        for solver in ("blocked", "fused"):

            def run(*a, solver=solver):
                out = selective_gaussian_memory(*a, chunk_size=64, solver=solver)
                return out.output, out.variance

            both(f"memory/{label}/{solver}", run, args, tuple(range(6)), results)


def attention_benchmarks(results, batch, length, heads=12, width=64):
    """Causal attention kernels on [B,H,T,D] inputs, as the Transformer uses."""
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as masks,
    )

    rng = np.random.default_rng(0)
    q, k, v = (
        jnp.asarray(rng.normal(size=(batch, heads, length, width)), jnp.bfloat16)
        for _ in range(3)
    )
    scale = width**-0.5

    def flash(block):
        sizes = None
        if block:
            sizes = fa.BlockSizes(
                block_q=block,
                block_k_major=block,
                block_k=block,
                block_b=1,
                block_q_major_dkv=block,
                block_k_major_dkv=block,
                block_k_dkv=block,
                block_q_dkv=block,
                block_k_major_dq=block,
                block_k_dq=block,
                block_q_dq=block,
            )
        return lambda q, k, v: fa.flash_attention(
            q, k, v, causal=True, sm_scale=scale, block_sizes=sizes
        )

    def splash_attention(block, fused):
        mask = masks.MultiHeadMask(
            [masks.CausalMask((length, length)) for _ in range(heads)]
        )
        sizes = splash.BlockSizes(
            block_q=block,
            block_kv=block,
            block_kv_compute=block,
            block_q_dkv=block,
            block_kv_dkv=block,
            block_kv_dkv_compute=block,
            block_q_dq=None if fused else block,
            block_kv_dq=None if fused else block,
            use_fused_bwd_kernel=fused,
        )
        kernel = splash.make_splash_mha(
            mask, block_sizes=sizes, head_shards=1, q_seq_shards=1
        )
        return lambda q, k, v: jax.vmap(kernel)(q * scale, k, v)

    variants = {
        "flash-128": flash(None),
        "flash-256": flash(256),
        "flash-512": flash(512),
        "splash-512": splash_attention(512, False),
        "splash-512-fused": splash_attention(512, True),
        "splash-1024-fused": splash_attention(1024, True),
    }
    for name, function in variants.items():
        both(f"attention/{name}", function, (q, k, v), (0, 1, 2), results)


def block_benchmarks(results, base, batch, length, variants):
    from lm.models.mamba3 import Mamba3Block
    from lm.models.mamba4 import SelectiveMemoryBlock
    from lm.models.transformer import TransformerBlock

    kinds = {
        "transformer": TransformerBlock,
        "mamba3": Mamba3Block,
        "memory": SelectiveMemoryBlock,
    }
    for name, (kind, overrides) in variants.items():
        config = dataclasses.replace(base, **overrides)
        module = kinds[kind](config)
        x = jnp.asarray(
            np.random.default_rng(0).normal(size=(batch, length, config.d_model)),
            jnp.bfloat16,
        )
        params = jax.jit(module.init)(jax.random.PRNGKey(0), x)["params"]

        def run(params, x, module=module):
            return module.apply({"params": params}, x)

        both(f"block/{name}", run, (params, x), (0, 1), results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--groups", nargs="+", default=["ssd", "memory", "attention", "blocks"]
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--single-host", action="store_true")
    options = parser.parse_args()
    info = initialize(not options.single_host)
    results = {"hardware": info, "sources": source_provenance()["sha256"]}
    batch, length = options.batch, options.length
    if "ssd" in options.groups:
        ssd_benchmarks(
            results,
            {
                "125m": (batch, length, 24, 128, 64),
                "60m": (max(batch // 2, 1), length, 16, 96, 64),
            },
        )
    if "parts" in options.groups:
        part_benchmarks(results, batch, length)
    if "memory" in options.groups:
        memory_benchmarks(results, {"125m": (batch, length, 12, 64, 128)})
    if "attention" in options.groups:
        attention_benchmarks(results, batch, length)
    if "blocks" in options.groups:
        frozen = json.loads(Path("lm/configs/mamba4-60m.json").read_text())["model"]
        base = ModelConfig(
            **{**frozen, "d_model": 768, "d_state": 128, "n_heads": 12, "d_ff": 2048}
        )
        variants = {
            "transformer": ("transformer", {}),
            "mamba3-chunked": ("mamba3", {"ssd_kernel": "chunked"}),
            "mamba3-flash": ("mamba3", {"ssd_kernel": "flash"}),
            "memory-blocked": ("memory", {"memory_solver": "blocked"}),
            "memory-fused": ("memory", {"memory_solver": "fused"}),
        }
        block_benchmarks(results, base, batch, length, variants)
    output = Path(options.output)
    atomic_json(output / f"host-{jax.process_index()}.json", results)
    if not options.single_host:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
