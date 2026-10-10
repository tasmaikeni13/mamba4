"""Compile one Pallas construct per process to locate Mosaic compiler aborts.

Without --case, every case runs in its own subprocess (a compiler CHECK
failure aborts the whole process) and a table of outcomes is printed. Run
with `python -m scripts.pod local --tag <tag> scripts.mosaic_probe`.
"""

import argparse
import subprocess
import sys

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32


def single(kernel, *inputs, out=None):
    out = out or jax.ShapeDtypeStruct(inputs[0].shape, F32)
    result = pl.pallas_call(kernel, out_shape=out)(*inputs)
    return jax.block_until_ready(result)


def case_roll(x):
    def kernel(x_ref, o_ref):
        value = x_ref[...]
        o_ref[...] = pltpu.roll(value, 1, 1) + pltpu.roll(value, 127, 1)

    return single(kernel, x)


def case_iota_rem(x):
    def kernel(x_ref, o_ref):
        lanes = lax.broadcasted_iota(jnp.int32, x_ref.shape, 1)
        o_ref[...] = jnp.where(lanes % 2 == 0, x_ref[...], -x_ref[...])

    return single(kernel, x)


def case_iota_and(x):
    def kernel(x_ref, o_ref):
        lanes = lax.broadcasted_iota(jnp.int32, x_ref.shape, 1)
        o_ref[...] = jnp.where((lanes & 1) == 0, x_ref[...], -x_ref[...])

    return single(kernel, x)


def case_partner(x):
    from lm.kernels.flashmamba import _partner

    def kernel(x_ref, o_ref):
        o_ref[...] = _partner(x_ref[...])

    return single(kernel, x)


def case_lane_slice(x):
    def kernel(x_ref, o_ref):
        left = x_ref[:, 0:64].astype(F32)
        right = x_ref[:, 64:128].astype(F32)
        o_ref[:, 0:64] = right * 2
        o_ref[:, 64:128] = left * 3

    return single(
        kernel, x.astype(jnp.bfloat16), out=jax.ShapeDtypeStruct(x.shape, F32)
    )


def case_cos_sin(x):
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.cos(x_ref[...]) + jnp.sin(x_ref[...])

    return single(kernel, x)


def case_mask_dot(x):
    from lm.kernels.flashmamba import _mask_dot

    def kernel(x_ref, o_ref):
        rows = lax.broadcasted_iota(jnp.int32, x_ref.shape, 0)
        columns = lax.broadcasted_iota(jnp.int32, x_ref.shape, 1)
        o_ref[...] = _mask_dot(jnp.where(rows >= columns, 1.0, 0.0), x_ref[...])

    return single(kernel, x)


def case_scratch_dynamic(x):
    def kernel(x_ref, o_ref, s_ref):
        block = pl.program_id(2)

        @pl.when(pl.program_id(1) == 0)
        def _():
            s_ref[block] = jnp.zeros(s_ref.shape[1:], F32)

        s_ref[block] = s_ref[block] + x_ref[...]
        o_ref[...] = s_ref[block]

    spec = pl.BlockSpec((None, 8, 128), lambda b, i, h: (h, 0, 0))
    stacked = jnp.broadcast_to(x[:8], (4, 8, 128))
    result = pl.pallas_call(
        kernel,
        grid=(2, 3, 4),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(stacked.shape, F32),
        scratch_shapes=[pltpu.VMEM((4, 8, 128), F32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
    )(stacked)
    return jax.block_until_ready(result)


def fused_inputs():
    rng = np.random.default_rng(0)
    batch, length, heads, state, width = 2, 256, 4, 128, 64
    c = jnp.asarray(rng.normal(size=(batch, length, state)), jnp.bfloat16)
    b = jnp.asarray(rng.normal(size=(batch, length, state)), jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(batch, length, heads * width)), jnp.bfloat16)
    dt = jnp.asarray(rng.uniform(0.001, 0.1, size=(batch, length, heads)), F32)
    adt = -dt * 2
    trap = jnp.asarray(rng.normal(size=dt.shape), F32)
    angles = jnp.asarray(rng.normal(size=(batch, length, 32)), F32)
    bias = jnp.asarray(rng.normal(size=(heads, 1, state)) * 0.1, F32)
    return c, b, x, adt, dt, trap, angles, bias, bias


def case_fused_forward(_):
    from lm.kernels.flashmamba import mamba3_fused

    args = fused_inputs()
    function = jax.jit(lambda *a: mamba3_fused(*a[:7], q_bias=a[7], k_bias=a[8]))
    return jax.block_until_ready(function(*args))


def case_fused_backward(_):
    from lm.kernels.flashmamba import mamba3_fused

    args = fused_inputs()

    def loss(*a):
        y, final = mamba3_fused(*a[:7], q_bias=a[7], k_bias=a[8])
        return jnp.sum(y.astype(F32) ** 2) + jnp.sum(final)

    grads = jax.jit(jax.grad(loss, argnums=(0, 2, 4, 6, 7)))(*args)
    return jax.block_until_ready(grads)


CASES = {
    name[5:].replace("_", "-"): function
    for name, function in dict(globals()).items()
    if name.startswith("case_")
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    options = parser.parse_args()
    if options.case:
        x = jnp.asarray(np.random.default_rng(1).normal(size=(128, 128)), F32)
        CASES[options.case](x)
        print(f"{options.case}: ok", flush=True)
        return
    for name in CASES:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.mosaic_probe", "--case", name],
            capture_output=True,
            text=True,
            timeout=900,
        )
        lines = (result.stdout + result.stderr).strip().splitlines()
        failure = next(
            (line for line in lines if "Check failed" in line or "Error" in line), ""
        )
        print(f"{name:18s} exit {result.returncode:4d} {failure[:220]}", flush=True)


if __name__ == "__main__":
    main()
