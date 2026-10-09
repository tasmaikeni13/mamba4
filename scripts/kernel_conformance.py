"""Measure fused TPU forward/backward errors against independent references.

The default requires TPU execution and uses pmap on every local chip on every
worker. The optional CPU invocation only smoke-checks the script; its report
explicitly records that no Pallas TPU kernel was verified.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lm.config import ModelConfig
from lm.decode import decode_step, initialize_cache
from lm.kernels.attention import attention_reference, causal_attention
from lm.kernels.mamba3 import mamba3_chunked, mamba3_sequential
from lm.kernels.mamba4 import (
    gaussian_memory,
    protected_cascade_memory,
    protected_diagnostics,
)
from lm.kernels.mamba4_reference import dense_reference
from lm.models.mamba4 import Mamba4LM
from lm.runtime import atomic_json, initialize, source_provenance


def errors(actual, expected, tolerance):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    difference = actual - expected
    maximum = float(np.max(np.abs(difference)))
    relative_max = maximum / max(float(np.max(np.abs(expected))), 1e-10)
    relative_l2 = float(np.linalg.norm(difference)) / max(
        float(np.linalg.norm(expected)), 1e-10
    )
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
    return {
        "max_absolute_error": maximum,
        "max_error_over_reference_max": relative_max,
        "relative_l2_error": relative_l2,
        "tolerance": tolerance,
        "finite": finite,
        "passed": finite and relative_max <= tolerance and relative_l2 <= tolerance,
    }


def replicated(array, devices):
    return np.broadcast_to(np.asarray(array), (devices, *array.shape)).copy()


def compare_case(
    name,
    production,
    reference,
    arrays,
    cotangent,
    forward_tolerance,
    gradient_tolerance,
):
    def objective(function, *inputs):
        output = function(*inputs)
        return jnp.sum(
            output.astype(jnp.float32) * cotangent.astype(jnp.float32)
        ), output

    argnums = tuple(range(len(arrays)))
    actual_fn = jax.pmap(
        jax.value_and_grad(
            lambda *inputs: objective(production, *inputs),
            argnums=argnums,
            has_aux=True,
        )
    )
    reference_fn = jax.pmap(
        jax.value_and_grad(
            lambda *inputs: objective(reference, *inputs), argnums=argnums, has_aux=True
        )
    )
    mapped_arrays = tuple(
        replicated(array, jax.local_device_count()) for array in arrays
    )
    (_, actual), gradients = actual_fn(*mapped_arrays)
    (_, expected), reference_gradients = reference_fn(*mapped_arrays)
    actual, expected, gradients, reference_gradients = jax.device_get(
        (actual, expected, gradients, reference_gradients)
    )
    report = {
        "name": name,
        "input_shapes": [list(array.shape) for array in arrays],
        "input_dtypes": [str(array.dtype) for array in arrays],
        "local_chips_checked": jax.local_device_count(),
        "forward": errors(actual, expected, forward_tolerance),
        "gradients": [
            errors(gradient, reference_gradient, gradient_tolerance)
            for gradient, reference_gradient in zip(gradients, reference_gradients)
        ],
    }
    report["passed"] = report["forward"]["passed"] and all(
        entry["passed"] for entry in report["gradients"]
    )
    return report


def attention_cases(rng):
    reports = []
    for length, heads in ((128, 2), (256, 4)):
        shape = (1, length, heads, 64)
        arrays = tuple(
            jnp.asarray(rng.normal(size=shape), jnp.bfloat16) for _ in range(3)
        )
        cotangent = jnp.asarray(rng.normal(size=shape), jnp.float32)

        def reference(q, k, v):
            with jax.default_matmul_precision("highest"):
                return attention_reference(
                    q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
                )

        reports.append(
            compare_case(
                f"attention_bf16_T{length}_H{heads}",
                causal_attention,
                reference,
                arrays,
                cotangent,
                0.035,
                0.075,
            )
        )
    return reports


def mamba3_cases(rng):
    reports = []
    for length, rank in ((128, 1), (139, 1), (65, 2)):
        heads, state_dim, channels = 4, 96, 64
        qshape = (1, length, heads, rank, state_dim)
        q, k = (
            jnp.asarray(rng.normal(size=qshape) * 0.5 + 1, jnp.bfloat16)
            for _ in range(2)
        )
        v = jnp.asarray(rng.normal(size=(*qshape[:4], channels)) * 0.2, jnp.bfloat16)
        dt = jnp.asarray(rng.uniform(0.01, 0.08, size=qshape[:3]), jnp.float32)
        adt = -dt * jnp.asarray(rng.uniform(0.1, 3, size=qshape[:3]), jnp.float32)
        trap = jnp.asarray(rng.normal(size=qshape[:3]), jnp.float32)
        angles = jnp.asarray(
            rng.normal(size=(*qshape[:3], state_dim // 4)) * 0.2, jnp.float32
        )
        arrays = (q, k, v, adt, dt, trap, angles)
        cotangent = jnp.asarray(rng.normal(size=v.shape), jnp.float32)
        pairwise = rank == 1

        def production(*inputs):
            return mamba3_chunked(*inputs, chunk_size=64, pairwise=pairwise)[0]

        def reference(*inputs):
            with jax.default_matmul_precision("highest"):
                inputs = tuple(array.astype(jnp.float32) for array in inputs)
                return mamba3_sequential(*inputs, pairwise=pairwise)[0]

        reports.append(
            compare_case(
                f"mamba3_bf16_T{length}_R{rank}",
                production,
                reference,
                arrays,
                cotangent,
                0.04,
                0.08,
            )
        )
        if length == 139:

            def production_final(*inputs):
                return mamba3_chunked(*inputs, chunk_size=64, pairwise=pairwise)[1]

            def reference_final(*inputs):
                with jax.default_matmul_precision("highest"):
                    inputs = tuple(array.astype(jnp.float32) for array in inputs)
                    return mamba3_sequential(*inputs, pairwise=pairwise)[1]

            final_cotangent = jnp.asarray(
                rng.normal(size=(1, heads, state_dim, channels)), jnp.float32
            )
            reports.append(
                compare_case(
                    "mamba3_bf16_T139_finalstate",
                    production_final,
                    reference_final,
                    arrays,
                    final_cotangent,
                    0.04,
                    0.08,
                )
            )
    return reports


def mamba4_cases(rng):
    reports = []
    for floor in ("cyclic", "fixed"):
        shape = (1, 5, 2, 4)
        keys, queries = (
            jnp.asarray(rng.normal(size=shape), jnp.bfloat16) for _ in range(2)
        )
        values = jnp.asarray(rng.normal(size=(*shape[:3], 3)), jnp.bfloat16)
        beta = jnp.asarray(rng.uniform(0.1, 1, size=shape[:3]), jnp.float32)
        decay = (
            jnp.asarray([0.96, 0.98], jnp.float32)
            if floor == "cyclic"
            else jnp.asarray(rng.uniform(0.9, 0.99, size=shape[:3]), jnp.float32)
        )
        epsilon = jnp.asarray([0.1, 0.2], jnp.float32)
        arrays = (keys, values, queries, beta, decay, epsilon)
        # The variance term also verifies the solve and epsilon/floor gradient.
        cotangent = jnp.asarray(rng.normal(size=(*shape[:3], 4)), jnp.float32)

        def production(*inputs):
            result = gaussian_memory(*inputs, floor=floor, chunk_size=3)
            return jnp.concatenate(
                (result.output, result.variance[..., None] * 0.01), axis=-1
            )

        def reference(*inputs):
            with jax.default_matmul_precision("highest"):
                output, variance, *_ = dense_reference(
                    *(array.astype(jnp.float32) for array in inputs), floor
                )
                return jnp.concatenate((output, variance[..., None] * 0.01), axis=-1)

        reports.append(
            compare_case(
                f"mamba4_bf16_{floor}",
                production,
                reference,
                arrays,
                cotangent,
                0.002,
                0.025,
            )
        )
    return reports


def protected_cases(rng):
    reports = []
    for residual in (0.0, 2e-3):
        keys = np.broadcast_to(
            np.array([1.0, 0.0, 0.0, 0.0], np.float32), (1, 12, 1, 4)
        ).copy()
        keys[:, 1::3, :, 1] = residual
        keys[:, 2::3, :, 2] = residual
        values = rng.normal(size=(1, 12, 1, 3)).astype(np.float32)
        queries = np.broadcast_to(
            np.array([1.0, 0.2, -0.1, 0.3], np.float32), keys.shape
        ).copy()

        def loss(k, v, q):
            result = protected_cascade_memory(k, v, q, budget=3, block_size=3)
            return jnp.sum(jnp.tanh(result.output)), protected_diagnostics(result.state)

        mapped = tuple(
            replicated(array, jax.local_device_count())
            for array in (keys, values, queries)
        )
        (value, diagnostics), gradients = jax.pmap(
            jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)
        )(*mapped)
        value, diagnostics, gradients = jax.device_get((value, diagnostics, gradients))
        finite = bool(
            np.all(np.isfinite(value))
            and all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
        )
        rank_margin = bool(np.all(diagnostics["protected_min_qr_diagonal"] > 1e-5))
        banks_finite = bool(np.all(diagnostics["protected_all_finite"]))
        reports.append(
            {
                "name": f"mamba4_protected_repeated_rankcut_{residual}",
                "local_chips_checked": jax.local_device_count(),
                "finite_loss_and_all_key_value_query_gradients": finite,
                "positive_retained_qr_diagonals": rank_margin,
                "diagnostics": {
                    key: np.asarray(value).tolist()
                    for key, value in diagnostics.items()
                },
                "passed": finite and rank_margin and banks_finite,
            }
        )
    config = ModelConfig(
        architecture="mamba4",
        vocab_size=31,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=24,
        expand=1,
        head_dim=8,
        key_dim=4,
        chunk_size=16,
        protected_anchor_budget=2,
        protected_block_size=64,
        dtype="bfloat16",
        max_seq_len=128,
    )
    model = Mamba4LM(config)
    tokens = (np.arange(65)[None] % config.vocab_size).astype(np.int32)
    # Repeated EOS is an especially important full-parameter NaN regression.
    tokens[:, 20:30] = 2
    params = model.init(jax.random.PRNGKey(811), jnp.asarray(tokens))["params"]
    params = jax.tree.map(
        lambda array: replicated(array, jax.local_device_count()), params
    )
    mapped_tokens = replicated(tokens, jax.local_device_count())
    expected = jax.pmap(lambda p, ids: model.apply({"params": p}, ids))(
        params, mapped_tokens
    )
    cache = jax.tree.map(
        lambda array: replicated(array, jax.local_device_count()),
        initialize_cache(config, 1),
    )

    def cached(parameters, ids, cached_state):
        def step(carry, inputs):
            token, position = inputs
            logits, carry = decode_step(config, parameters, token, carry, position)
            return carry, logits

        _, outputs = jax.lax.scan(
            step,
            cached_state,
            (jnp.swapaxes(ids, 0, 1), jnp.arange(ids.shape[1], dtype=jnp.int32)),
        )
        return jnp.swapaxes(outputs, 0, 1)

    actual = jax.pmap(cached)(params, mapped_tokens, cache)
    conformity = errors(*jax.device_get((actual, expected)), 0.035)
    eos = np.full(tokens.shape, 2, np.int32)

    def eos_loss(parameters, ids):
        logits = model.apply({"params": parameters}, ids)
        return -jnp.mean(
            jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[..., 2]
        )

    loss, gradients = jax.pmap(jax.value_and_grad(eos_loss))(
        params, replicated(eos, jax.local_device_count())
    )
    loss, gradients = jax.device_get((loss, gradients))
    finite = bool(
        np.all(np.isfinite(loss))
        and all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    )
    reports.append(
        {
            "name": "mamba4_bf16_protected_cached_vs_prefill_T65_and_eos_full_gradients",
            "local_chips_checked": jax.local_device_count(),
            "protected_block_size": 64,
            "context_tokens": 65,
            "cached_decode": conformity,
            "finite_repeated_eos_loss_and_all_parameter_gradients": finite,
            "passed": conformity["passed"] and finite,
        }
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="lm/results/kernel-conformance")
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    hardware = initialize(not args.single_host)
    if hardware["backend"] != "tpu" and not args.allow_cpu:
        raise RuntimeError("This conformance gate requires actual TPU execution")
    rng = np.random.default_rng(702)
    reports = (
        attention_cases(rng)
        + mamba3_cases(rng)
        + mamba4_cases(rng)
        + protected_cases(rng)
    )
    passed = all(report["passed"] for report in reports)
    result = {
        "hardware": hardware,
        "provenance": source_provenance(),
        "pallas_tpu_verified": hardware["backend"] == "tpu",
        "precision": "bf16 production operands, fp32 accumulators; highest-precision independent reference",
        "cases": reports,
        "passed": passed,
    }
    atomic_json(Path(args.output) / f"host-{jax.process_index()}.json", result)
    print(
        json.dumps(
            {
                "host": hardware["hostname"],
                "process_index": jax.process_index(),
                "passed": passed,
                "cases": [
                    {"name": report["name"], "passed": report["passed"]}
                    for report in reports
                ],
            }
        ),
        flush=True,
    )
    if not args.single_host:
        jax.distributed.shutdown()
    if not passed:
        raise RuntimeError(
            "Kernel forward/backward conformance failed; inspect the saved errors"
        )


if __name__ == "__main__":
    main()
