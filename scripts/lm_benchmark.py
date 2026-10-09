"""Compile and measure the actual all-device LM paths on synthetic tokens.

Run via scripts.pod on every worker; this never opens the training corpus.
All timings block until device results are ready and separate compilation
from steady execution. Cost/memory reports are compiler or runtime evidence,
not inferred from the architecture name. Decode is measured separately when
an actual cache API is available.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import time

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from lm.config import ModelConfig, TrainConfig
from lm.runtime import atomic_json, initialize, source_provenance
from lm.train import (
    create_model,
    make_diagnostic_step,
    make_steps,
    optimizer,
    parameter_ledger,
)


def compiler_report(executable):
    report = {}
    for name in ("memory_analysis", "cost_analysis"):
        try:
            value = getattr(executable, name)()
            if value is not None:
                if hasattr(value, "_asdict"):
                    value = value._asdict()
                elif not isinstance(value, (dict, list, int, float, str)):
                    value = {
                        field: getattr(value, field)
                        for field in dir(value)
                        if field.endswith("_size_in_bytes")
                    }
                report[name] = value
        except (AttributeError, NotImplementedError, RuntimeError) as error:
            report[name] = {"unavailable": str(error)}
    return report


def timed(function, *args):
    started = time.monotonic()
    result = function(*args)
    jax.block_until_ready(result)
    return result, time.monotonic() - started


def run(args):
    hardware = initialize(not args.single_host)
    frozen = json.loads(Path(args.config).read_text())
    config = ModelConfig(**frozen["model"])
    training = TrainConfig(**frozen["training"])
    batch_size = args.global_batch or training.global_batch
    length = args.sequence_length or training.sequence_length
    if batch_size % jax.device_count():
        raise ValueError("Benchmark batch must divide all devices")
    local_devices = jax.local_device_count()
    per_device = batch_size // jax.device_count()
    model = create_model(config)
    params = model.init(
        jax.random.PRNGKey(training.seed),
        jnp.zeros((1, config.chunk_size), jnp.int32),
    )["params"]
    tx, _ = optimizer(training, params)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    ledger = parameter_ledger(params)
    replicated = jax.device_put_replicated(state, jax.local_devices())
    shape = (local_devices, per_device, length)
    tokens = (np.arange(np.prod(shape)).reshape(shape) % config.vocab_size).astype(
        np.int32
    )
    batch = {
        "input_ids": tokens,
        "targets": (tokens + 1) % config.vocab_size,
        "loss_mask": np.ones(shape, np.float32),
    }
    train_step, eval_step = make_steps(model)
    started = time.monotonic()
    train_executable = train_step.lower(replicated, batch).compile()
    train_compile = time.monotonic() - started
    training_times = []
    for _ in range(args.steps + 1):
        (replicated, metrics), seconds = timed(train_executable, replicated, batch)
        training_times.append(seconds)
    started = time.monotonic()
    eval_executable = eval_step.lower(replicated.params, batch).compile()
    eval_compile = time.monotonic() - started
    evaluation_times = []
    for _ in range(args.steps + 1):
        metrics_eval, seconds = timed(eval_executable, replicated.params, batch)
        evaluation_times.append(seconds)
    result = {
        "provenance": source_provenance(),
        "architecture": config.architecture,
        "model_config": dataclasses.asdict(config),
        "hardware": hardware,
        "parameter_ledger": ledger,
        "global_batch": batch_size,
        "sequence_length": length,
        "compile_seconds": {"train": train_compile, "prefill_loss": eval_compile},
        "train": {
            "first_seconds": training_times[0],
            "steady_seconds": training_times[1:],
            "tokens_per_second": batch_size
            * length
            / float(np.median(training_times[1:])),
            "loss": float(jax.device_get(metrics["loss"])[0]),
            "grad_norm": float(jax.device_get(metrics["grad_norm"])[0]),
            "compiler": compiler_report(train_executable),
        },
        "prefill_loss": {
            "first_seconds": evaluation_times[0],
            "steady_seconds": evaluation_times[1:],
            "tokens_per_second": batch_size
            * length
            / float(np.median(evaluation_times[1:])),
            "nll_sum": float(jax.device_get(metrics_eval["nll_sum"])[0]),
            "compiler": compiler_report(eval_executable),
        },
        "runtime_memory": [device.memory_stats() for device in jax.local_devices()],
    }
    if config.architecture == "mamba4":
        diagnostic_step = make_diagnostic_step(model)
        started = time.monotonic()
        diagnostic_executable = diagnostic_step.lower(
            replicated.params, batch["input_ids"]
        ).compile()
        diagnostic_compile = time.monotonic() - started
        diagnostic_metrics, diagnostic_seconds = timed(
            diagnostic_executable, replicated.params, batch["input_ids"]
        )
        diagnostic_metrics = {
            name: float(jax.device_get(value)[0])
            for name, value in diagnostic_metrics.items()
        }
        finite = all(np.isfinite(value) for value in diagnostic_metrics.values())
        finite_flags = all(
            value == 1
            for name, value in diagnostic_metrics.items()
            if name.endswith("allfinite")
        )
        eigenvalue_minima = [
            value
            for name, value in diagnostic_metrics.items()
            if name.endswith("precision_eigenvalue_min")
        ]
        positive_precision = bool(eigenvalue_minima) and min(eigenvalue_minima) > 0
        result["memory_diagnostics"] = diagnostic_metrics
        result["memory_diagnostic_execution"] = {
            "compile_seconds": diagnostic_compile,
            "seconds": diagnostic_seconds,
            "compiler": compiler_report(diagnostic_executable),
            "passed": finite and finite_flags and positive_precision,
        }
        if not result["memory_diagnostic_execution"]["passed"]:
            atomic_json(
                Path(args.output)
                / f"{config.architecture}-failed-diagnostics-host-{jax.process_index()}.json",
                result,
            )
            raise RuntimeError(
                f"Memory factor diagnostics failed: {diagnostic_metrics}"
            )
    if args.decode:
        from lm.decode import decode_step, initialize_cache

        decode_config = dataclasses.replace(
            config, max_seq_len=max(length, args.decode_context + args.decode_steps + 1)
        )
        cache = initialize_cache(decode_config, per_device)
        cache = jax.device_put_replicated(cache, jax.local_devices())
        decode_tokens = np.ones((local_devices, per_device), np.int32)
        prime_compile, prime_seconds = 0.0, 0.0
        if args.decode_context:

            def prime(parameters, cached, ids):
                def step(carry, position):
                    _, carry = decode_step(
                        decode_config, parameters, ids, carry, position
                    )
                    return carry, None

                return jax.lax.scan(
                    step, cached, jnp.arange(args.decode_context, dtype=jnp.int32)
                )[0]

            mapped_prime = jax.pmap(prime, donate_argnums=(1,))
            started = time.monotonic()
            prime_executable = mapped_prime.lower(
                replicated.params, cache, decode_tokens
            ).compile()
            prime_compile = time.monotonic() - started
            cache, prime_seconds = timed(
                prime_executable, replicated.params, cache, decode_tokens
            )
        mapped_decode = jax.pmap(
            lambda parameters, ids, cached, position: decode_step(
                decode_config, parameters, ids, cached, position
            ),
            in_axes=(0, 0, 0, None),
            donate_argnums=(2,),
        )
        started = time.monotonic()
        executable = mapped_decode.lower(
            replicated.params, decode_tokens, cache, jnp.array(0, jnp.int32)
        ).compile()
        decode_compile = time.monotonic() - started
        decode_times = []
        for position in range(
            args.decode_context, args.decode_context + args.decode_steps + 1
        ):
            (logits, cache), seconds = timed(
                executable,
                replicated.params,
                decode_tokens,
                cache,
                jnp.array(position, jnp.int32),
            )
            decode_times.append(seconds)
        result["cached_decode"] = {
            "prefix_context_tokens": args.decode_context,
            "cache_capacity_tokens": decode_config.max_seq_len,
            "context_warmup_compile_seconds": prime_compile,
            "context_warmup_seconds": prime_seconds,
            "compile_seconds": decode_compile,
            "first_seconds": decode_times[0],
            "steady_seconds": decode_times[1:],
            "tokens_per_second": batch_size / float(np.median(decode_times[1:])),
            "cache_bytes_per_device": sum(
                leaf.size * leaf.dtype.itemsize
                for leaf in jax.tree.leaves(initialize_cache(decode_config, per_device))
            ),
            "compiler": compiler_report(executable),
            "finite_logits": bool(np.all(np.isfinite(jax.device_get(logits)))),
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / f"{config.architecture}-host-{jax.process_index()}.json", result
    )
    if jax.process_index() == 0:
        print(json.dumps(result), flush=True)
    if not args.single_host:
        jax.distributed.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="lm/results/benchmarks")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--decode-context", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.decode_steps < 1 or args.decode_context < 0:
        parser.error("At least one steady sample is required")
    run(args)


if __name__ == "__main__":
    main()
