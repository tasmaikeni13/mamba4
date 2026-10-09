"""Trained in-context regression with predictive variance, against exact Bayes.

Each sequence draws w ~ N(0, I_8) and 32 examples x_i ~ N(0, I_8),
y_i = w·x_i + 0.5 ε_i. Positions alternate x_i and y_i; at every x_i the model
outputs a mean and a log variance for y_i, trained by Gaussian negative log
likelihood. The Bayes-optimal predictor under this exact model is the
posterior predictive (ridge with prior precision 1 and noise variance 0.25).
Scores per example index: squared error, Gaussian NLL and 90% interval
coverage for the model and for Bayes. This is the trained counterpart of
Theorems 5.4/5.7 (in-context regression risk) and Proposition 5.6
(calibrated predictive variance); a model can match Bayes only by learning
both the regression and its uncertainty.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from lm.runtime import atomic_json
from validation.synthetic import SPECS, _local, compiled

DIM, EXAMPLES, NOISE = 8, 32, 0.5
SEEDS = {"train": 11, "development": 12, "evaluation": 13}
Z90 = 1.6448536269514722


def regression_batch(rng, size):
    w = rng.normal(size=(size, DIM))
    x = rng.normal(size=(size, EXAMPLES, DIM))
    y = np.einsum("bed,bd->be", x, w) + NOISE * rng.normal(size=(size, EXAMPLES))
    features = np.zeros((size, 2 * EXAMPLES, DIM + 2), np.float32)
    features[:, 0::2, :DIM] = x
    features[:, 1::2, DIM] = y
    features[:, 1::2, DIM + 1] = 1
    targets = np.zeros((size, 2 * EXAMPLES), np.float32)
    mask = np.zeros((size, 2 * EXAMPLES), np.float32)
    targets[:, 0::2] = y
    mask[:, 0::2] = 1
    return {"features": features, "targets": targets, "mask": mask, "x": x, "y": y}


def bayes(x, y):
    """Exact posterior predictive mean and variance for every example index."""
    size = x.shape[0]
    means = np.zeros((size, EXAMPLES))
    variances = np.zeros((size, EXAMPLES))
    for i in range(EXAMPLES):
        seen_x, seen_y = x[:, :i], y[:, :i]
        precision = np.eye(DIM) + np.einsum("bed,bef->bdf", seen_x, seen_x) / NOISE**2
        rhs = np.einsum("bed,be->bd", seen_x, seen_y) / NOISE**2
        factor = np.linalg.cholesky(precision)
        mean_w = np.linalg.solve(
            np.swapaxes(factor, 1, 2), np.linalg.solve(factor, rhs[..., None])
        )[..., 0]
        query = x[:, i]
        solved = np.linalg.solve(factor, query[..., None])[..., 0]
        means[:, i] = np.einsum("bd,bd->b", query, mean_w)
        variances[:, i] = NOISE**2 + np.einsum("bd,bd->b", solved, solved)
    return means, variances


def build_model(spec):
    from flax import linen as nn

    from lm.models.common import RMSNorm
    from lm.models.mamba3 import Mamba3Block
    from lm.models.mamba4 import SelectiveMemoryBlock
    from lm.models.transformer import TransformerBlock

    blocks = {"A": TransformerBlock, "S": Mamba3Block, "M": SelectiveMemoryBlock}
    config = spec.model_config()

    class Regressor(nn.Module):
        @nn.compact
        def __call__(self, features, train=False):
            x = nn.Dense(config.d_model, name="input")(features)
            for index, kind in enumerate(spec.kinds):
                x = blocks[kind](config, name=f"layer_{index}")(x, train)
            x = RMSNorm(config.d_model, name="final_norm")(x).astype("float32")
            out = nn.Dense(2, name="head")(x)
            return out[..., 0], out[..., 1]

    return Regressor()


def gaussian_nll(mean, log_variance, target):
    import jax.numpy as jnp

    return 0.5 * (
        jnp.log(2 * jnp.pi)
        + log_variance
        + (target - mean) ** 2 / jnp.exp(log_variance)
    )


def train_one(spec, learning_rate, steps, global_batch, log_every=200):
    import jax
    import jax.numpy as jnp
    import optax

    model = build_model(spec)
    sample = regression_batch(np.random.default_rng(0), 1)["features"]
    params = model.init(jax.random.PRNGKey(0), sample)["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        learning_rate / 20, learning_rate, steps // 20, steps, learning_rate / 10
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            schedule,
            b1=0.9,
            b2=0.98,
            weight_decay=0.1,
            mask=jax.tree_util.tree_map(lambda p: p.ndim >= 2, params),
        ),
    )

    def local_loss(p, data):
        mean, log_variance = model.apply({"params": p}, data["features"], train=True)
        loss = gaussian_nll(mean, log_variance, data["targets"]) * data["mask"]
        count = jax.lax.psum(jnp.sum(data["mask"]), "data")
        return jnp.sum(loss) / count

    def train_step(p, state, data):
        loss, grads = jax.value_and_grad(local_loss)(p, data)
        grads = jax.lax.psum(grads, "data")
        updates, state = tx.update(grads, state, p)
        return optax.apply_updates(p, updates), state, jax.lax.psum(loss, "data")

    def predict(p, data):
        return model.apply({"params": p}, data["features"])

    step = jax.pmap(train_step, axis_name="data", donate_argnums=(0, 1))
    predict = jax.pmap(predict, axis_name="data")
    params = jax.device_put_replicated(params, jax.local_devices())
    label = f"regression-{spec.name}-{learning_rate}"
    opt_state = compiled(jax.pmap(tx.init), f"init-{label}", params)(params)
    rng = np.random.default_rng(SEEDS["train"])
    curve, executable, started = [], None, time.perf_counter()
    for index in range(steps):
        data = regression_batch(rng, global_batch)
        local = _local(
            {k: data[k] for k in ("features", "targets", "mask")}, global_batch
        )
        if executable is None:
            executable = compiled(step, f"train-{label}", params, opt_state, local)
        params, opt_state, loss = executable(params, opt_state, local)
        if index % log_every == 0 or index == steps - 1:
            value = float(np.asarray(loss)[0])
            curve.append([index, value])
            if not np.isfinite(value):
                break
    return params, predict, curve, time.perf_counter() - started


def score(predict, params, seed, rows, global_batch):
    import jax
    from jax.experimental import multihost_utils

    rng = np.random.default_rng(seed)
    collected = {"mean": [], "log_variance": [], "x": [], "y": []}
    executable = None
    for _ in range(-(-rows // global_batch)):
        data = regression_batch(rng, global_batch)
        local = _local({"features": data["features"]}, global_batch)
        if executable is None:
            executable = compiled(predict, f"predict-{seed}", params, local)
        mean, log_variance = jax.device_get(executable(params, local))
        for name, value in (("mean", mean), ("log_variance", log_variance)):
            gathered = np.asarray(multihost_utils.process_allgather(np.asarray(value)))
            collected[name].append(gathered.reshape(global_batch, -1)[:, 0::2])
        collected["x"].append(data["x"])
        collected["y"].append(data["y"])
    arrays = {k: np.concatenate(v)[:rows] for k, v in collected.items()}
    bayes_mean, bayes_variance = bayes(arrays["x"], arrays["y"])
    y = arrays["y"]
    variance = np.exp(arrays["log_variance"])

    def nll(mean, var):
        return 0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)

    def coverage(mean, var):
        return np.abs(y - mean) <= Z90 * np.sqrt(var)

    per_index = {
        "model_mse": ((arrays["mean"] - y) ** 2).mean(0),
        "bayes_mse": ((bayes_mean - y) ** 2).mean(0),
        "model_nll": nll(arrays["mean"], variance).mean(0),
        "bayes_nll": nll(bayes_mean, bayes_variance).mean(0),
        "model_coverage90": coverage(arrays["mean"], variance).mean(0),
        "bayes_coverage90": coverage(bayes_mean, bayes_variance).mean(0),
    }
    late = slice(EXAMPLES // 2, EXAMPLES)
    return {
        "rows": rows,
        "per_example_index": {k: v.round(5).tolist() for k, v in per_index.items()},
        "late_examples": {k: float(v[late].mean()) for k, v in per_index.items()},
        "all_examples": {k: float(v.mean()) for k, v in per_index.items()},
    }


def run(options):
    import jax

    from lm.runtime import initialize

    hardware = initialize(not options.local)
    jax.config.update("jax_enable_compilation_cache", False)
    output = Path(options.output)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "steps": options.steps,
        "global_batch": options.batch,
        "task": {"dim": DIM, "examples": EXAMPLES, "noise": NOISE},
        "runs": {},
    }
    for name in options.models:
        spec = SPECS[name]
        candidates, best = {}, None
        for learning_rate in options.learning_rates:
            params, predict, curve, seconds = train_one(
                spec, learning_rate, options.steps, options.batch
            )
            development = score(
                predict, params, SEEDS["development"], options.rows, options.batch
            )
            value = development["all_examples"]["model_nll"]
            finite = bool(np.isfinite(value))
            candidates[str(learning_rate)] = {
                "curve": curve,
                "seconds": seconds,
                "finite": finite,
                "development_nll": value if finite else None,
            }
            if finite and (best is None or value < best[0]):
                best = (value, learning_rate, params, predict)
            print(name, learning_rate, f"dev nll={value:.4f}", f"{seconds:.0f}s")
        if best is None:
            results["runs"][name] = {"candidates": candidates, "evaluation": None}
            continue
        _, learning_rate, params, predict = best
        evaluation = score(
            predict, params, SEEDS["evaluation"], options.rows, options.batch
        )
        results["runs"][name] = {
            "spec": dataclasses.asdict(spec),
            "selected_learning_rate": learning_rate,
            "candidates": candidates,
            "evaluation": evaluation,
        }
        print(name, "eval", evaluation["late_examples"], flush=True)
        if jax.process_index() == 0:
            atomic_json(output / "regression.json", results)
    if not options.local:
        jax.distributed.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["transformer", "mamba3", "mamba4", "mamba4-shift"],
    )
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=[5e-4, 1.5e-3, 5e-3]
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--local", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
