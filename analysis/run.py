"""Run the frozen operator protocol and save trial-level evidence.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for predictable CPU use.
The --quick mode is for conformance/CI only and never certifies research gates.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import norm, ttest_1samp

from analysis.extensions import drifting_operator, interpolation_bounds

from analysis.memory import (
    AffineSummary,
    CyclicFloorMemory,
    EvidenceState,
    ProtectedCascade,
    delta_read,
    diagonal_read,
    effective_weights,
    hebbian_read,
    interpolation,
    mamba3_read,
    softmax_read,
    tree_prefix,
    unit_rows,
)
from analysis.statistics import holm, mean_interval, paired_bootstrap, wilson


def seeded(config, name):
    words = np.frombuffer(hashlib.sha256(name.encode()).digest()[:16], dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([config["root_seed"], *words]))


def geometry(rng, count, dim, name):
    if name == "orthogonal":
        rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        # Above d, different arbitrary values reuse an address. This is an
        # explicitly inconsistent-key collision scenario, not independent keys.
        return rotation[:, np.arange(count) % dim].T
    if name == "random":
        return unit_rows(rng.normal(size=(count, dim)))
    if name == "clustered":
        common = unit_rows(rng.normal(size=(1, dim)))
        return unit_rows(common + 0.1 * rng.normal(size=(count, dim)))
    raise ValueError(name)


def read_models(keys, values, queries, chosen):
    state = EvidenceState.from_batch(keys, values)
    result = {
        "ridge": state.read(queries, chosen["ridge"]),
        "softmax": softmax_read(keys, values, queries, chosen["softmax"]),
        "hebbian": hebbian_read(keys, values, queries),
        "diagonal": diagonal_read(keys, values, queries, chosen["diagonal"]),
        "delta": delta_read(keys, values, queries, chosen["delta"]),
        "mamba3_siso_core": mamba3_read(keys, values, queries, **chosen["mamba3"]),
        "mamba3_mimo_tied_core": mamba3_read(
            keys, values, queries, rank=2, **chosen["mamba3"]
        ),
    }
    if len(keys) <= keys.shape[1] and np.linalg.matrix_rank(keys) == len(keys):
        result["protected_qr"] = interpolation(keys, values, queries)
    return result


def tune(config, scenario, count, kind, functional=False):
    rng = seeded(config, "development/" + scenario)
    candidates = {
        "ridge": config["ridge_grid"],
        "softmax": config["temperature_grid"],
        "diagonal": config["ridge_grid"],
        "delta": config["delta_grid"],
        "mamba3": config["mamba3_grid"],
    }
    losses = {name: np.zeros(len(grid)) for name, grid in candidates.items()}
    for _ in range(config["development_trials"]):
        k = geometry(rng, count, config["key_dim"], kind)
        v = rng.normal(size=(count, config["value_dim"]))
        if functional:
            alpha = rng.normal(size=(4, count))
            q, target = alpha @ k, alpha @ v
        else:
            q, target = k, v
        state = EvidenceState.from_batch(k, v)
        functions = {
            "ridge": lambda candidate: state.read(q, candidate),
            "softmax": lambda candidate: softmax_read(k, v, q, candidate),
            "diagonal": lambda candidate: diagonal_read(k, v, q, candidate),
            "delta": lambda candidate: delta_read(k, v, q, candidate),
            "mamba3": lambda candidate: mamba3_read(k, v, q, **candidate),
        }
        for name, grid in candidates.items():
            for i, candidate in enumerate(grid):
                losses[name][i] += np.mean((functions[name](candidate) - target) ** 2)
    chosen = {
        name: grid[int(np.argmin(losses[name]))] for name, grid in candidates.items()
    }
    diagnostics = {
        name: (loss / config["development_trials"]).tolist()
        for name, loss in losses.items()
    }
    return chosen, diagnostics


def recall(config, raw):
    rows, comparisons, selections = [], [], {}
    for kind in config["geometries"]:
        for count in config["loads"]:
            scenario = f"recall/{kind}/K{count}"
            selected, dev = tune(config, scenario, count, kind)
            selections[scenario] = {"selected": selected, "development_mse": dev}
            rng = seeded(config, "evaluation/" + scenario)
            losses, accuracy = {}, {}
            conditions = []
            for _ in range(config["evaluation_trials"]):
                k = geometry(rng, count, config["key_dim"], kind)
                v = rng.normal(size=(count, config["value_dim"]))
                conditions.append(
                    np.linalg.cond(k.T @ k + selected["ridge"] * np.eye(k.shape[1]))
                )
                for model, output in read_models(k, v, k, selected).items():
                    losses.setdefault(model, []).append(np.mean((output - v) ** 2))
                    classification = np.argmin(
                        np.sum((output[:, None] - v[None]) ** 2, axis=-1), axis=-1
                    )
                    accuracy.setdefault(model, []).append(
                        np.mean(classification == np.arange(count))
                    )
            raw[scenario + "/condition"] = np.array(conditions)
            for model, loss in losses.items():
                raw[scenario + "/" + model] = np.array(loss)
                raw[scenario + "/" + model + "/accuracy"] = np.array(accuracy[model])
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        **mean_interval(loss),
                        "recall_accuracy": float(np.mean(accuracy[model])),
                    }
                )
                if model != "ridge":
                    diff = np.asarray(loss) - losses["ridge"]
                    if np.std(diff) < 1e-30:
                        p = 0.0 if np.mean(diff) > 0 else 1.0
                    else:
                        p = float(ttest_1samp(diff, 0, alternative="greater").pvalue)
                    comparisons.append(
                        {
                            "scenario": scenario,
                            "baseline": model,
                            "p_ridge_better": p,
                            "loss": loss,
                            "ridge_loss": losses["ridge"],
                        }
                    )
            print(f"completed {scenario}", flush=True)
    family = len(comparisons)
    adjusted = holm([c["p_ridge_better"] for c in comparisons])
    final = []
    for i, c in enumerate(comparisons):
        interval = paired_bootstrap(
            c.pop("loss"),
            c.pop("ridge_loss"),
            seeded(config, "resample/" + c["scenario"] + c["baseline"]),
            config["bootstrap_draws"],
            family,
        )
        final.append(
            {**c, **interval, "holm_p": float(adjusted[i]), "family_size": family}
        )
    return rows, final, selections


def functional_queries(config, raw):
    result = {}
    for kind in ["random", "clustered"]:
        scenario = "functional/" + kind
        selected, dev = tune(config, scenario, 8, kind, functional=True)
        rng = seeded(config, "evaluation/" + scenario)
        losses = {}
        for _ in range(config["evaluation_trials"]):
            k = geometry(rng, 8, config["key_dim"], kind)
            v = rng.normal(size=(8, config["value_dim"]))
            alpha = rng.normal(size=(4, 8))
            for name, read in read_models(k, v, alpha @ k, selected).items():
                losses.setdefault(name, []).append(np.mean((read - alpha @ v) ** 2))
        result[kind] = {
            "selected": selected,
            "development_mse": dev,
            "mse": {name: mean_interval(value) for name, value in losses.items()},
        }
        for name, value in losses.items():
            raw[scenario + "/" + name] = np.asarray(value)
    return result


def regression_risk(config, raw):
    """Fixed design/operator; replicate noise to test the exact conditional risk."""
    result = {}
    for name in ["homoscedastic", "heteroscedastic", "discounted"]:
        rng = seeded(config, "evaluation/risk/" + name)
        n, d, dv, eps = 24, 8, config["value_dim"], 0.1
        k, w, q = (
            unit_rows(rng.normal(size=(n, d))),
            rng.normal(size=(dv, d)),
            rng.normal(size=d),
        )
        beta = np.geomspace(0.1, 10, n) if name == "heteroscedastic" else np.ones(n)
        decay = np.full(n, 0.9 if name == "discounted" else 1.0)
        effective = effective_weights(beta, decay)
        gamma = effective / beta
        s = k.T @ (effective[:, None] * k)
        a = s + eps * np.eye(d)
        y = np.linalg.solve(a, q)
        bias = float(eps * eps * np.linalg.norm(w @ y) ** 2)
        # Var(noise_i)=1/beta_i; the covariance uses gamma^2 beta, not S.
        noise_matrix = k.T @ ((gamma * gamma * beta)[:, None] * k)
        variance = float(dv * y @ noise_matrix @ y)
        values = (
            k @ w.T
            + rng.normal(size=(config["risk_trials"], n, dv))
            / np.sqrt(beta)[None, :, None]
        )
        reads = np.einsum("tnv,n->tv", values, effective * (k @ y))
        loss = np.sum((reads - w @ q) ** 2, axis=1)
        interval = mean_interval(loss)
        result[name] = {
            "theory": bias + variance,
            "bias": bias,
            "variance": variance,
            "paper_S_variance": float(dv * y @ s @ y),
            "empirical": interval,
            "standard_errors_from_theory": abs(interval["mean"] - bias - variance)
            / interval["se"],
        }
        raw["risk/" + name] = loss
    return result


def calibration(config, raw):
    result = {}
    total = config["calibration_trials"]
    for name in ["in_model", "misspecified", "duplicate_conflict", "heavy_tailed"]:
        rng = seeded(config, "evaluation/calibration/" + name)
        hits, standardized, losses, confidences = [], [], [], []
        for _ in range(total):
            d, n, eps = 8, 16, 0.25
            k, q = (
                unit_rows(rng.normal(size=(n, d))),
                unit_rows(rng.normal(size=(1, d)))[0],
            )
            w = rng.normal(size=d) / np.sqrt(eps)
            noise = rng.normal(size=n)
            if name == "misspecified":
                w *= 20
            if name == "heavy_tailed":
                noise = rng.standard_t(3, size=n) / np.sqrt(3)
            target = w @ q
            values = (k @ w + noise)[:, None]
            if name == "duplicate_conflict":
                k = np.tile(np.eye(d)[0], (n, 1))
                q = k[0]
                # Equal address, alternating arbitrary conflicting labels.
                values = (np.arange(n) % 2 * 20 - 10 + 0.1 * noise)[:, None]
                target = -10.0
            state = EvidenceState.from_batch(k, values)
            read = state.read(q, eps)[0, 0]
            c = state.confidence(q, eps)[0]
            residual = read - target
            hits.append(abs(residual) <= norm.ppf(0.975) * np.sqrt(c))
            standardized.append(residual / np.sqrt(c))
            losses.append(residual * residual)
            confidences.append(c)
        raw["calibration/" + name + "/z"] = np.asarray(standardized)
        raw["calibration/" + name + "/loss"] = np.asarray(losses)
        raw["calibration/" + name + "/confidence"] = np.asarray(confidences)
        result[name] = {
            "latent_95_coverage": wilson(sum(hits), total),
            "z_mean": mean_interval(standardized),
            "z_second_moment": mean_interval(np.asarray(standardized) ** 2),
            "mean_confidence": float(np.mean(confidences)),
            "mse": mean_interval(losses),
        }
    # New noisy observation: use c + beta_query^-1, separate from latent Wq.
    rng = seeded(config, "evaluation/calibration/predictive")
    hits = []
    for _ in range(total):
        k = unit_rows(rng.normal(size=(16, 8)))
        q = unit_rows(rng.normal(size=(1, 8)))[0]
        w = rng.normal(size=8) / 0.5
        v = (k @ w + rng.normal(size=16))[:, None]
        state = EvidenceState.from_batch(k, v)
        target = w @ q + rng.normal() / np.sqrt(2.0)
        hits.append(
            abs(state.read(q, 0.25)[0, 0] - target)
            <= norm.ppf(0.975) * np.sqrt(state.confidence(q, 0.25)[0] + 0.5)
        )
    result["new_observation"] = {
        "predictive_95_coverage": wilson(sum(hits), total),
        "query_precision": 2.0,
    }
    return result


def repeated_writes(config, raw):
    result = {}
    for count in [1, 4, 16, 64]:
        for kind in ["equal_noise", "unequal_noise"]:
            rng = seeded(config, f"evaluation/repeated/{kind}/{count}")
            beta = (
                np.ones(count)
                if kind == "equal_noise"
                else np.geomspace(0.1, 10, count)
            )
            noise = rng.normal(size=(config["repeated_trials"], count)) / np.sqrt(beta)
            target = 2.0
            ordinary = target + noise.mean(axis=1)
            precision = target + (noise @ beta) / beta.sum()
            ridge = (target * beta.sum() + noise @ beta) / (beta.sum() + 0.01)
            scenario = f"repeated/{kind}/{count}"
            losses = {
                "ordinary_softmax": (ordinary - target) ** 2,
                "precision_softmax": (precision - target) ** 2,
                "ridge": (ridge - target) ** 2,
            }
            result[scenario] = {
                "precision_unbiased_theory": 1 / beta.sum(),
                "ridge_theory": (0.01 * target / (beta.sum() + 0.01)) ** 2
                + beta.sum() / (beta.sum() + 0.01) ** 2,
                "ordinary_theory": np.sum(1 / beta) / count**2,
                "mse": {name: mean_interval(loss) for name, loss in losses.items()},
            }
            for name, loss in losses.items():
                raw[scenario + "/" + name] = loss
    return result


def stability_and_floor(config):
    rng = seeded(config, "evaluation/floor")
    result = []
    for d in [8, 32, 64]:
        for decay in [0.999, 0.99, 0.95]:
            eps = 0.01
            memory = CyclicFloorMemory.create(d, 2, eps, decay)
            factor_errors, read_errors = [], []
            min_floor, max_floor = float("inf"), 0.0
            for _ in range(4 * d):
                k, v = unit_rows(rng.normal(size=(1, d)))[0], rng.normal(size=2)
                memory.write(k, v)
                a = memory.state.s + np.diag(memory.prior)
                factor_errors.append(
                    np.linalg.norm(memory.lower @ memory.lower.T - a)
                    / np.linalg.norm(a)
                )
                q = rng.normal(size=(1, d))
                read_errors.append(
                    np.linalg.norm(
                        memory.read(q) - (memory.state.c @ np.linalg.solve(a, q.T)).T
                    )
                )
                min_floor = min(min_floor, memory.prior.min())
                max_floor = max(max_floor, memory.prior.max())
            old_max = d * (1 - decay) / (1 - decay**d)
            old_min = old_max * decay ** (d - 1)
            result.append(
                {
                    "dim": d,
                    "decay": decay,
                    "old_rule_min_over_eps": old_min,
                    "old_rule_max_over_eps": old_max,
                    "old_condition": (1 - decay) * d <= 1,
                    "new_min_over_eps": min_floor / eps,
                    "new_max_over_eps": max_floor / eps,
                    "new_max_theory": decay ** (-(d - 1)),
                    "max_factor_relative_error": max(factor_errors),
                    "max_read_error": max(read_errors),
                }
            )
    return result


def cascade_experiment(config, raw):
    rng = seeded(config, "evaluation/cascade")
    d, dv, count = 8, 2, 256
    protected = ProtectedCascade(d, dv)
    naive, results = {}, []
    for index in range(count):
        k, v = unit_rows(rng.normal(size=(d, d))), rng.normal(size=(d, dv))
        ids = np.arange(index * d, (index + 1) * d)
        protected.append(k, v, ids)
        block, level = (k, v, EvidenceState.from_batch(k, v)), 0
        while level in naive:
            pk, pv, ps = naive.pop(level)
            block = (
                np.concatenate([pk, block[0]]),
                np.concatenate([pv, block[1]]),
                ps.merge(block[2]),
            )
            level += 1
        naive[level] = block
        if index + 1 in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]:
            protected_loss = [
                np.mean((b.read(b.keys) - b.values) ** 2) for b in protected.blocks
            ]
            naive_loss = [
                np.mean((state.read(keys, 1e-8) - values) ** 2)
                for keys, values, state in naive.values()
            ]
            results.append(
                {
                    "blocks_ingested": index + 1,
                    "original_occupied_blocks": len(naive),
                    "original_popcount": (index + 1).bit_count(),
                    "original_all_item_mse": float(
                        np.average(
                            naive_loss, weights=[len(b[0]) for b in naive.values()]
                        )
                    ),
                    "protected_occupied_levels": len(protected.levels),
                    "protected_banks": len(protected.blocks),
                    "protected_exact_items": sum(len(b.keys) for b in protected.blocks),
                    "protected_retained_mse": float(np.mean(protected_loss)),
                    "protected_merges": protected.merges,
                    "discarded_exact_items": (index + 1) * d
                    - sum(len(b.keys) for b in protected.blocks),
                }
            )
    # Exactness for retained IDs, never a claim that discarded arbitrary values survive.
    raw["cascade/retained_ids"] = np.concatenate([b.ids for b in protected.blocks])
    return results


def hop_experiment(config, raw):
    rng = seeded(config, "evaluation/hops")
    trials = config["evaluation_trials"]
    result = []
    for name in ["permutation", "many_to_one"]:
        for eps in [1e-8, 0.01, 0.1]:
            for hops in [1, 2, 4, 8]:
                losses, norms, bounds = [], [], []
                for _ in range(trials):
                    d = 16
                    successors = (
                        rng.permutation(d)
                        if name == "permutation"
                        else np.zeros(d, dtype=int)
                    )
                    k, v = np.eye(d), np.eye(d)[successors]
                    m = EvidenceState.from_batch(k, v).c / (1 + eps)
                    q, true = k[rng.integers(d)].copy(), None
                    true = q.copy()
                    for _ in range(hops):
                        q, true = m @ q, v.T @ true
                    error = np.linalg.norm(q - true)
                    norm_m = np.linalg.norm(m, 2)
                    one_step = eps / (1 + eps)
                    bound = one_step * sum(norm_m**j for j in range(hops))
                    losses.append(error * error)
                    norms.append(norm_m)
                    bounds.append(error <= bound + 1e-12)
                scenario = f"hops/{name}/{eps}/{hops}"
                raw[scenario] = np.asarray(losses)
                result.append(
                    {
                        "graph": name,
                        "ridge": eps,
                        "hops": hops,
                        "operator_norm": float(np.mean(norms)),
                        "mse_norm": mean_interval(losses),
                        "bound_violations": trials - sum(bounds),
                    }
                )
    return result


def diagnostics(config):
    rng = seeded(config, "evaluation/diagnostics")
    n, d, dv = 129, 8, 3
    k, v = unit_rows(rng.normal(size=(n, d))), rng.normal(size=(n, dv))
    decay, beta = rng.uniform(0.9, 1, n), rng.uniform(0.1, 2, n)
    elements = [
        AffineSummary(lam, b * np.outer(key, key), b * np.outer(value, key))
        for key, value, b, lam in zip(k, v, beta, decay)
    ]
    scan = tree_prefix(elements)
    state = EvidenceState.zeros(d, dv)
    scan_error = 0.0
    for i, element in enumerate(scan):
        state.write(k[i], v[i], beta[i], decay[i])
        scan_error = max(
            scan_error,
            np.max(np.abs(state.s - element.s)),
            np.max(np.abs(state.c - element.c)),
        )
    precision_rows = []
    for separation in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        keys = np.array([[1.0, 0.0], [1.0, separation]])
        keys = unit_rows(keys)
        values = np.array([[1.0], [-1.0]])
        for eps in [1e-8, 1e-4, 0.01]:
            reference = EvidenceState.from_batch(keys, values).read(keys, eps)
            k32, v32 = keys.astype(np.float32), values.astype(np.float32)
            a32 = k32.T @ k32 + np.float32(eps) * np.eye(2, dtype=np.float32)
            try:
                float32 = ((v32.T @ k32) @ np.linalg.solve(a32, k32.T)).T
                fp_error = float(np.max(np.abs(float32 - reference)))
                failed = False
            except np.linalg.LinAlgError:
                fp_error, failed = None, True
            qr = interpolation(keys, values, keys)
            precision_rows.append(
                {
                    "key_separation": separation,
                    "ridge": eps,
                    "condition": float(np.linalg.cond(keys.T @ keys + eps * np.eye(2))),
                    "float64_recall_mse": float(np.mean((reference - values) ** 2)),
                    "float32_vs_float64_max_error": fp_error,
                    "float32_failed": failed,
                    "protected_qr_mse": float(np.mean((qr - values) ** 2)),
                }
            )
    # Same unordered observations, different ordered target: every state-only
    # read from the undiscounted evidence must return the same answer.
    order = {
        "gaussian_AB_BA_identical": True,
        "best_balanced_accuracy_bound": 0.5,
        "affine_order_head_AB": 3.0,
        "affine_order_head_BA": -1.0,
        "construction": "A:z->-z+1; B:z->z+2; initial z=0",
    }
    # Fixed-state bits and equal-state attention accounting (floats are not bits).
    state_words = d * (d + 1) // 2 + dv * d
    kv_budget = state_words // (d + dv)
    return {
        "scan_all_prefix_max_abs_error": float(scan_error),
        "conditioning": precision_rows,
        "order_counterexample_and_repair": order,
        "resource_accounting_example": {
            "d": d,
            "dv": dv,
            "ridge_state_words_symmetric": state_words,
            "equal_word_KV_capacity": kv_budget,
            "query_keys_need_address_bits": True,
        },
        "read_gradient_counterexample": {
            "query_unwritten_direction": True,
            "d_read_d_C_grows_as": "1/epsilon",
            "gradient_at_epsilon_1e_8": 1e8,
        },
    }


def code_digest():
    paths = sorted(Path("analysis").glob("*.py"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="analysis/configs/protocol.json")
    parser.add_argument("--output", default="analysis/results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if args.quick:
        for key in [
            "development_trials",
            "evaluation_trials",
            "risk_trials",
            "calibration_trials",
            "repeated_trials",
        ]:
            config[key] = 24 if key == "development_trials" else 64
        config["bootstrap_draws"] = 200
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    raw = {}
    summary = {
        "mode": "quick" if args.quick else "full",
        "protocol": config,
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": code_digest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    rows, comparisons, selections = recall(config, raw)
    summary.update({"recall_comparisons": comparisons, "selections": selections})
    for name, fn in [
        ("functional", functional_queries),
        ("risk", regression_risk),
        ("calibration", calibration),
        ("repeated", repeated_writes),
        ("cascade", cascade_experiment),
        ("hops", hop_experiment),
    ]:
        summary[name] = fn(config, raw)
        print(f"completed {name}", flush=True)
    summary["floor"] = stability_and_floor(config)
    summary["interpolation_bound"] = interpolation_bounds(config, seeded, raw)
    summary["drift"] = drifting_operator(config, seeded, raw)
    summary["diagnostics"] = diagnostics(config)
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["git_base_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    with (out / "recall.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "trials.npz", **raw)
    summary["trials_sha256"] = hashlib.sha256(
        (out / "trials.npz").read_bytes()
    ).hexdigest()
    (out / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n"
    )
    print(
        f"Saved {len(raw)} trial arrays; runtime {summary['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
