"""Further falsification: weighted interpolation bounds and drifting operators."""

import numpy as np

from analysis.memory import EvidenceState, unit_rows
from analysis.statistics import mean_interval, paired_bootstrap


def interpolation_bounds(config, seeded, raw):
    result = []
    for dim in [8, 16, 32]:
        for fraction in [0.5, 1.0]:
            count = int(dim * fraction)
            rng = seeded(config, f"evaluation/interpolation_bound/{dim}/{count}")
            ratios, conditions, errors = [], [], []
            violations = 0
            for _ in range(config["evaluation_trials"]):
                keys = unit_rows(rng.normal(size=(count, dim)))
                values = rng.normal(size=(count, config["value_dim"]))
                beta = np.exp(rng.uniform(-3, 3, count))
                ridge = 1e-4
                state = EvidenceState.from_batch(keys, values, beta)
                reads = state.read(keys, ridge)
                error = np.linalg.norm(reads - values, axis=1)
                eigmin = np.linalg.eigvalsh(keys @ keys.T)[0]
                bound = (
                    ridge
                    * np.sqrt(beta.max() / beta)
                    * np.linalg.norm(values, 2)
                    / (beta.min() * eigmin + ridge)
                )
                ratios.append(float(np.max(error / bound)))
                conditions.append(np.linalg.cond(keys @ keys.T))
                errors.append(float(np.mean(error**2)))
                violations += int(np.any(error > bound * (1 + 1e-5) + 1e-10))
            scenario = f"bound/d{dim}/K{count}"
            raw[scenario + "/ratio"] = np.array(ratios)
            raw[scenario + "/condition"] = np.array(conditions)
            raw[scenario + "/error"] = np.array(errors)
            result.append(
                {
                    "dim": dim,
                    "count": count,
                    "trials": len(ratios),
                    "bound_violations": violations,
                    "max_error_to_bound": max(ratios),
                    "median_key_gram_condition": float(np.median(conditions)),
                    "recall_mse_norm": mean_interval(errors),
                }
            )
    return result


def drifting_operator(config, seeded, raw):
    """All reads are causal. Each full trajectory is one independent cluster."""
    decays = [1.0, 0.99, 0.95, 0.9, 0.5]
    windows = [4, 16, 64, 256]
    result = []
    n, burnin = 512, 128
    dev_trials = config["development_trials"]
    eval_trials = max(64, config["evaluation_trials"] // 4)
    for process_variance in [0.0, 0.001, 0.05]:
        scenario = f"drift/Q{process_variance}"

        def generate(rng, trials):
            latent = np.cumsum(
                rng.normal(size=(trials, n)) * np.sqrt(process_variance), axis=1
            )
            observed = latent + rng.normal(size=(trials, n))
            losses = {}
            for decay in decays:
                numerator, mass = np.zeros(trials), 0.0
                error = []
                for i in range(n):
                    numerator, mass = (
                        decay * numerator + observed[:, i],
                        decay * mass + 1,
                    )
                    if i >= burnin:
                        error.append((numerator / (mass + 0.01) - latent[:, i]) ** 2)
                losses[f"ridge_lambda_{decay}"] = np.mean(error, axis=0)
            prefix = np.cumsum(observed, axis=1)
            for window in windows:
                error = []
                for i in range(burnin, n):
                    start = max(0, i + 1 - window)
                    total = prefix[:, i] - (prefix[:, start - 1] if start else 0)
                    error.append((total / (i + 1 - start) - latent[:, i]) ** 2)
                losses[f"attention_window_{window}"] = np.mean(error, axis=0)
            all_history = prefix / np.arange(1, n + 1)
            losses["attention_all_history_uniform"] = np.mean(
                (all_history[:, burnin:] - latent[:, burnin:]) ** 2, axis=1
            )
            return losses

        dev = generate(seeded(config, "development/" + scenario), dev_trials)
        best_ridge = min(
            [name for name in dev if name.startswith("ridge")],
            key=lambda name: dev[name].mean(),
        )
        best_attention = min(
            [name for name in dev if name.startswith("attention_window")],
            key=lambda name: dev[name].mean(),
        )
        evaluation = generate(seeded(config, "evaluation/" + scenario), eval_trials)
        for name, loss in evaluation.items():
            raw[scenario + "/" + name] = loss
        result.append(
            {
                "process_variance": process_variance,
                "trajectory_length": n,
                "burnin": burnin,
                "selected_ridge": best_ridge,
                "selected_attention": best_attention,
                "development_loss": {
                    name: float(loss.mean()) for name, loss in dev.items()
                },
                "evaluation_loss": {
                    name: mean_interval(loss) for name, loss in evaluation.items()
                },
                "attention_minus_ridge": paired_bootstrap(
                    evaluation[best_attention],
                    evaluation[best_ridge],
                    seeded(config, "resample/" + scenario),
                    config["bootstrap_draws"],
                    3,
                ),
            }
        )
    return result
