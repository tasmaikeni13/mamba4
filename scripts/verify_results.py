"""Audit actual trial data and contracts; never mark a failed dominance gate green."""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path("analysis/results")
    summary = json.loads((out / "summary.json").read_text())
    config = json.loads(Path("analysis/configs/protocol.json").read_text())
    assert summary["mode"] == "full", "Smoke runs are not full scientific evidence"
    assert summary["protocol"] == config
    assert summary["protocol_sha256"] == digest("analysis/configs/protocol.json")
    for path, sha in summary["source_sha256"].items():
        assert digest(path) == sha, f"Stale numerical evidence: {path} changed"
    assert summary["trials_sha256"] == digest(out / "trials.npz")
    raw = np.load(out / "trials.npz", allow_pickle=False)
    for name in raw.files:
        assert np.all(np.isfinite(raw[name])), f"Nonfinite raw result: {name}"
    rows = list(csv.DictReader((out / "recall.csv").open()))
    scenarios = {
        f"recall/{kind}/K{count}"
        for kind in config["geometries"]
        for count in config["loads"]
    }
    assert {row["scenario"] for row in rows} == scenarios
    for scenario in scenarios:
        models = {row["model"] for row in rows if row["scenario"] == scenario}
        assert {
            "ridge",
            "softmax",
            "hebbian",
            "diagonal",
            "delta",
            "mamba3_siso_core",
            "mamba3_mimo_tied_core",
        } <= models
    for row in rows:
        data = raw[row["scenario"] + "/" + row["model"]]
        assert len(data) == config["evaluation_trials"] == int(row["n"])
        assert np.isclose(data.mean(), float(row["mean"]), atol=1e-30, rtol=1e-12)
        assert np.isclose(
            data.std(ddof=1) / np.sqrt(len(data)),
            float(row["se"]),
            atol=1e-30,
            rtol=1e-12,
        )
    family = summary["recall_comparisons"]
    assert all(c["family_size"] == len(family) for c in family)
    for comparison in family:
        difference = (
            raw[comparison["scenario"] + "/" + comparison["baseline"]]
            - raw[comparison["scenario"] + "/ridge"]
        )
        assert np.isclose(difference.mean(), comparison["mean"], atol=1e-30, rtol=1e-12)
    for name, result in summary["risk"].items():
        data = raw["risk/" + name]
        assert len(data) == config["risk_trials"]
        assert np.isclose(data.mean(), result["empirical"]["mean"])
        # Five standard errors is a declared numerical/statistical consistency
        # check, not a theorem that a 95% interval must always include truth.
        assert result["standard_errors_from_theory"] < 5
    for name in ["in_model", "misspecified", "duplicate_conflict", "heavy_tailed"]:
        data = raw["calibration/" + name + "/z"]
        coverage = summary["calibration"][name]["latent_95_coverage"]
        assert len(data) == config["calibration_trials"]
        assert np.sum(np.abs(data) <= 1.959963984540054) == coverage["successes"]
    coverage = summary["calibration"]["in_model"]["latent_95_coverage"]
    se = np.sqrt(0.95 * 0.05 / coverage["n"])
    assert abs(coverage["rate"] - 0.95) < 5 * se
    assert all(row["bound_violations"] == 0 for row in summary["interpolation_bound"])
    assert all(row["bound_violations"] == 0 for row in summary["hops"])
    for row in summary["floor"]:
        assert row["new_min_over_eps"] >= 1 - 1e-12
        assert np.isclose(row["new_max_over_eps"], row["new_max_theory"])
        assert row["max_factor_relative_error"] < 1e-10
        assert row["max_read_error"] < 1e-8
    for row in summary["cascade"]:
        assert row["original_popcount"] == row["original_occupied_blocks"]
        assert row["protected_retained_mse"] < 1e-16
        assert row["protected_exact_items"] == 8 * row["protected_banks"]
        assert row["protected_merges"] < row["blocks_ingested"]
    assert summary["diagnostics"]["scan_all_prefix_max_abs_error"] < 1e-10
    formal = json.loads((out / "formal-audit.json").read_text())
    assert formal["theorem_count"] >= 50 and formal["new_axioms"] == []
    for path, sha in formal["source_sha256"].items():
        assert digest(path) == sha, f"Stale proof audit: {path} changed"
    assert formal["log_sha256"] == digest(out / "formal-axioms.txt")
    original = Path("docs/mamba4-original.md").read_text()
    ledger = Path("docs/claim-ledger.md").read_text()
    numbered = re.findall(
        r"^### (?:Theorem|Lemma|Definition|Corollary|Proposition) (\d+\.\d+)",
        original,
        re.M,
    )
    for claim in numbered:
        assert f"| {claim} " in ledger, f"Missing original claim {claim}"
    artifacts = [
        "mamba4.md",
        "docs/mathematical-analysis.md",
        "docs/claim-ledger.md",
        "docs/iterations.md",
        "CLAUDE.md",
        "phases/README.md",
        "analysis/results/REPORT.md",
        "analysis/results/figures/recall.png",
        "analysis/results/figures/calibration.png",
        "analysis/results/figures/cascade.png",
        "analysis/results/figures/conditioning.png",
    ]
    for path in artifacts:
        assert Path(path).is_file(), f"Missing deliverable: {path}"
    ridge_advantage = all(
        c["low"] > -1e-10 for c in family if c["baseline"] == "softmax"
    )
    gates = {
        "phase_01_investigation_complete": True,
        "phase_02_investigation_complete": True,
        "original_named_claims_audited": len(numbered),
        "universal_peer_dominance": ridge_advantage,
        "full_language_model_comparison": False,
        "measured_TPU_speedup": False,
        "phase_04_screen_win": False,
        "later_phases_executed": False,
        "reason": "Corrected contracts and full operator investigation verified. Universal softmax dominance is falsified; trained screen and hardware claims require later phases.",
    }
    assert not gates["universal_peer_dominance"], (
        "Known overcapacity falsification must remain visible"
    )
    (out / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    status = {
        "authorized_scope": [1, 2],
        "phases": [
            {
                "id": 1,
                "status": "completed",
                "meaning": "Mathematical/formal investigation",
                "evidence": "analysis/results/formal-audit.json",
            },
            {
                "id": 2,
                "status": "completed",
                "meaning": "Statistical/Monte Carlo investigation",
                "evidence": "analysis/results/summary.json",
                "universal_dominance": "falsified",
            },
            *[{"id": i, "status": "planned"} for i in range(3, 8)],
        ],
        "promotion": "No trained performance or TPU speed claim. Review corrected methods before authorizing phase 03.",
    }
    Path("phases/status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(
        f"Verified {len(rows)} model/scenario rows, {len(raw.files)} raw arrays, {formal['theorem_count']} Lean theorems and all {len(numbered)} named original claims. Dominance gate remains false."
    )


if __name__ == "__main__":
    main()
