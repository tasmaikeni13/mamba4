"""Statistics and log handling of the screen audit, without TPUs."""

import numpy as np
import pytest

from validation import run_screen
from validation.verify_screen import block_bootstrap, paired_statistics


def entry(values):
    return {
        "sequence_nll_sum": (np.asarray(values) * 1024).tolist(),
        "sequence_targets": [1024] * len(values),
    }


def test_paired_statistics_detects_a_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.normal(3.5, 0.3, size=256)
    models = {
        "transformer": {
            s: entry(base + 0.05) for s in ("screen_heldout", "fresh_holdout")
        },
        "mamba3": {s: entry(base + 0.02) for s in ("screen_heldout", "fresh_holdout")},
        "mamba4": {s: entry(base) for s in ("screen_heldout", "fresh_holdout")},
    }
    audits = {
        name: {"heldout": {"nll": float(np.mean(base) + shift)}}
        for name, shift in (("transformer", 0.05), ("mamba3", 0.02), ("mamba4", 0.0))
    }
    result = paired_statistics({"models": models}, audits)
    row = result["screen_heldout"]["paired"]["mamba4_minus_mamba3"]
    assert row["mean"] == pytest.approx(-0.02)
    assert row["block_bootstrap_95"][1] < 0
    assert row["sequences_mamba4_lower"] == 256


def test_sequence_nll_must_reproduce_the_audit():
    values = np.full(64, 3.0)
    models = {
        name: {s: entry(values) for s in ("screen_heldout", "fresh_holdout")}
        for name in ("transformer", "mamba3", "mamba4")
    }
    audits = {name: {"heldout": {"nll": 3.0}} for name in models}
    audits["mamba3"]["heldout"]["nll"] = 3.1
    with pytest.raises(ValueError, match="does not reproduce"):
        paired_statistics({"models": models}, audits)


def test_block_bootstrap_interval_covers_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    low, high = block_bootstrap(rng.normal(0, 1, size=512), np.random.default_rng(2))
    assert low < 0 < high


def test_log_absorption_handles_local_and_moved_process_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(run_screen, "LOG", tmp_path / "metrics.jsonl")
    monkeypatch.setattr(run_screen, "MERGED", tmp_path / "metrics.merged.jsonl")
    monkeypatch.setattr(run_screen, "HOSTS", ["local"])
    run_screen._write_log(['{"step": 1}', '{"step": 2}'])
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n')
    assert run_screen.absorb_logs() == 1
    # A process 0 on another host starts a fresh file after a reboot.
    (tmp_path / "metrics.jsonl").write_text('{"step": 3}\n{"step": 4}\n')
    assert run_screen.absorb_logs() == 2
    lines = (tmp_path / "metrics.merged.jsonl").read_text().split("\n")
    assert [line for line in lines if line] == [
        '{"step": 1}',
        '{"step": 2}',
        '{"step": 3}',
        '{"step": 3}',
        '{"step": 4}',
    ]
