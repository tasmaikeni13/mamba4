"""Independent all-device masking and interrupted-run verification on four CPUs.

The subprocess selects the CPU backend before JAX import, so these tests do
not initialize or interfere with the physical TPU pod used by training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def training_evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp("training-audit")
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((directory / "evidence.json").read_text())


def test_global_gradient_weights_real_targets_including_empty_device(training_evidence):
    assert training_evidence["devices"] == 4
    assert training_evidence["uneven_device_counts"] == [6, 4, 1, 0]
    assert training_evidence["weighted_gradient_matches_unsharded"]


def test_checkpoint_restart_restores_parameters_optimizer_and_budget(training_evidence):
    assert training_evidence["restart_state_exact"]
    assert training_evidence["training_targets"] == 35
    assert training_evidence["optimizer_steps"] == 3
    assert training_evidence["final_step_targets"] == 3
    assert training_evidence["retained_checkpoints"] == 2


def test_evaluation_and_resume_rejections(training_evidence):
    assert training_evidence["heldout_targets"] == 7
    assert training_evidence["evaluation_matches_unsharded"]
    assert training_evidence["rejected_changed_protocol"]
    assert training_evidence["rejected_checkpoint_corruption"]
    assert training_evidence["rejected_target_metadata_corruption"]


def _worker(directory: Path) -> None:
    from dataclasses import replace
    import hashlib
    from types import SimpleNamespace

    from flax import linen as nn
    from flax import serialization
    from flax.training.train_state import TrainState
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numpy.testing import assert_allclose, assert_array_equal
    import optax

    from lm.config import ModelConfig, TrainConfig
    from lm.data import TokenCorpus
    import lm.train as training

    assert jax.local_device_count() == 4

    class TinyLM(nn.Module):
        @nn.compact
        def __call__(self, inputs, train=False):
            hidden = nn.Embed(11, 5)(inputs)
            return nn.Dense(11)(hidden)

    model = TinyLM()
    inputs = jnp.arange(24).reshape(8, 3) % 11
    targets = (inputs + 3) % 11
    mask = jnp.asarray(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=jnp.float32,
    )
    batch = {"input_ids": inputs, "targets": targets, "loss_mask": mask}
    params = model.init(jax.random.PRNGKey(93), inputs)["params"]

    def reference_loss(parameters):
        logits = model.apply({"params": parameters}, inputs)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.sum(losses * mask) / jnp.sum(mask)

    reference_loss_value, reference_grads = jax.value_and_grad(reference_loss)(params)
    tx = optax.sgd(0.07)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    reference_state = state.apply_gradients(grads=reference_grads)
    replicated = jax.device_put_replicated(state, jax.local_devices())
    train_step, eval_step = training.make_steps(model)
    updated, metrics = train_step(replicated, training.shard_batch(batch, 8))
    actual = training.unreplicate(updated)
    values = training.unreplicate(metrics)
    for expected, observed in zip(
        jax.tree_util.tree_leaves(reference_state.params),
        jax.tree_util.tree_leaves(actual.params),
        strict=True,
    ):
        assert_allclose(observed, expected, rtol=2e-6, atol=2e-7)
    assert_allclose(values["loss"], reference_loss_value, rtol=1e-6)
    assert_allclose(values["grad_norm"], optax.global_norm(reference_grads), rtol=2e-6)
    assert int(values["targets"]) == 11
    eval_values = training.unreplicate(
        eval_step(
            jax.device_put_replicated(params, jax.local_devices()),
            training.shard_batch(batch, 8),
        )
    )
    assert int(eval_values["targets"]) == 11
    assert_allclose(
        eval_values["nll_sum"] / eval_values["targets"],
        reference_loss_value,
        rtol=1e-6,
    )

    # Exercise the actual run entry point, including its fingerprint validation,
    # optimizer schedule, exact masked final step, evaluation and checkpoint I/O.
    data_directory = directory / "data"
    data_directory.mkdir()
    split_meta = {}
    for name, size in (("train", 36), ("eval", 10)):
        ids = (np.arange(size) + (0 if name == "train" else 3)) % 17
        payload = ids.astype("<u2").tobytes()
        (data_directory / f"{name}.bin").write_bytes(payload)
        split_meta[name] = {
            "file": f"{name}.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "token_count": size,
            "target_count": size - 1,
        }
    (data_directory / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mamba4.tokens.v1",
                "dtype": "<u2",
                "tokenizer": {"vocab_size": 17, "eos_id": 16},
                "splits": split_meta,
            }
        )
    )
    model_config = ModelConfig(
        architecture="transformer",
        vocab_size=17,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=12,
        chunk_size=2,
        dtype="float32",
        max_seq_len=2,
        remat=False,
    )
    train_config = TrainConfig(
        token_budget=35,
        global_batch=8,
        sequence_length=2,
        eval_tokens=7,
        eval_every=2,
        checkpoint_every=1,
        log_every=1,
        warmup_steps=1,
    )
    protocol = {"model": model_config.to_dict(), "training": train_config.to_dict()}
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(protocol))

    def arguments(output):
        return SimpleNamespace(
            config=str(config_path),
            data=str(data_directory),
            output=str(output),
            single_host=True,
            allow_small=True,
        )

    full_output = directory / "full"
    resumed_output = directory / "resumed"
    training.run(arguments(full_output))
    original_save = training.save_checkpoint

    def interrupt_after_checkpoint(output, checkpoint_state, metadata):
        original_save(output, checkpoint_state, metadata)
        if int(training.unreplicate(checkpoint_state).step) == 1:
            raise RuntimeError("simulated interruption after committed checkpoint")

    training.save_checkpoint = interrupt_after_checkpoint
    try:
        try:
            training.run(arguments(resumed_output))
        except RuntimeError as error:
            assert "simulated interruption" in str(error)
        else:
            raise AssertionError("The simulated interruption did not happen")
    finally:
        training.save_checkpoint = original_save
    training.run(arguments(resumed_output))

    def checkpoint_path(output):
        pointer = json.loads((output / "latest.json").read_text())["checkpoint"]
        return output / pointer

    full_checkpoint = checkpoint_path(full_output)
    resumed_checkpoint = checkpoint_path(resumed_output)
    full_state = serialization.msgpack_restore(
        (full_checkpoint / "state.msgpack").read_bytes()
    )
    resumed_state = serialization.msgpack_restore(
        (resumed_checkpoint / "state.msgpack").read_bytes()
    )
    for expected, observed in zip(
        jax.tree_util.tree_leaves(full_state),
        jax.tree_util.tree_leaves(resumed_state),
        strict=True,
    ):
        assert_array_equal(observed, expected)
    full_result = json.loads((full_output / "result.json").read_text())
    resumed_result = json.loads((resumed_output / "result.json").read_text())
    assert resumed_result["heldout"] == full_result["heldout"]
    assert resumed_result["fingerprint"] == full_result["fingerprint"]
    assert full_result["training_targets"] == 35
    assert full_result["optimizer_steps"] == 3
    assert full_result["heldout"]["targets"] == 7

    def expect_rejected(expected_text):
        try:
            training.run(arguments(resumed_output))
        except RuntimeError as error:
            assert expected_text in str(error), str(error)
        else:
            raise AssertionError("Invalid resume was accepted")

    changed = {
        **protocol,
        "training": replace(train_config, learning_rate=0.01).to_dict(),
    }
    config_path.write_text(json.dumps(changed))
    expect_rejected("Resume refused")
    config_path.write_text(json.dumps(protocol))
    payload_path = resumed_checkpoint / "state.msgpack"
    original_payload = payload_path.read_bytes()
    payload_path.write_bytes(original_payload + b"corrupted")
    expect_rejected("checksum")
    payload_path.write_bytes(original_payload)
    metadata_path = resumed_checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(
        json.dumps({**metadata, "training_targets": metadata["training_targets"] - 1})
    )
    expect_rejected("accounting")
    metadata_path.write_text(json.dumps(metadata))

    corpus = TokenCorpus(data_directory, verify_hashes=True)
    final_batch = corpus.batch(2, 8, 2)
    evidence = {
        "devices": jax.local_device_count(),
        "uneven_device_counts": [6, 4, 1, 0],
        "weighted_gradient_matches_unsharded": True,
        "evaluation_matches_unsharded": True,
        "restart_state_exact": True,
        "training_targets": resumed_result["training_targets"],
        "optimizer_steps": resumed_result["optimizer_steps"],
        "final_step_targets": int(final_batch["loss_mask"].sum()),
        "heldout_targets": resumed_result["heldout"]["targets"],
        "retained_checkpoints": len(
            list((resumed_output / "checkpoints").glob("step-*"))
        ),
        "rejected_changed_protocol": True,
        "rejected_checkpoint_corruption": True,
        "rejected_target_metadata_corruption": True,
    }
    (directory / "evidence.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        _worker(Path(sys.argv[2]))
    else:
        raise SystemExit("Run through pytest, or specify --worker TEMP_DIRECTORY")
