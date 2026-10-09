# 60M screen evidence audit

This audit follows the currently authorized task: implement Transformer,
Mamba-3 and Mamba 4 with production kernels, then train each approximately
60M model on exactly 1B FineWeb-Edu targets with one seed. It stops at the
60M screen. The older phase files and completion audit describe earlier
authorization and do not override the user's subsequent training instruction.

Snapshot: 2026-10-08, approximately 18:25 UTC. Transformer has completed;
Mamba-3 is still training; Mamba 4 has not yet completed its production-shape
benchmark or training run. These are distinct statuses, and job launch is not
completion. The table records authoritative local artifacts and independent
checks, rather than treating planned work or passing operator tests as trained
results.

| Requirement | Evidence inspected | Result at this snapshot |
|---|---|---|
| Read prompt and phase-03/04 contracts; stop before 125M | `prompt.md`, phases 03/04, `lm/configs/screen-protocol.json` | Scope preserved: three 60M runs, 1B targets each, seed 42; no scaling authorization inferred from a diagnostic win. |
| Full Transformer, official Mamba-3 and corrected Mamba 4 | `lm/models/`, `lm/kernels/`, parameter ledger and implementation documentation | Implemented; the Mamba-3 run uses official SISO features, while MIMO primitive conformance is separately tested. Learned Gaussian, protected QR, order and routing states have explicit separate contracts. |
| Approximately 60M and matched embeddings/nonembedding capacity | Configurations and actual parameter trees | Transformer 59,985,920; Mamba-3 59,968,896; Mamba 4 59,989,408. Every model shares 25,731,584 embedding parameters. Final actual checkpoint counts are independently checked per completed run. |
| Pin data/tokenizer acquisition and exclude held-out documents | Complete corpus hashes, document ledger/order hashes, `validate_data`, `docs/fineweb-data.md` | Passed. Dataset revision `66b78cd134ab7b0545a1a618a1cc029f3091eb20`; GPT2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; exact document-hash overlap zero. Near-duplicate exclusion is not claimed. |
| Exactly 1B training targets and common stream/masks | Train/eval memmaps, final-batch tests, completed Transformer step log | Passed for data and Transformer: 7,630 steps, final 51,712 valid targets. Evaluation uses 2,097,152 targets. The final 1,131,600 training targets explicitly replay training documents only. Mamba-3/Mamba 4 final budgets remain unproved until their actual completion. |
| Shared loss, optimizer, schedule, precision, seed and data order | Frozen configurations, source hashes, weighted-gradient/restart tests | Shared protocol implemented. Global gradients divide by the globally summed valid-target count. Four-CPU tests cover an empty device and reproduce interrupted/resumed optimizer state. Final cross-model equality remains a required audit. |
| All 16 v4 chips across four distinct hosts | Hardware records and Transformer four-worker final results | Passed for observed topology and completed Transformer: four hosts, four local devices each, 16 unique v4 device IDs/coordinates. JAX process 0 is physical worker 2, so canonical evidence is collected from every physical host. |
| Fused production forward/backward conformance | Four source-consistent `lm/results/kernel-conformance/host-*.json` files | Passed: 11 declared cases on every local chip, including bf16 attention, Mamba-3 tail/final-state/MIMO cases, cyclic/fixed Gaussian factors, repeated/near-dependent protected keys, cached/full block-boundary agreement, and repeated-EOS full-parameter gradients. This proves the tested configurations, not a training win. |
| Production-shape training/prefill/cached-decode speed, compiler cost and memory | Existing benchmark artifacts | Incomplete. Existing peer benchmarks have no execution provenance; Transformer lacks cached decode; Mamba-3 decode lacks a recorded mature prefix; no full Mamba 4 benchmark exists yet. Final acceptance needs all three architectures on four hosts, B128/L1024, cached context at least 1024, numeric FLOPs/bytes, compiler memory and observed device peaks. |
| Preserve immutable executable sources through live runs | `source_provenance()` and completed Transformer manifest | Passed at snapshot: 39 frozen execution files match the Transformer manifest and every current conformance artifact. Validation code is versioned separately and does not change these files. |
| Durable resumable checkpoints and actual trained parameter state | Final Transformer msgpack, metadata, checksum, parameter names/shapes/counts | Passed for Transformer. Its actual state contains step 7,630 and finite parameters with the declared ledger. Actual final-payload hashes independently read on all four physical workers agree; sidecar is `lm/results/screen-60m-v1/transformer-checkpoints.json`. |
| Healthy finite training and improved held-out loss | Every Transformer training entry and actual final evaluation | Passed: all 7,630 losses/global gradient norms finite; initial NLL 10.9291391373 improves to 3.5352357328, perplexity 34.3031001268. Mamba-3's live log is healthy but cannot substitute for its final evaluation. |
| Frozen trained recall with prompt-level uncertainty | Actual Transformer `recall.json`, saved prompt arrays/texts and recomputation | Passed for Transformer: all 128 prompts and score aggregates reproduce exactly; accuracy 18/128 = 0.140625. The same frozen prompt hash is required for both remaining models. This is a diagnostic, not a claim of robust superiority. |
| Observe learned Mamba 4 factors, gates and protected geometry | Instrumentation in `lm/train.py` and final-auditor checks | Implementation present; trained evidence pending. Require initial, every scheduled evaluation, and final probes for every layer, positive precision eigenvalues, finite condition numbers, positive priors/epsilon/write precision, valid decay gates, finite protected states and positive retained QR diagonals. Probes use one fixed held-out batch, not every training token. |
| Supervised pod lifecycle and failed-run evidence | `scripts/pod.py`, verified process-group cleanup helper, completed controller records | Process-group handles and checked cleanup are implemented; zombie handling is repaired. Transformer controller records exit codes `[0,0,0,0]`. Failed prior probe/benchmark records remain separate. Cleanup failure must remain a failure; an SSH disconnect alone is insufficient terminal evidence. |
| Actual completion of all three requested runs | Run results, final checkpoints, metrics and four-worker agreement | Transformer independently audited complete. Mamba-3 live and Mamba 4 pending. The full objective is not complete at this snapshot. |
| Preserve peer wins and decide the declared screen gate | Final same-protocol NLL comparison | Pending all three results. A Mamba 4 loss must remain in the report; one training seed cannot establish robustness. No 125M run follows automatically. |

The independent Transformer audit used the existing frozen `audit_run` helper
with actual corpus-file verification and a newly reconstructed frozen recall
dataset. It checked all optimizer steps, the real final msgpack parameter tree,
all worker results/topologies, protocol/source fingerprints, and raw recall
recomputation. Its final checkpoint SHA256 is
`0607687b7a8fa6dc1a7f52b142964c81e8092a3c46390316482100ac6a35081f`;
its run fingerprint is
`04a9186d78f083f97cde4a5327d0d5701515846eb4ce96c888d2dcd3e190fa5b`.

The original frozen final verifier assumes every conformance case has
`forward`/`gradients` error dictionaries. The last three actual protected cases
use specialized schemas with per-chip QR diagnostics, cached/full error data
and repeated-EOS gradient flags. Editing that verifier during training would
change the recorded execution-source fingerprint. Therefore
`validation/verify_screen.py` supplies separately versioned final validation,
retaining the original data/run/checkpoint/hardware checks and adding numeric
schema-aware engineering checks and learned Mamba 4 diagnostics. It records its
own revision/code hashes separately from the 39 frozen execution-source hashes.
The specialized cases are checked using numeric margins/errors and their
computed finite/gradient flags; `passed: true` alone is insufficient.

```sh
JAX_PLATFORMS=cpu uv run pytest -q validation/tests/test_screen_audit.py
JAX_PLATFORMS=cpu uv run python -m validation.verify_screen \
  --require-crosshost-checkpoints
```

The final auditor's revision `screen-final-audit-v2` uses the actual frozen
`layer_i/memory/` diagnostic namespace. A four-CPU subprocess integration test
creates probes through the real `Mamba4LM`, `make_diagnostic_step` and
`unreplicate`, then passes those actual scalar outputs through both the training
log and benchmark validators. Benchmark validation also checks the recorded
diagnostic execution status/timings and positive, finite learned factor/QR
evidence. This corrects an earlier fixture that invented a `mixer` namespace.

The new audit's 24 tests pass, including adversarial missing-case, false QR,
NaN/error-threshold, repeated-EOS-gradient, stale-source, immature-context,
compiler-cost/memory, missing-model/host, learned-factor and physical-checksum
failures. Actual existing conformance reports also pass its case-aware checks.
The final command intentionally cannot succeed until fresh source-consistent
benchmarks, all three complete runs, final Mamba 4 diagnostics and independently
verified four-worker checkpoint sidecars exist. Completion and a predictive
screen win are recorded separately; neither is inferred from an active job.

## Snapshot: 2026-10-09, 05:15 UTC

Mamba-3 completed and its independent audit passed (held-out NLL
3.4757549912; `lm/results/screen-60m-v1/mamba3-audit.json`). The frozen
Mamba 4 benchmark passed on all four hosts and its training run started at
19:12 UTC. It did **not** complete. A Cloud TPU defragmentation preemption
sent SIGTERM at 02:43:51 UTC after 1,026 logged steps; HW/SW maintenance
reboots followed at 03:33 and 04:23 UTC. No controller or worker survived.
The step-1,000 checkpoint (131,072,000 targets) is byte-identical on all four
hosts (SHA256 `10c958e9…65d6f8`) and remains resumable with the v1 sources.

At matched optimizer steps the v1 model trailed both completed peers:

| Step | Transformer NLL | Mamba-3 NLL | Mamba 4 v1 NLL |
|---:|---:|---:|---:|
| 500 | 4.9144 | 4.7641 | 5.2963 |
| 1,000 | 4.3596 | 4.2364 | 4.6073 |

Its steady step took 25.53 s, versus 0.86 s for Mamba-3 and 0.094 s for the
Transformer, so completion needed about 47 more exclusive pod hours. The run
was therefore not resumed; its final NLL is unobserved, and no screen win is
claimed. Every peer and partial result is retained in
`lm/results/screen-60m-v1/learning-curves.json` and
`mamba4-interrupted.json`. The diagnosis and versioned replacement are
recorded as iteration H in [the iteration log](iterations.md).
