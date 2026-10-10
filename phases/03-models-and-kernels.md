# Phase 03: full models and fused kernels

Implemented in `lm/`; hardware conformance and final performance gates are
in progress. Dependencies: phases 01–02. The production Mamba 4 composition
and its explicit prefill/decode/routing costs are recorded in
`docs/mamba4-implementation.md`; no theoretical cost is inferred from fusion.

Implement parameter-matched 60M Mamba 4, official Mamba-3 and Transformer in
JAX for the user's v4-32 pod (16 v4 chips). Discover and record actual logical
device topology/count, memory, compiler and precision before sharding.
Use identical tokenizer, batch, sequence length, loss, optimizer budget and
data ordering. Mamba-3 must include learned projections, normalization,
complex phases, trapezoidal terms, MIMO where evaluated and output gating;
the phase-02 tied operator reference is not a replacement.

Implement fused scan/write/read paths and the peers' appropriate fused kernels;
favor correct conformance and useful fusion over excessive kernel tuning.
Compare forward/backward against the CPU reference, verify causality and loss
gradients, and measure prefill/decode/roofline/memory. Test fixed-floor fallback,
epsilon learning, constant/variable gate semantics and protected routing.
Count QR reselection, buffers, metadata and all-bank reads. Reject unsupported
quadratic decode claims for a different floor model. All production runs use
the verified fused paths; retain a numerical fallback for diagnosis.

Gate: full-model parameter ledger, gradient conformance, no data leakage,
all-device execution, stable factors and reproducible throughput/memory results.
If a repair changes phase-01 guarantees, update the paper/proofs and rerun the
affected phase-02 evidence before screening.

Evidence: `lm/results/parameter-ledger.json`, all four
`lm/results/hardware/host-*.json` collective reports, CPU numerical and
gradient tests, kernel timings (`lm/results/kernels/`) and source-matched
benchmarks (`lm/results/benchmarks/`).
Current authorization covers phases 03–04 only; no 125M scaling follows
automatically.

The Mamba 4 kernel is the selective fixed-floor memory
(`selective_gaussian_memory`, `spd_solve` with loop, blocked and unrolled
schedules, `order_memory_chunked`). The hybrid composition reuses the official
Mamba-3 block and its pinned step recurrence.

CPU conformance covers sequential dense references, all-input gradients,
schedules, decode, continuation, and rotary, wide-head and key-aligned
variants. TPU evidence covers per-block and solver timings
(`lm/results/kernels/`) and the full run itself.

Two engineering failures are recorded in the iteration log: the compile time
of a Python-unrolled factor, and launch-identity mismatches in isolated
multi-host programs.
