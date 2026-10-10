# 60M screen evidence audit

The authorized screen trains a Transformer, official Mamba-3 and Mamba 4 at
about 60M parameters on exactly 1B FineWeb-Edu training targets, with one
seed each, on all 16 chips of the v4-32 pod. It stops before 125M scaling.
`validation/verify_screen.py` re-audits every run from its raw logs and
checkpoints. Its record is `lm/results/screen-60m/audit.json`, summarized in
`lm/results/screen-60m/REPORT.md`.

| Requirement | Evidence | Result |
|---|---|---|
| Matched size | Actual parameter trees | Transformer 59,985,920; Mamba-3 59,968,896; Mamba 4 59,893,216. Total and non-embedding counts within 1%; all three share 25,731,584 embedding parameters. |
| Pinned data and leakage-free holdout | Corpus hashes, document ledger, `validate_data`, `docs/fineweb-data.md` | Passed. Dataset revision `66b78cd134ab7b0545a1a618a1cc029f3091eb20`, GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; exact document overlap with the held-out split is zero. |
| Exact budget, identical stream and schedule | Step logs and run manifests | Every run: 7,630 optimizer steps, exactly 1,000,000,000 targets. The final 1,131,600 targets replay training documents only. Training protocols are identical. |
| All 16 chips | Hardware records from every worker | Four hosts with four v4 chips each, 16 unique device IDs. |
| Finite, healthy training | Every loss and global gradient norm | All finite. Held-out NLL improves from 10.93 at initialization to 3.5352 (Transformer), 3.4758 (Mamba-3) and 3.4719 (Mamba 4). |
| Durable final state | Final payload hash on every worker | Equal on all four workers for all three runs (`*-checkpoints.json`). |
| Shared source identity | Peer manifests against Mamba 4's execution sources | Identical apart from Mamba 4 files and additive config fields. |
| Memory stays above its floor | 17 scheduled diagnostics of every memory layer | Minimum precision eigenvalue 0.7208, against a floor minimum of 0.7179; maximum condition 137.9; all states finite. |
| Paired held-out intervals | Per-sequence NLL on the held-out split and a fresh confirmatory holdout | Mamba 4 minus Mamba-3: −0.0039 (95% block bootstrap −0.0052 to −0.0025) on the held-out split, −0.0045 (−0.0059 to −0.0031) on the fresh holdout. Against the Transformer, about −0.063 on both. |
| Frozen recall diagnostic | Per-prompt records (`recall-paired.json`) | Correct out of 128: 18 (Transformer), 30 (Mamba-3), 69 (Mamba 4). |

Every interval describes evaluation sequences only. One training seed per
model cannot estimate seed variability, so robustness belongs to the later
three-seed study. Raw records keep the run identifiers under which they were
produced; the Transformer and Mamba-3 runs ran under the original 60M screen
protocol, which shares every setting compared here.
