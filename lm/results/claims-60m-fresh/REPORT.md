# Claims-60m results

Pre-registered tests (`validation/claims/PROTOCOL.md`) of the paper's
retrieval, long-context, calibration and decode claims on the trained
60M checkpoints. One training seed; prompts or documents are the
sampling units. Verdicts compare Mamba 4 with each peer under the frozen
rules; "read zeroed" is an evaluation-time ablation, not a model.

## Verdicts

| Claim | vs Transformer | vs Mamba-3 |
|---|---|---|
| Exact retrieval (within 1,024) | refuted | supported |
| Long context | refuted | refuted |
| Calibrated confidence | refuted | refuted |
| Context-independent decode | supported | — |

## Exact copy: top-1 accuracy on the second copy

| L | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed | Mamba 4 without key shift | Mamba 4 without key shift, read zeroed | vs T | vs M3 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 16 | 42.3% | 5.7% | 11.8% | 2.3% | 11.9% | 4.3% | trails | exceeds |
| 64 | 35.8% | 2.8% | 7.5% | 0.5% | 9.7% | 1.1% | trails | exceeds |
| 256 | 13.5% | 0.1% | 0.2% | 0.0% | 0.3% | 0.0% | trails | exceeds |
| 512 | 2.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | trails | exceeds |
| 1024 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 2048 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 4096 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |

## Associative recall: top-1 accuracy at 16 queries

| K | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed | Mamba 4 without key shift | Mamba 4 without key shift, read zeroed | vs T | vs M3 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 16 | 3.5% | 0.1% | 0.6% | 0.0% | 1.8% | 0.5% | trails | matches |
| 32 | 1.1% | 0.1% | 0.2% | 0.0% | 0.3% | 0.1% | trails | matches |
| 64 | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 128 | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | matches | matches |
| 240 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 496 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 1008 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 2032 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |

## Passkey retrieval: exact match, pooled over five depths

| Length | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed | Mamba 4 without key shift | Mamba 4 without key shift, read zeroed |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 68.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1,024 | 41.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2,048 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 4,096 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 8,192 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 16,384 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Long-document NLL by position (512 documents)

| Positions | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed | Mamba 4 without key shift | Mamba 4 without key shift, read zeroed |
|---|---:|---:|---:|---:|---:|---:|
| 0-128 | 3.694 | 3.608 | 3.606 | 3.644 | 3.598 | 3.636 |
| 128-512 | 3.472 | 3.414 | 3.410 | 3.503 | 3.411 | 3.505 |
| 512-1024 | 3.503 | 3.456 | 3.451 | 3.581 | 3.453 | 3.591 |
| 1024-2048 | 4.904 | 3.507 | 3.504 | 3.649 | 3.504 | 3.660 |
| 2048-4096 | 5.800 | 3.542 | 3.537 | 3.691 | 3.538 | 3.705 |
| 4096-8192 | 6.091 | 3.535 | 3.529 | 3.689 | 3.530 | 3.702 |
| 8192-16384 | 6.491 | 3.515 | 3.508 | 3.671 | 3.509 | 3.683 |

NLL at 8,192–16,384 minus NLL at 512–1,024 (paired over documents): Transformer +2.988 [+2.935, +3.040]; Mamba-3 +0.059 [+0.027, +0.091]; Mamba 4 +0.057 [+0.025, +0.090]; Mamba 4, read zeroed +0.090 [+0.058, +0.123]; Mamba 4 without key shift +0.056 [+0.025, +0.089]; Mamba 4 without key shift, read zeroed +0.092 [+0.061, +0.125].

## Calibration at answer slots

| Family | Transformer ECE / AUROC | Mamba-3 ECE / AUROC | Mamba 4 ECE / AUROC | Mamba 4, read zeroed ECE / AUROC | Mamba 4 without key shift ECE / AUROC | Mamba 4 without key shift, read zeroed ECE / AUROC |
|---|---:|---:|---:|---:|---:|---:|
| copy | 0.0405 / 0.761 | 0.0547 / 0.692 | 0.0543 / 0.710 | 0.0364 / 0.693 | 0.0348 / 0.804 | 0.0124 / 0.764 |
| mqar | 0.0124 / 0.550 | 0.0095 / 0.409 | 0.0081 / 0.737 | 0.0144 / 0.862 | 0.0040 / 0.823 | 0.0067 / 0.765 |
| passkey | 0.0880 / 0.646 | 0.1737 / 0.716 | 0.1951 / 0.626 | 0.1674 / 0.705 | 0.2136 / 0.629 | 0.2074 / 0.587 |

Mamba 4 memory-variance AUROC (lower variance predicting a correct
answer), memory layers in order: copy 0.04, 0.14, 0.03, 0.09; mqar 0.16, 0.34, 0.33, 0.41; passkey 0.33, 0.41, 0.69, 0.39.

## One-token decode (median, one sequence per chip)

| Context | Transformer ms (cache MiB) | Mamba-3 ms (cache MiB) | Mamba 4 ms (cache MiB) | Mamba 4 without key shift ms (cache MiB) |
|---:|---:|---:|---:|---:|
| 1,024 | 4.61 (22.9) | 2.77 (7.6) | 5.36 (7.3) | 5.37 (7.3) |
| 4,096 | 10.92 (88.9) | 2.76 (7.6) | 5.38 (7.3) | 5.33 (7.3) |
| 16,384 | 28.44 (352.9) | 2.74 (7.6) | 5.31 (7.3) | 5.31 (7.3) |
| 65,536 | 904.13 (1408.9) | 2.79 (7.6) | 5.28 (7.3) | 5.31 (7.3) |

Full-sequence forward time per 16,384-token document (one per chip): Transformer 0.355 s; Mamba-3 0.197 s; Mamba 4 0.758 s; Mamba 4, read zeroed 0.744 s; Mamba 4 without key shift 0.759 s; Mamba 4 without key shift, read zeroed 0.744 s.
