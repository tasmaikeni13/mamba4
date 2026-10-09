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

| L | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed | vs T | vs M3 |
|---:|---:|---:|---:|---:|---|---|
| 16 | 41.8% | 3.2% | 11.1% | 2.5% | trails | exceeds |
| 64 | 36.3% | 2.8% | 9.4% | 1.1% | trails | exceeds |
| 256 | 12.8% | 0.1% | 0.2% | 0.0% | trails | exceeds |
| 512 | 2.5% | 0.0% | 0.0% | 0.0% | trails | exceeds |
| 1024 | 0.0% | 0.0% | 0.0% | 0.0% | trails | matches |
| 2048 | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 4096 | 0.0% | 0.0% | 0.0% | 0.0% | matches | exceeds |

## Associative recall: top-1 accuracy at 16 queries

| K | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed | vs T | vs M3 |
|---:|---:|---:|---:|---:|---|---|
| 16 | 2.9% | 0.0% | 2.4% | 0.2% | matches | exceeds |
| 32 | 0.6% | 0.0% | 0.5% | 0.1% | matches | exceeds |
| 64 | 0.2% | 0.0% | 0.2% | 0.1% | matches | matches |
| 128 | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 240 | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 496 | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 1008 | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |
| 2032 | 0.0% | 0.0% | 0.0% | 0.0% | matches | matches |

## Passkey retrieval: exact match, pooled over five depths

| Length | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed |
|---:|---:|---:|---:|---:|
| 512 | 63.7% | 0.0% | 0.0% | 0.0% |
| 1,024 | 36.9% | 0.0% | 0.0% | 0.0% |
| 2,048 | 0.0% | 0.0% | 0.0% | 0.0% |
| 4,096 | 0.0% | 0.0% | 0.0% | 0.0% |
| 8,192 | 0.0% | 0.0% | 0.0% | 0.0% |
| 16,384 | 0.0% | 0.0% | 0.0% | 0.0% |

## Long-document NLL by position (336 documents)

| Positions | Transformer | Mamba-3 | Mamba 4 | Mamba 4, read zeroed |
|---|---:|---:|---:|---:|
| 0-128 | 3.704 | 3.625 | 3.614 | 3.650 |
| 128-512 | 3.495 | 3.440 | 3.436 | 3.538 |
| 512-1024 | 3.494 | 3.453 | 3.447 | 3.591 |
| 1024-2048 | 4.892 | 3.492 | 3.485 | 3.655 |
| 2048-4096 | 5.800 | 3.517 | 3.513 | 3.688 |
| 4096-8192 | 6.079 | 3.520 | 3.514 | 3.693 |
| 8192-16384 | 6.515 | 3.511 | 3.504 | 3.683 |

NLL at 8,192–16,384 minus NLL at 512–1,024 (paired over documents): Transformer +3.020 [+2.958, +3.085]; Mamba-3 +0.058 [+0.019, +0.098]; Mamba 4 +0.057 [+0.017, +0.097]; Mamba 4, read zeroed +0.092 [+0.053, +0.133].

## Calibration at answer slots

| Family | Transformer ECE / AUROC | Mamba-3 ECE / AUROC | Mamba 4 ECE / AUROC | Mamba 4, read zeroed ECE / AUROC |
|---|---:|---:|---:|---:|
| copy | 0.0408 / 0.761 | 0.0553 / 0.689 | 0.0348 / 0.802 | 0.0123 / 0.722 |
| mqar | 0.0135 / 0.661 | 0.0095 / 0.321 | 0.0031 / 0.855 | 0.0073 / 0.859 |
| passkey | 0.0837 / 0.653 | 0.1652 / 0.717 | 0.2149 / 0.640 | 0.2066 / 0.585 |

Mamba 4 memory-variance AUROC (lower variance predicting a correct
answer), memory layers in order: copy 0.06, 0.06, 0.05, 0.11; mqar 0.14, 0.34, 0.29, 0.26; passkey 0.41, 0.66, 0.91, 0.80.

## One-token decode (median, one sequence per chip)

| Context | Transformer ms (cache MiB) | Mamba-3 ms (cache MiB) | Mamba 4 ms (cache MiB) |
|---:|---:|---:|---:|
| 1,024 | 4.62 (22.9) | 2.77 (7.6) | 5.26 (7.3) |
| 4,096 | 10.96 (88.9) | 2.77 (7.6) | 5.27 (7.3) |
| 16,384 | 28.40 (352.9) | 2.76 (7.6) | 5.28 (7.3) |
| 65,536 | 902.75 (1408.9) | 2.75 (7.6) | 5.24 (7.3) |

Full-sequence forward time per 16,384-token document (one per chip): Transformer 0.355 s; Mamba-3 0.198 s; Mamba 4 0.759 s; Mamba 4, read zeroed 0.744 s.
