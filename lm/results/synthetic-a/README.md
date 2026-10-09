# Synthetic study A (development)

Matched small models trained from scratch for 3,000 steps of 256 sequences at 512 tokens. The learning rate is chosen on a development seed; accuracy uses a separate evaluation seed. One training seed per model.

Two blocks of width 128 with a tied 12,288-token embedding. This first study applied weight decay to every parameter, including gates, floors and norms; later studies decay matrices only, as lm.train does. The Transformer and Mamba-3 had not converged at this budget, so these numbers compare learning speed, not final capability.

Per-sequence state at 512 tokens (floats): Mamba-3 24,576; Mamba 4, β ≤ 16 and floor 0.02 20,608; Mamba 4 without key shift 20,608; Transformer 262,144.

## Associative recall: accuracy over all K queries

| Model | K=8 | K=16 | K=32 | K=64 | K=128 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 32.4% | 17.5% | 9.2% | 3.7% | 1.2% | 0.0005 |
| Mamba-3 | 14.0% | 8.8% | 8.4% | 3.9% | 3.2% | 0.0015 |
| Mamba 4 without key shift | 98.5% | 98.3% | 84.5% | 43.2% | 28.9% | 0.0005 |
| Mamba 4, β ≤ 16 and floor 0.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.005 |

## Retention among distractors: 8 pairs, trained at 512 tokens

| Model | Length=512 | Length=1024 | Length=2048 | Length=4096 | Length=8192 | LR |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 33.7% | 11.4% | 11.7% | 3.8% | 0.4% | 0.0015 |
| Mamba-3 | 54.6% | 0.4% | 0.1% | 0.0% | 0.0% | 0.0015 |
| Mamba 4 without key shift | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.005 |
| Mamba 4, β ≤ 16 and floor 0.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.005 |

Non-finite training at some learning rate: mamba4-strong-lowfloor/mqar, mamba4-strong-lowfloor/noisy
