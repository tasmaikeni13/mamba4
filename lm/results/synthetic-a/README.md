# Synthetic study A (development)

Matched small models (two blocks, width 128, tied 12,288-token embedding) trained
from scratch for 3,000 steps of 256 sequences at 512 tokens, with three learning
rates each; the learning rate is chosen on a development seed and accuracy is
scored on a separate evaluation seed (256 sequences per group). This study
applied weight decay to every parameter, including gates, floors and norms;
later studies decay matrices only, as `lm.train` does. One training seed.

Per-sequence state: Transformer KV cache at 512 tokens 262,144 floats;
Mamba-3 24,576; Mamba 4 20,608.

## Associative recall (accuracy over all K queries)

| Model | K=8 | 16 | 32 | 64 | 128 | learning rate |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 32.4% | 17.5% | 9.2% | 3.7% | 1.2% | 0.0005 |
| Mamba-3 | 14.0% | 8.8% | 8.4% | 3.9% | 3.2% | 0.0015 |
| Mamba 4 (screen-60m-v2 memory layer) | 98.5% | 98.3% | 84.5% | 43.2% | 28.9% | 0.0005 |
| Mamba 4, β ≤ 16 and floor 0.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.005 |

## Retention among distractors (8 pairs; trained at 512 tokens)

| Model | 512 | 1,024 | 2,048 | 4,096 | 8,192 | learning rate |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 33.7% | 11.4% | 11.7% | 3.8% | 0.4% | 0.0015 |
| Mamba-3 | 54.6% | 0.4% | 0.1% | 0.0% | 0.0% | 0.0015 |
| Mamba 4 (screen-60m-v2 memory layer) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.005 |
| Mamba 4, β ≤ 16 and floor 0.02 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.005 |

The β ≤ 16, floor 0.02 variant produced non-finite losses at every learning
rate. The Transformer and Mamba-3 had not converged at this budget; a longer,
corrected study is needed before comparing their final capability.
