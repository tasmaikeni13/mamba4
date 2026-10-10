# Claims-60m: the reported evaluation

*Version labels normalized in the standalone cleanup; the original wording is at commit `5234621`.*

Recorded 2026-10-09 21:03 UTC (commit db79c18), while the final Mamba 4 was still
training and before any model was scored on these arrays.

The rules, metrics, statistics and verdicts of `PROTOCOL.md` apply unchanged.
Only the arrays and the subject model change:

- Tasks: `data/claims`, built by `validation/claims.py build --seed 20261011`
  from the pinned shard `sample/10BT/012_00000.parquet`. Array hash
  `bc0e72684b9875e81387e143312386c61726955531368123676518dead4a23ed`. The
  512 long documents exclude the screen corpus, the fresh holdout and the 336
  documents of the first evaluation.
- Subject: the final Mamba 4 (`lm/configs/mamba4-60m.json`, memory
  keys aligned to the preceding context). Verdicts are computed for it.
- Peers: the same Transformer and Mamba-3 checkpoints as before.
- Reference: the unaligned model as `mamba4_no_shift`. It is the same model
  without the key shift, an ablation, and is reported but given no verdict.
  Each Mamba 4 also gets its evaluation-time read-zeroed ablation.

These arrays were not used to select anything. A development look on CPU used
24 prompts per group from the first evaluation (`data/claims-development`) and the
step-4,000 checkpoint of the final model. At that point copy accuracy was about 2% at 16 and
64 words, and no passkey was exact.

Predictions, recorded before scoring:
- The key shift will not close the in-window copy gap to the Transformer.
- Passkey exact match will stay near zero for both Mamba 4 models.
- Long-document NLL will remain the lowest of the three architectures at
  every position bucket beyond 128 tokens.
- The key shift will change NLL by less than 0.005 nats in every bucket.
