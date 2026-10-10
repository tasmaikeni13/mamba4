# Composition: development selection rule

*Version labels normalized in the standalone cleanup; the original wording is at commit `5234621`.*

Recorded 2026-10-09 07:08 UTC, after pilots p2 and p3 and before the p4/p5
results were observed.

Every pilot replays the first 1,000 optimizer steps of the frozen screen
protocol: seed 42, initialization, data order, batch 128 x 1,024, AdamW and
the complete 7,630-step schedule. Its loss at step s is computed on training
batch s before that batch updates the model, so it is an unseen-data signal.
The held-out split is never opened by a pilot. Peers are compared on the same
batches through their logged per-step losses
(`lm/results/screen-60m/learning-curves.json`).

Selection rule for the single full Mamba 4 run:

1. Candidates must match both peers within 1% in total and non-embedding
   parameters and complete 1,000 finite steps on all 16 chips.
2. Choose the lowest mean training loss over steps 901–1,000.
3. If two candidates differ by less than 0.005 nats there, prefer the faster
   median step time.

The selected configuration is frozen as the composition of `lm/configs/mamba4-60m.json`
before the full run. Every pilot record, including losing and failed
candidates, remains in `lm/results/pilots/` and the iteration log.
The final held-out result can neither change the selection nor be replaced.
