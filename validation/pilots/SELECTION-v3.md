# Claims repair: development selection rule

Recorded 2026-10-09 18:55 UTC, before any pilot of this rule ran.

The claims suite (`lm/results/claims-60m/`) found that the screen-60m-v2
language model rarely copies and never retrieves a passkey. The synthetic
study (`lm/runs/synthetic-a/`, development seeds only) found that the same
memory layer reaches 98% recall at 8–16 stored pairs and 100% retention to
8,192 tokens when trained on those tasks. The mechanism works. What is missing
is that language-model training rarely teaches it to retrieve. Strong writes
with a low floor (β up to 16, floor 0.02) diverged in every synthetic run, so
those candidates were dropped before piloting.

The candidate p9 is the frozen screen-60m-v2 configuration plus
`memory_key_shift`. Each value is written under the key of the context before
it, while the query sees the current context, so a read returns what followed
the current context last time. This makes copying the memory's default
geometry rather than something the convolution has to learn; H3's shift SSM
uses the same idea. It adds no parameters and no compute.

p9 replays the first 1,000 protocol steps exactly as the screen-60m-v2 pilots
did, on unseen training batches; the held-out split is never opened. p9 is
adopted for one full, versioned 60M run if all of these hold:

1. It completes 1,000 finite steps on all 16 chips, with parameter counts
   unchanged.
2. Its mean training loss over steps 901–1,000 is at most 0.005 nats above
   p5's (the frozen v2 configuration's pilot) on the identical batches.
3. Its median step time is within 10% of p5's.

Otherwise no new 60M model is trained, and the claims results stand for the
v2 model. Pilot retrieval is not measured; retrieval is judged only on the
final model with the fresh claims task set (`data/claims-v2`, seed 20261011,
shard 012), which no development step uses.
