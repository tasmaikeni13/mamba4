# Phase 05: budget-friendly 125M selection

Planned only, after the 60M screening gate. Build Mamba 4 and Transformer at
125M matched parameters. Before allocating resources, fix a modest capped
sweep budget (for example, four 50M-token configurations per architecture),
varying learning rate and the method-specific ridge/forget constraints without
giving one model extra hidden trials. Use a development set and a fixed early
stopping rule; record unsuccessful trials and total compute.

Choose the later 3B-token FineWeb-Edu revision/sampling plan, disjoint validation
and final test set, three final seeds and a fixed training protocol. Verify
token availability and reproducible streaming. Do not claim this dataset plan
has already been prepared during the first two phases.

Gate: auditable sweep, selected settings independent of final test results,
updated parameter/cost ledger and a complete main-run configuration.
