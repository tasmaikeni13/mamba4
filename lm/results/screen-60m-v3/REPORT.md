# 60M, 1B-token, single-seed screen (protocol screen-60m-v3)

| Model | Parameters | Held-out NLL | Fresh-holdout NLL | Train tokens/s | Recall accuracy |
|---|---:|---:|---:|---:|---:|
| transformer | 59,985,920 | 3.535236 | 3.496043 | 1,379,725 | 0.141 |
| mamba3 | 59,968,896 | 3.475755 | 3.437771 | 152,335 | 0.234 |
| mamba4 | 59,893,216 | 3.471891 | 3.433268 | 93,393 | 0.539 |

Declared Mamba 4 screen win: **True**.

- screen_heldout, mamba4_minus_transformer: -0.06334 nats (95% block bootstrap -0.06525 to -0.06138; 1957/2048 sequences lower)
- screen_heldout, mamba4_minus_mamba3: -0.00386 nats (95% block bootstrap -0.00521 to -0.00251; 1118/2048 sequences lower)
- fresh_holdout, mamba4_minus_transformer: -0.06277 nats (95% block bootstrap -0.06466 to -0.06081; 1971/2048 sequences lower)
- fresh_holdout, mamba4_minus_mamba3: -0.00450 nats (95% block bootstrap -0.00593 to -0.00312; 1130/2048 sequences lower)

Scheduled held-out NLL at matched optimizer steps:

| Step | Transformer | Mamba-3 | Mamba 4 v3 |
|---:|---:|---:|---:|
| 500 | 4.9144 | 4.7641 | 4.7552 |
| 1,000 | 4.3596 | 4.2364 | 4.1927 |
| 1,500 | 4.0881 | 3.9915 | 3.9694 |
| 2,000 | 3.9457 | 3.8592 | 3.8459 |
| 2,500 | 3.8592 | 3.7741 | 3.7652 |
| 3,000 | 3.7925 | 3.7164 | 3.7049 |
| 3,500 | 3.7419 | 3.6657 | 3.6580 |
| 4,000 | 3.7021 | 3.6287 | 3.6205 |
| 4,500 | 3.6581 | 3.5886 | 3.5812 |
| 5,000 | 3.6266 | 3.5587 | 3.5527 |
| 5,500 | 3.6015 | 3.5345 | 3.5299 |
| 6,000 | 3.5794 | 3.5154 | 3.5102 |
| 6,500 | 3.5586 | 3.4963 | 3.4922 |
| 7,000 | 3.5444 | 3.4841 | 3.4801 |
| 7,500 | 3.5367 | 3.4769 | 3.4731 |

Learned memory at every probe: minimum precision eigenvalue 0.7208 against floor minimum 0.7179; maximum condition 137.9.

![Learning curves](learning-curves.png)

Peers are the audited screen-60m-v1 runs; their execution sources are
byte-identical to v3 apart from Mamba 4 files and additive config fields.
Intervals treat whole 1,024-token sequences in blocks of eight as units.
One training seed cannot establish robustness. No 125M run was performed.
