# Matched FineWeb-Edu data protocol

The 60M screen uses
[codelion/fineweb-edu-1B](https://huggingface.co/datasets/codelion/fineweb-edu-1B/tree/66b78cd134ab7b0545a1a618a1cc029f3091eb20),
revision `66b78cd134ab7b0545a1a618a1cc029f3091eb20`, including all ten
parquet shards. Tokenization uses the byte-level GPT2 tokenizer from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e),
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, with vocabulary size
50,257 and EOS ID 50,256. No model weights are downloaded. These source
revisions were resolved from the Hugging Face repository APIs on 2026-10-08.

```sh
uv run python scripts/prepare_fineweb.py --allow-train-replay
uv run pytest -q lm/tests/test_data.py
```

The preparation script hashes and verifies source files against their Hub LFS
SHA256 values, then tokenizes every document without implicit special tokens
and appends one EOS. Raw text is not normalized. Files, local content hashes,
library versions and preparation code hashes appear in the final manifest.
Preparation is resumable at completed source/tokenized-shard boundaries. The
ignored `data/fineweb-edu-1b/` directory needs approximately 7–9 GB, including
downloaded sources, reusable tokenized shards and final streams.

Document identity is SHA256 of exact UTF-8 text. The preparation first sorts
unique hashes, applies a NumPy PCG64 permutation with seed 42, and reserves
whole hash groups until at least 2,097,153 tokenized tokens are held out. Every
source document with a held-out hash is excluded from training. Evaluation
uses one representative per held-out group; training retains source documents
from the remaining groups, permuted using the same deterministic RNG stream.
This prevents exact-document leakage; it does not claim near-duplicate or
semantic-overlap exclusion. `documents.npy` retains source shard/row, token
offset/length, content hash and split for every document. Separate order
arrays preserve the exact packing order.

Training packs **1,000,000,001 stream tokens** to yield exactly
**1,000,000,000 next-token targets**. Evaluation packs **2,097,153 stream
tokens** to yield **2,097,152 held-out targets**. Final documents can be
truncated; their source row and truncation are recorded. The actual pinned
source supplies 998,868,400 training stream targets after evaluation exclusion,
so completing the requested budget needs **1,131,600 replayed training tokens**
(0.11316%). The explicit `--allow-train-replay` option fills that remainder from
a fresh PCG64 document permutation with seed 43, using only training documents.
The manifest records available source tokens, replay tokens, replay seed and
the full document order; evaluation documents remain excluded. Without this
option, an insufficient source budget is an error. Final binary files use
little-endian uint16,
which represents the complete 50,257-token vocabulary. The manifest appears
only after stream counts, split exclusion and file hashes are validated.

`lm.data.TokenCorpus` opens read-only memmaps and exposes `batch(step,
global_batch, sequence_length, target_budget=None, split="train")` returning
`input_ids`, `targets`, and `loss_mask` arrays. Batches map each global step to
consecutive disjoint target intervals. Sequences share only their boundary
context token. State resets at sequence boundaries; models can attend across
EOS inside a packed sequence. This policy is identical for every architecture.
EOS targets count toward the budget. `iter_batches`, `num_steps` and
`target_count` support resumes and audit checks; `verify_hashes=True` verifies
the actual binary content at open time.

With global batch 128 and sequence length 1024, the run has 7,630 optimizer
steps: 7,629 full steps and a final 51,712-target step. Remaining final-batch
positions are padded with EOS and have zero loss mask. The training objective
must divide the sum of masked cross entropy by the sum of masks, including
across all device/process shards. Data order, vocabulary and target masks must
match for Transformer, Mamba-3 and Mamba 4. Each host reads its assigned rows
of the same global batch and uses the same verified corpus hash.

The completed corpus was independently audited on 2026-10-08 by reconstructing
the SHA256 of both complete streams from the tokenized source shards and
ordered document ledger. The audit checked vocabulary bounds, all file/order
hashes, zero training/evaluation content-hash overlap, the final target mask,
and exact 1B target accounting. All 970,397 source documents have distinct
content hashes; the split retains 968,346 training documents and excludes
2,051 held-out documents. Machine-readable evidence is in the local ignored
`data/fineweb-edu-1b/audit.json` and full provenance in `manifest.json`.

| Artifact | SHA256 |
|---|---|
| `train.bin` | `8cf4568cc5cc0e1b5a47c3dd0ccfe156fffa3138f0ccfa3b146f54fb5a8c9ef9` |
| `eval.bin` | `b89de73ad474394997398be7f4e7a9e82316f7a6ef8a1b2f71a6cbb5ae9bc510` |
| `manifest.json` | `35c5390c0bf3a43131d8a4950d4a0082903bf2175c085ca233064cb5593e4054` |
