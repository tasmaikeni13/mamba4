# Frozen trained recall diagnostic

`lm.recall` evaluates the final trained checkpoint without further updates.
The fixed protocol `trained-textual-recall-v1` uses seed 20261008 independently
of training seed 42 and the exact GPT2 tokenizer used for FineWeb preparation.
Its 128 prompts form one global batch padded to sequence length 1024.

Each prompt supplies eight fresh arbitrary name-to-color facts, neutral prose,
and a question about exactly one supplied name. Colors are balanced globally:
red, blue, green, yellow, black, white, orange and purple each serve as gold
answer 16 times. Every answer is one leading-space GPT2 token. Candidate order,
fact order, target name and distractor text placement are deterministic and
shared across all architectures. The queried fact occurs once, and the answer
does not appear after the question. A model scores the logits immediately
after `Answer:`; following EOS padding cannot influence a causal model.

The six groups combine intended lengths 128, 512 and 1024 with near/far fact
placement. Near prompts place the queried fact immediately before the question;
far prompts place it near the start. Actual token lengths and fact-to-question
distances are saved per prompt, so the diagnostic does not infer distance from
a group label. Seven other facts provide association distractors; neutral prose
fills the remaining context. Short prompts retain all eight facts even if their
byte-BPE tokenization needs a few additional tokens.

`build_recall_dataset(tokenizer_path)` returns `input_ids`, `target_positions`,
`candidate_token_ids`, and `gold_index` arrays, along with complete textual
prompts and a manifest containing tokenizer/array hashes. `make_recall_step`
creates a `pmap` evaluator that returns eight full-vocabulary log probabilities
per prompt. All hosts participate in `evaluate_recall`; only these small score
arrays are gathered before prompt-level aggregation. The same prompt array hash
must be recorded for Transformer, Mamba-3 and Mamba 4.

Reported metrics include accuracy among eight candidates, the correct answer's
full-vocabulary NLL and its NLL after conditioning on the eight candidates.
Accuracy uses Wilson 95% intervals; NLL uses 5,000 deterministic whole-prompt
bootstrap samples. All raw per-prompt candidate scores, predictions, text hashes
and group results remain in the output. Prompt samples are the statistical unit;
facts or context tokens within a prompt are not independent observations.

This small synthetic diagnostic measures retrieval from supplied context. It
does not prove general recall superiority, uncertainty calibration, or robustness
across training seeds, and is not a gate for choosing hyperparameters. The
single-seed 60M screen retains every architecture's outcome. Groups are
descriptive; their unadjusted intervals do not support six separate confirmatory
claims.

## Results (frozen prompts, final checkpoints)

All models scored the identical 128 prompts; their per-prompt text hashes
match. Accuracy is among eight candidates. NLL is the gold answer's negative
log likelihood after conditioning on the eight candidates.

| Model | Correct of 128 | Accuracy (Wilson 95%) |
|---|---:|---:|
| Transformer | 18 | 0.141 (0.091–0.211) |
| Mamba-3 | 30 | 0.234 (0.169–0.315) |
| Mamba 4 | 69 | 0.539 (0.453–0.623) |
| Mamba 4 without key alignment (ablation) | 45 | 0.352 (0.274–0.438) |

Exact paired McNemar tests on the same prompts:

| Mamba 4 vs | Only Mamba 4 correct | Only the other correct | p |
|---|---:|---:|---:|
| Mamba-3 | 45 | 6 | 1.8 × 10⁻⁸ |
| Transformer | 58 | 7 | 4.3 × 10⁻¹¹ |
| The ablation | 34 | 10 | 0.0004 |

The diagnostic and its prompts were fixed before training. The tests were
computed afterwards and are descriptive. With one training seed and six
descriptive groups, they support no general recall claim. This diagnostic
scores 8-way multiple choice. The stricter claims suite (exact copying,
full-vocabulary associative recall, passkey retrieval) is reported in
`lm/results/claims-60m/`. There Mamba 4 still trails the Transformer inside
its 1,024-token window. Raw per-prompt scores are in each run's `recall.json`,
and the paired records are in `lm/results/screen-60m/recall-paired.json`.
