"""Frozen architecture and experiment configuration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "transformer"
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 11
    n_heads: int = 8
    d_ff: int = 1368
    d_state: int = 64
    expand: int = 2
    head_dim: int = 64
    n_groups: int = 1
    rope_fraction: float = 0.5
    mimo_rank: int = 1
    chunk_size: int = 64
    dtype: str = "bfloat16"
    param_dtype: str = "float32"
    norm_eps: float = 1e-5
    dropout_rate: float = 0.0
    max_seq_len: int = 1024
    remat: bool = True
    mamba3_outproj_norm: bool = False
    key_dim: int = 16
    memory_floor: str = "cyclic"
    memory_decay: float = 0.99
    memory_epsilon: float = 0.05
    memory_hops: int = 1
    protected_anchor_budget: int = 4
    protected_block_size: int = 64
    protected_routing: str = "all"
    # screen-60m-v2 fields; defaults reproduce the frozen v1 composition.
    memory_mixer: str = "v1"
    layer_pattern: str = ""
    conv_kernel: int = 0
    floor_min: float = 0.25
    floor_init: float = 1.0
    order_head: bool = True
    qk_activation: str = "silu"
    memory_rope_fraction: float = 0.0
    memory_head_dim: int = 0
    memory_solver: str = "loop"

    @property
    def num_heads(self) -> int:
        return self.d_model * self.expand // self.head_dim

    @property
    def value_dim(self) -> int:
        return self.head_dim

    @property
    def memory_value_dim(self) -> int:
        """Selective memory heads may be wider than the hybrid Mamba-3 heads."""
        return self.memory_head_dim or self.head_dim

    @property
    def memory_heads(self) -> int:
        return self.d_model * self.expand // self.memory_value_dim

    @property
    def layer_kinds(self) -> str:
        """One character per layer: M conjugate memory, S official Mamba-3."""
        pattern = self.layer_pattern or "M" * self.n_layers
        if len(pattern) != self.n_layers or set(pattern) - {"M", "S"}:
            raise ValueError("layer_pattern must contain n_layers M/S characters")
        return pattern

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    token_budget: int = 1_000_000_000
    global_batch: int = 128
    sequence_length: int = 1024
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    clip_norm: float = 1.0
    eval_every: int = 500
    eval_tokens: int = 2_000_000
    checkpoint_every: int = 500
    log_every: int = 10

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch * self.sequence_length

    @property
    def steps(self) -> int:
        return (self.token_budget + self.tokens_per_step - 1) // self.tokens_per_step

    def to_dict(self) -> dict:
        return asdict(self)
