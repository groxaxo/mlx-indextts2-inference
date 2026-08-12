"""Configuration dataclasses for IndexTTS models."""

from dataclasses import dataclass, field
from typing import List, Optional


# Keep IndexTTS 2.5 short-form generation bounded by default.  Callers may
# explicitly raise this per segment for long-form work.
DEFAULT_V25_MAX_MEL_TOKENS = 256


def default_max_mel_tokens(model_version: str) -> int:
    """Return the conservative per-segment generation cap for a model version."""
    if model_version == "2.5":
        return DEFAULT_V25_MAX_MEL_TOKENS
    if model_version == "2.0":
        return 1500
    return 800


@dataclass
class ConformerConfig:
    """Configuration for Conformer encoder."""
    input_size: int = 100  # mel bands for 1.5, 1024 for 2.0 (W2V-BERT)
    output_size: int = 512
    linear_units: int = 2048
    attention_heads: int = 8
    num_blocks: int = 6
    dropout_rate: float = 0.0
    input_layer: str = "conv2d2"
    pos_enc_layer_type: str = "rel_pos"
    normalize_before: bool = True
    use_cnn_module: bool = True
    cnn_module_kernel: int = 15
    perceiver_mult: int = 2


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    model_dim: int = 1024
    heads: int = 16
    layers: int = 20
    max_mel_tokens: int = 605
    max_text_tokens: int = 402

    # Token configuration
    number_text_tokens: int = 12000
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    start_text_token: int = 0
    stop_text_token: int = 1

    # Conditioning
    use_mel_codes_as_input: bool = True
    mel_length_compression: int = 1024
    condition_type: str = "conformer_perceiver"
    condition_num_latent: int = 32
    max_conditioning_inputs: int = 1

    # Conformer config (nested)
    condition_module: Optional[ConformerConfig] = None
    # Emotion condition module (for v2)
    emo_condition_module: Optional[ConformerConfig] = None

    def __post_init__(self):
        if self.condition_module is None:
            self.condition_module = ConformerConfig()


@dataclass
class BigVGANConfig:
    """Configuration for BigVGAN vocoder."""
    # Architecture
    resblock: str = "1"
    upsample_rates: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [8, 8, 4, 4, 4, 4])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    # Input
    gpt_dim: int = 1024
    num_mels: int = 100

    # Speaker embedding
    speaker_embedding_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # Activation
    activation: str = "snakebeta"
    snake_logscale: bool = True

    # Misc
    feat_upsample: bool = False
    use_tanh_at_final: bool = True


@dataclass
class MelConfig:
    """Configuration for mel spectrogram extraction."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    normalize: bool = False


@dataclass
class IndexTTSConfig:
    """Main configuration for IndexTTS model."""
    gpt: GPTConfig = field(default_factory=GPTConfig)
    bigvgan: BigVGANConfig = field(default_factory=BigVGANConfig)
    mel: MelConfig = field(default_factory=MelConfig)

    # Paths
    bpe_model: str = "bpe.model"
    gpt_checkpoint: str = "gpt.pth"
    bigvgan_checkpoint: str = "bigvgan_generator.pth"

    # Version
    version: Optional[float] = None
    sample_rate: int = 24000

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "IndexTTSConfig":
        """Load configuration from YAML file."""
        import yaml
        import dataclasses

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Helper to filter dict to only known fields
        def filter_fields(data_dict: dict, dataclass_type) -> dict:
            known_fields = {f.name for f in dataclasses.fields(dataclass_type)}
            return {k: v for k, v in data_dict.items() if k in known_fields}

        # Parse nested configs
        gpt_data = data.get("gpt", {}).copy()
        condition_module_data = gpt_data.pop("condition_module", {})
        emo_condition_module_data = gpt_data.pop("emo_condition_module", {})

        gpt_config = GPTConfig(
            **filter_fields(gpt_data, GPTConfig),
            condition_module=ConformerConfig(**filter_fields(condition_module_data, ConformerConfig)) if condition_module_data else None,
            emo_condition_module=ConformerConfig(**filter_fields(emo_condition_module_data, ConformerConfig)) if emo_condition_module_data else None,
        )

        bigvgan_config = BigVGANConfig(**filter_fields(data.get("bigvgan", {}), BigVGANConfig))

        mel_data = data.get("dataset", {}).get("mel", {})
        mel_config = MelConfig(**filter_fields(mel_data, MelConfig)) if mel_data else MelConfig()

        return cls(
            gpt=gpt_config,
            bigvgan=bigvgan_config,
            mel=mel_config,
            bpe_model=data.get("dataset", {}).get("bpe_model", "bpe.model"),
            gpt_checkpoint=data.get("gpt_checkpoint", "gpt.pth"),
            bigvgan_checkpoint=data.get("bigvgan_checkpoint", "bigvgan_generator.pth"),
            version=data.get("version"),
            sample_rate=data.get("dataset", {}).get("sample_rate", 24000),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_omegaconf(cls, cfg) -> "IndexTTSConfig":
        """Load configuration from OmegaConf object (IndexTTS 2.0 config).

        Args:
            cfg: OmegaConf object loaded from config.yaml

        Returns:
            IndexTTSConfig instance
        """
        from omegaconf import OmegaConf
        import dataclasses

        # Helper to filter dict to only known fields
        def filter_fields(data_dict: dict, dataclass_type) -> dict:
            known_fields = {f.name for f in dataclasses.fields(dataclass_type)}
            return {k: v for k, v in data_dict.items() if k in known_fields}

        # Convert OmegaConf to dict
        gpt_data = OmegaConf.to_container(cfg.gpt, resolve=True)
        condition_module_data = gpt_data.pop("condition_module", {})
        emo_condition_module_data = gpt_data.pop("emo_condition_module", {})

        gpt_config = GPTConfig(
            **filter_fields(gpt_data, GPTConfig),
            condition_module=ConformerConfig(**filter_fields(condition_module_data, ConformerConfig)) if condition_module_data else None,
            emo_condition_module=ConformerConfig(**filter_fields(emo_condition_module_data, ConformerConfig)) if emo_condition_module_data else None,
        )

        # BigVGAN config (v2 doesn't have bigvgan in config, use defaults)
        bigvgan_config = BigVGANConfig()

        # Mel config from s2mel preprocess params
        if hasattr(cfg, 's2mel') and hasattr(cfg.s2mel, 'preprocess_params'):
            mel_data = {
                'sample_rate': cfg.s2mel.preprocess_params.sr,
                'n_fft': cfg.s2mel.preprocess_params.spect_params.n_fft,
                'hop_length': cfg.s2mel.preprocess_params.spect_params.hop_length,
                'win_length': cfg.s2mel.preprocess_params.spect_params.win_length,
                'n_mels': cfg.s2mel.preprocess_params.spect_params.n_mels,
            }
            mel_config = MelConfig(**filter_fields(mel_data, MelConfig))
        else:
            mel_config = MelConfig()

        return cls(
            gpt=gpt_config,
            bigvgan=bigvgan_config,
            mel=mel_config,
            bpe_model=cfg.dataset.bpe_model if hasattr(cfg, 'dataset') else "bpe.model",
            gpt_checkpoint=cfg.gpt_checkpoint if hasattr(cfg, 'gpt_checkpoint') else "gpt.pth",
            bigvgan_checkpoint="",  # v2 uses external BigVGAN
            version=cfg.version if hasattr(cfg, 'version') else 2.0,
            sample_rate=cfg.s2mel.preprocess_params.sr if hasattr(cfg, 's2mel') else 22050,
        )
