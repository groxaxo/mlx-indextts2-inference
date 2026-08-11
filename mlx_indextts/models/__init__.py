"""Models package for MLX-IndexTTS."""

from mlx_indextts.models.gpt import UnifiedVoice
from mlx_indextts.models.gpt2 import GPT2Model
from mlx_indextts.models.bigvgan import BigVGAN
from mlx_indextts.models.conformer import ConformerEncoder
from mlx_indextts.models.perceiver import PerceiverResampler
from mlx_indextts.models.ecapa_tdnn import ECAPATDNN
from mlx_indextts.models.activations import Snake, SnakeBeta
from mlx_indextts.models.gpt_v2 import UnifiedVoiceV2

__all__ = [
    "UnifiedVoice",
    "GPT2Model",
    "BigVGAN",
    "ConformerEncoder",
    "PerceiverResampler",
    "ECAPATDNN",
    "Snake",
    "SnakeBeta",
    "UnifiedVoiceV2",
    "UnifiedVoiceV25",
    "EnhancedCodecV25",
]


def __getattr__(name: str):
    """Load 2.5-only model classes only when the v25 extra is in use."""
    if name == "UnifiedVoiceV25":
        from mlx_indextts.models.gpt_v25 import UnifiedVoiceV25

        return UnifiedVoiceV25
    if name == "EnhancedCodecV25":
        from mlx_indextts.models.codec_v25 import EnhancedCodecV25

        return EnhancedCodecV25
    raise AttributeError(name)
