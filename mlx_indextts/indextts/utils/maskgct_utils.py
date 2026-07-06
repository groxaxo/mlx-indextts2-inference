"""MaskGCT utilities for IndexTTS 2.0.

Simplified version containing only the functions needed for inference:
- build_semantic_model: Load W2V-BERT model
- build_semantic_codec: Build RepCodec model
"""

import os

import torch
from transformers import Wav2Vec2BertModel

from mlx_indextts.indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec


W2V_BERT_MODEL_ID = "facebook/w2v-bert-2.0"


def get_w2v_bert_source():
    """Return the W2V-BERT source, allowing offline/local overrides."""
    local_dir = os.environ.get("INDEXTTS_W2V_BERT_DIR")
    if local_dir:
        return local_dir
    return W2V_BERT_MODEL_ID


def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'):
    """Build W2V-BERT semantic model.

    Args:
        path_: Path to wav2vec2bert_stats.pt file containing mean/std

    Returns:
        Tuple of (semantic_model, semantic_mean, semantic_std)
    """
    semantic_model = Wav2Vec2BertModel.from_pretrained(get_w2v_bert_source())
    semantic_model.eval()
    stat_mean_var = torch.load(path_, weights_only=True)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg):
    """Build RepCodec semantic codec model.

    Args:
        cfg: Configuration object with codec parameters

    Returns:
        RepCodec model instance
    """
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    return semantic_codec
