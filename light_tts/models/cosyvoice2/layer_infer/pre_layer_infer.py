import torch
import torch.distributed as dist
import numpy as np

from light_tts.models.cosyvoice2.layer_weights.pre_and_post_layer_weight import CosyVoice2PreAndPostLayerWeight
from light_tts.models.llama.infer_struct import LlamaInferStateInfo
from light_tts.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from light_tts.utils.infer_utils import mark_cost_time


class CosyVoice2PreLayerInfer(LlamaPreLayerInfer):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: CosyVoice2PreAndPostLayerWeight):
        input_embdings = torch.embedding(layer_weight.text_llm_audio_emb, input_ids, padding_idx=-1)
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: CosyVoice2PreAndPostLayerWeight):
        input_embdings = torch.embedding(layer_weight.text_llm_audio_emb, input_ids, padding_idx=-1)
        return input_embdings