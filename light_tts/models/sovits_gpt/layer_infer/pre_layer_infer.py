import torch
import torch.distributed as dist
import numpy as np

from light_tts.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from light_tts.models.sovits_gpt.layer_weights.pre_and_post_layer_weight import TtsPreAndPostLayerWeight
from light_tts.models.sovits_gpt.infer_struct import TtsInferStateInfo
from light_tts.common.basemodel import PreLayerInferTpl
from light_tts.utils.infer_utils import mark_cost_time
from torch.nn import functional as F
import torch
from torch import nn

from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

class TtsPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: TtsInferStateInfo, layer_weight: TtsPreAndPostLayerWeight):
        input_emb = torch.embedding(layer_weight.text_audio_emb, input_ids, padding_idx=-1)
        bert_input = torch.addmm(layer_weight.bert_proj_bias, infer_state.bert_features, layer_weight.bert_proj_weight) # bert 特征必须按照顺序排列好，否则可能加错位置
        input_emb[infer_state.mask_tensor == 1, :] += bert_input
        return input_emb + infer_state.pos_emb



    def token_forward(self, input_ids, infer_state: TtsInferStateInfo, layer_weight: TtsPreAndPostLayerWeight):
        return torch.embedding(layer_weight.text_audio_emb, input_ids, padding_idx=-1) + infer_state.pos_emb
