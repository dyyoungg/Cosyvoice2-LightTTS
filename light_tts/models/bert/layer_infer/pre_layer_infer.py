import torch
import torch.distributed as dist
import numpy as np

from light_tts.models.bert.layer_weights.pre_and_post_layer_weight import BertPreAndPostLayerWeight
from light_tts.models.bert.infer_struct import BertInferStateInfo
from light_tts.common.basemodel import PreLayerInferTpl
from light_tts.utils.infer_utils import mark_cost_time
import torch.nn.functional as F


class BertPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        tp_vob_ids = np.linspace(0, network_config["vocab_size"], self.world_size_ + 1, dtype=np.int64)
        self.vob_start_id_, self.vob_end_id_ = int(tp_vob_ids[self.tp_rank_]), int(tp_vob_ids[self.tp_rank_ + 1])
        return
    
    def context_forward(self, input_ids, infer_state: BertInferStateInfo, layer_weight: BertPreAndPostLayerWeight):
        input_embdings = torch.embedding(layer_weight.wte_weight_, input_ids)
        pos_embdings = torch.embedding(layer_weight.pos_emb_weight, infer_state.position_ids)
        input_embdings.add_(pos_embdings)
        pos_embdings = None
        input_embdings.add_(layer_weight.token_type_emb_weight[0])
        input_embdings = F.layer_norm(input_embdings, (input_embdings.shape[1],), layer_weight.layer_norm_weight, layer_weight.layer_norm_bias, eps=1e-12)
        return input_embdings