import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from light_tts.models.sovits_gpt.layer_weights.pre_and_post_layer_weight import TtsPreAndPostLayerWeight
from light_tts.models.sovits_gpt.infer_struct import TtsInferStateInfo
from light_tts.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from light_tts.common.basemodel import PostLayerInferTpl
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

class TtsPostLayerInfer(PostLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        return

    def _slice_get_last_input(self, input_embdings, infer_state: TtsInferStateInfo):

        if infer_state.is_prefill:
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size

        assert False, "Error State"

    def token_forward(
        self,
        input_embdings,
        infer_state: TtsInferStateInfo,
        layer_weight: TtsPreAndPostLayerWeight,
        return_logics=False,
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None
        last_input = last_input.permute(1, 0)
        logic_batch = torch.mm(layer_weight.predict_head_weight, last_input)
        logic_batch = logic_batch.float().permute(1, 0)
        return logic_batch

