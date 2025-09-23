from light_tts.utils.infer_utils import mark_cost_time
from light_tts.common.basemodel.infer_struct import InferStateInfo
from light_tts.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from light_tts.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight

class BaseLayerInfer:

    def __init__(self) -> None:
        pass

    @mark_cost_time("pre context forward")  # dont to remove this, will make performence down, did not know why
    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")
    
    def splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")