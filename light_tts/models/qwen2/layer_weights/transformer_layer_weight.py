import torch
import math
import numpy as np
from light_tts.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Qwen2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)

    def _load_qkvo_weights(self, weights):
        super()._load_qkvo_weights(weights)
        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        if self._q_bias_name in weights:
            self.q_bias_ = self._cuda(weights[self._q_bias_name])
        if self._k_bias_name in weights:
            self.k_bias_ = weights[self._k_bias_name]
        if self._v_bias_name in weights:
            self.v_bias_ = weights[self._v_bias_name]
        self._try_cat_to(["k_bias_", "v_bias_"], "kv_bias_", cat_dim=0)