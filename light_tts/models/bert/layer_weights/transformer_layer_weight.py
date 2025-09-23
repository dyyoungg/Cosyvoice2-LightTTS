import torch
import math
import numpy as np
from light_tts.common.basemodel import TransformerLayerWeight


class BertTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.query_weight_,
            self.query_bias_,
            self.key_weight_,
            self.key_bias_,
            self.value_weight_,
            self.value_bias_,
            self.att_norm_weight_,
            self.att_norm_bias_,
            self.up_proj,
            self.up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            self.ffn_norm_weight,
            self.ffn_norm_bias,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # q k v weights for llama
        if f"bert.encoder.layer.{self.layer_num_}.attention.self.query.weight" in weights:
            self.query_weight_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.self.query.weight"][:, :].transpose(0, 1))
        if f"bert.encoder.layer.{self.layer_num_}.attention.self.query.bias" in weights:
            self.query_bias_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.self.query.bias"][:])
        
        if f"bert.encoder.layer.{self.layer_num_}.attention.self.key.weight" in weights:
            self.key_weight_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.self.key.weight"][:, :].transpose(0, 1))
        if f"bert.encoder.layer.{self.layer_num_}.attention.self.key.bias" in weights:
            self.key_bias_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.self.key.bias"][:])

        if f"bert.encoder.layer.{self.layer_num_}.attention.self.value.weight" in weights:
            self.value_weight_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.self.value.weight"][:, :].transpose(0, 1))
        if f"bert.encoder.layer.{self.layer_num_}.attention.self.value.bias" in weights:
            self.value_bias_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.self.value.bias"][:])

        
        if f"bert.encoder.layer.{self.layer_num_}.attention.output.dense.weight" in weights:
            self.o_weight_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.output.dense.weight"][:, :].transpose(0, 1))
        if f"bert.encoder.layer.{self.layer_num_}.attention.output.dense.bias" in weights:
            self.o_bias_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.output.dense.bias"][:])


        # attention output dense params
        if f"bert.encoder.layer.{self.layer_num_}.attention.output.LayerNorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.output.LayerNorm.weight"])
        if f"bert.encoder.layer.{self.layer_num_}.attention.output.LayerNorm.bias" in weights:
            self.att_norm_bias_ = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.attention.output.LayerNorm.bias"])
        return

    def _load_ffn_weights(self, weights):
        if f"bert.encoder.layer.{self.layer_num_}.intermediate.dense.weight" in weights:
            self.up_proj = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.intermediate.dense.weight"].transpose(0, 1))
        if f"bert.encoder.layer.{self.layer_num_}.intermediate.dense.bias" in weights:
            self.up_proj_bias = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.intermediate.dense.bias"])

        if f"bert.encoder.layer.{self.layer_num_}.output.dense.weight" in weights:
            self.down_proj = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.output.dense.weight"].transpose(0, 1))
        if f"bert.encoder.layer.{self.layer_num_}.output.dense.bias" in weights:
            self.down_proj_bias = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.output.dense.bias"])
        
        if f"bert.encoder.layer.{self.layer_num_}.output.LayerNorm.weight" in weights:
            self.ffn_norm_weight = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.output.LayerNorm.weight"])
        if f"bert.encoder.layer.{self.layer_num_}.output.LayerNorm.bias" in weights:
            self.ffn_norm_bias = self._cuda(weights[f"bert.encoder.layer.{self.layer_num_}.output.LayerNorm.bias"])
        return
