import torch
import math
import numpy as np
from light_tts.common.basemodel import TransformerLayerWeight


class TtsTransformerLayerWeight(TransformerLayerWeight):
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
            self.att_norm_weight_,
            self.att_norm_bias_,
            self.q_weight_,
            self.kv_weight_,
            self.q_bias_,
            self.kv_bias_,
            self.o_weight_,
            self.o_bias_,
            self.ffn_norm_weight_,
            self.ffn_norm_bias_,
            self.ffn_1_weight_,
            self.ffn_1_bias_,
            self.ffn_2_weight_,
            self.ffn_2_bias_,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        att_norm_weight_name = f"model.h.layers.{self.layer_num_}.norm1.weight"
        if att_norm_weight_name in weights:
            self.att_norm_weight_ = self._cuda(weights[att_norm_weight_name])
        att_norm_bias_name = f"model.h.layers.{self.layer_num_}.norm1.bias"
        if att_norm_bias_name in weights:
            self.att_norm_bias_ = self._cuda(weights[att_norm_bias_name])

        # attention params
        n_embed = self.network_config_["hidden_size"]
        qkv_weight_name = f"model.h.layers.{self.layer_num_}.self_attn.in_proj_weight"
        qkv_bias_name = f"model.h.layers.{self.layer_num_}.self_attn.in_proj_bias"
        if qkv_weight_name in weights:
            qkv_weights = weights[qkv_weight_name]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)

            self.q_weight_ = self._cuda(q_weights[
                n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1), :
            ].transpose(0, 1))
            self.k_weight_ = k_weights[
                n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1), :
            ].transpose(0, 1)
            self.v_weight_ = v_weights[
                n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1), :
            ].transpose(0, 1)

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        if qkv_bias_name in weights:
            qkv_bias = weights[qkv_bias_name]
            split_size = qkv_bias.shape[0] // 3
            q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size, dim=0)
            self.q_bias_ = self._cuda(q_bias[n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1)])
            self.k_bias_ = k_bias[n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1)]
            self.v_bias_ = v_bias[n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1)]

        self._try_cat_to(["k_bias_", "v_bias_"], "kv_bias_", cat_dim=0)
        
        o_weight_name = f"model.h.layers.{self.layer_num_}.self_attn.out_proj.weight"
        o_bias_name = f"model.h.layers.{self.layer_num_}.self_attn.out_proj.bias"

        # attention output dense params
        if o_weight_name in weights:
            self.o_weight_ = self._cuda(weights[o_weight_name][
                :, n_embed * self.tp_rank_ : n_embed * (self.tp_rank_ + 1)].transpose(0, 1))

        if o_bias_name in weights:
            self.o_bias_ = self._cuda(weights[o_bias_name])

    def _load_ffn_weights(self, weights):
        ffn_norm_weight_name = f"model.h.layers.{self.layer_num_}.norm2.weight"
        ffn_norm_bias_name = f"model.h.layers.{self.layer_num_}.norm2.bias"
        if ffn_norm_weight_name in weights:
            self.ffn_norm_weight_ = self._cuda(weights[ffn_norm_weight_name])
        if ffn_norm_bias_name in weights:
            self.ffn_norm_bias_ = self._cuda(weights[ffn_norm_bias_name])

        # ffn params
        inter_size = self.network_config_["intermediate_size"]
        ffn_1_weight_name = f"model.h.layers.{self.layer_num_}.linear1.weight"
        ffn_1_bias_name = f"model.h.layers.{self.layer_num_}.linear1.bias"

        if ffn_1_weight_name in weights:
            self.ffn_1_weight_ = weights[ffn_1_weight_name]
            self.ffn_1_weight_ = self._cuda(self.ffn_1_weight_[inter_size * self.tp_rank_ : inter_size * (self.tp_rank_ + 1), :].transpose(0, 1))

        if ffn_1_bias_name in weights:
            self.ffn_1_bias_ = self._cuda(weights[ffn_1_bias_name][inter_size * self.tp_rank_ : inter_size * (self.tp_rank_ + 1)])
        
        ffn_2_weight_name = f"model.h.layers.{self.layer_num_}.linear2.weight"
        ffn_2_bias_name = f"model.h.layers.{self.layer_num_}.linear2.bias"

        if ffn_2_weight_name in weights:
            self.ffn_2_weight_ = weights[ffn_2_weight_name]
            self.ffn_2_weight_ = self._cuda(self.ffn_2_weight_[:, inter_size * self.tp_rank_ : inter_size * (self.tp_rank_ + 1)].transpose(0, 1))

        if ffn_2_bias_name in weights:
            self.ffn_2_bias_ = self._cuda(weights[ffn_2_bias_name])

        return
