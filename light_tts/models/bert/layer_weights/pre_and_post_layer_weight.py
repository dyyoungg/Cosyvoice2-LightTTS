import torch
import numpy as np
from light_tts.common.basemodel import PreAndPostLayerWeight


class BertPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        if "bert.embeddings.word_embeddings.weight" in weights:
            self.wte_weight_ = self._cuda(weights["bert.embeddings.word_embeddings.weight"][:, :])
        if "bert.embeddings.position_embeddings.weight" in weights:
            self.pos_emb_weight = self._cuda(weights["bert.embeddings.position_embeddings.weight"][:, :])
        if "bert.embeddings.token_type_embeddings.weight" in weights:
            self.token_type_emb_weight = self._cuda(weights["bert.embeddings.token_type_embeddings.weight"][:,:])
        if "bert.embeddings.LayerNorm.weight" in weights:
            self.layer_norm_weight = self._cuda(weights["bert.embeddings.LayerNorm.weight"][:])
        if "bert.embeddings.LayerNorm.bias" in weights:
            self.layer_norm_bias = self._cuda(weights["bert.embeddings.LayerNorm.bias"][:])
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.wte_weight_, self.pos_emb_weight, self.token_type_emb_weight, self.layer_norm_weight, self.layer_norm_bias]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
