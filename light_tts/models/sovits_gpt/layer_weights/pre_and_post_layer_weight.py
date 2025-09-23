import numpy as np
from light_tts.common.basemodel import PreAndPostLayerWeight
import torch
import pdb

class TtsPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        if "model.bert_proj.weight" in weights:
            self.bert_proj_weight = self._cuda(weights["model.bert_proj.weight"].transpose(0, 1))
        if "model.bert_proj.bias" in weights:
            self.bert_proj_bias = self._cuda(weights["model.bert_proj.bias"])

        if "model.ar_text_embedding.word_embeddings.weight" in weights:
            self.text_emb = self._cuda(weights["model.ar_text_embedding.word_embeddings.weight"])
        
        if "model.ar_text_position.alpha" in weights:
            self.text_pos_alpha = self._cuda(weights["model.ar_text_position.alpha"])

        if "model.ar_audio_embedding.word_embeddings.weight" in weights:
            self.audio_emb = self._cuda(weights["model.ar_audio_embedding.word_embeddings.weight"])
        
        if "model.ar_audio_position.alpha" in weights:
            self.audio_pos_alpha = self._cuda(weights["model.ar_audio_position.alpha"])

        if "model.ar_predict_layer.weight" in weights:
            self.predict_head_weight = self._cuda(weights["model.ar_predict_layer.weight"])

        if self.text_emb is not None and self.audio_emb is not None:
            self.text_audio_emb = torch.cat([self.text_emb, self.audio_emb], dim=0)
            del self.text_emb
            del self.audio_emb
        
        return

    def verify_load(self):
        errors = "weights load not ok"
        # self.text_head_bias_, self.semantic_head_bias_
        weights = [self.bert_proj_weight, 
                   self.bert_proj_bias, 
                   self.text_pos_alpha,
                   self.audio_pos_alpha, 
                   self.predict_head_weight,
                   self.text_audio_emb]
        
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors

        return
