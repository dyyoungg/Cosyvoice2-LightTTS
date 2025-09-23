import torch
import numpy as np
from light_tts.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight

class CosyVoice2PreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        
        if "model.embed_tokens.weight" in weights:
            self.wte_weight_ = self._cuda(weights["model.embed_tokens.weight"][split_start:split_end, :])
            tie_word_embeddings = self.network_config_.get("tie_word_embeddings", False)
            if tie_word_embeddings:
                self.lm_head_weight_ = self.wte_weight_
        if "lm_head.weight" in weights:
            self.lm_head_weight_ = self._cuda(weights["lm_head.weight"][split_start:split_end, :])
        if "llm_embedding.weight" in weights:
            self.llm_embedding_weight_ = self._cuda(weights["llm_embedding.weight"])
        if "speech_embedding.weight" in weights:
            self.speech_embedding_weight_ = self._cuda(weights["speech_embedding.weight"])
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])
        if "llm_decoder.weight" in weights:
            self.llm_decoder_weight_ = self._cuda(weights["llm_decoder.weight"])
        if "llm_decoder.bias" in weights:
            self.llm_decoder_bias_ = self._cuda(weights["llm_decoder.bias"][:,None])
        
        if self.wte_weight_ is not None and self.llm_embedding_weight_ is not None and self.speech_embedding_weight_ is not None:
            self.text_llm_audio_emb = torch.cat([self.wte_weight_, self.llm_embedding_weight_, self.speech_embedding_weight_], dim=0)
            del self.wte_weight_
            del self.llm_embedding_weight_
            del self.speech_embedding_weight_
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.text_llm_audio_emb,
            self.lm_head_weight_,
            self.final_norm_weight_,
            self.llm_decoder_weight_,
            self.llm_decoder_bias_
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
