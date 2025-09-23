import os
import torch
from light_tts.models.sovits_gpt.layer_infer.pre_layer_infer import TtsPreLayerInfer
from light_tts.models.sovits_gpt.layer_infer.post_layer_infer import TtsPostLayerInfer
from light_tts.models.sovits_gpt.layer_infer.transformer_layer_infer import TtsTransformerLayerInfer
from light_tts.models.sovits_gpt.layer_weights.pre_and_post_layer_weight import TtsPreAndPostLayerWeight
from light_tts.models.sovits_gpt.layer_weights.transformer_layer_weight import TtsTransformerLayerWeight
from light_tts.models.sovits_gpt.infer_struct import TtsInferStateInfo
from light_tts.common.basemodel import TpPartBaseModel
from light_tts.common.mem_utils import select_mem_manager_class
from light_tts.utils.log_utils import init_logger
from light_tts.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from light_tts.utils.config_utils import get_config_json, get_style_gpt_path
import torch
import math

logger = init_logger(__name__)

class TTSTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = TtsPreAndPostLayerWeight
    transformer_weight_class = TtsTransformerLayerWeight

    # infer class
    pre_layer_infer_class = TtsPreLayerInfer
    post_layer_infer_class = TtsPostLayerInfer
    transformer_layer_infer_class = TtsTransformerLayerInfer

    # infer state class
    infer_state_class = TtsInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
        
    def _init_config(self):
        self.weight_path = get_style_gpt_path(self.weight_dir_, self.style_name)

        model_datas = torch.load(self.weight_path, map_location="cpu")
        self.weight_dict = model_datas["weight"]
        model_config = model_datas["config"]

        self.config =  {
            "eos_id": model_config["model"]["EOS"],
         
            # "max_position": 1024,
            # "max_position_embeddings": 1024,
            
            "hidden_size": model_config["model"]["hidden_dim"],
            "n_embed": model_config["model"]["hidden_dim"],
            
            "intermediate_size": model_config["model"]["linear_units"],
            
            "num_attention_heads": model_config["model"]["head"],
            "num_key_value_heads": model_config["model"]["head"],
            
            "num_hidden_layers": model_config["model"]["n_layer"],
            "n_layer": model_config["model"]["n_layer"],
            
            "vocab_size": model_config["model"]["vocab_size"],

            "phoneme_vocab_size": model_config["model"]["phoneme_vocab_size"],

            "rms_norm_eps": 1e-5
        }

        self.text_vob_size = self.config["phoneme_vocab_size"]
        self.audio_vob_size = self.config["vocab_size"]

        self.dtype  =torch.float16
        return

    def _verify_must(self):
        return
    
    def _verify_params(self):
        return

    def _init_custom(self):
        """
        模型特殊的一些初始化
        """
        self._init_to_get_pos_emb()
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, self.dtype, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, self.dtype, network_config=self.config, mode=self.mode)
            for i in range(self.config.get("num_hidden_layers"))
        ]

        load_hf_weights(
            "fp32" if self.dtype == torch.float32 else "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict)
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        del self.weight_dict           
        return 

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(self.max_total_token_num, 
                                                     dtype=self.dtype,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"],
                                                     always_copy=True)
        return

    def _init_to_get_pos_emb(self,):
        self.max_text_position = self.max_seq_length # 文本最大位置长度
        self.max_semantic_position = self.max_seq_length # 语音最大位置长度
        
        hidden_size = self.config["hidden_size"]
        position_ids = torch.arange(self.max_text_position, dtype=torch.long, device="cpu").view(-1, 1)
        pe = torch.zeros(self.max_text_position, hidden_size, device="cpu")
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32, device="cpu")
            * -(math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position_ids * div_term)
        pe[:, 1::2] = torch.cos(position_ids * div_term)
        
        self.semantic_pos_cached = (pe.cuda()*self.pre_post_weight.audio_pos_alpha).half().cuda()
        self.text_semantic_pos_cached = torch.cat([pe.cuda()*self.pre_post_weight.text_pos_alpha, self.semantic_pos_cached], dim=0).half()

        self.supported_max_input_len = self.max_seq_length
        return