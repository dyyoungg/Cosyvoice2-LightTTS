import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import torch
from typing import final
import numpy as np
from typing import List
from light_tts.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from .infer_struct import BertInferStateInfo
from light_tts.common.build_utils import repair_config
from .layer_weights.pre_and_post_layer_weight import BertPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import BertTransformerLayerWeight
from .layer_infer.pre_layer_infer import BertPreLayerInfer
from .layer_infer.transformer_layer_infer import BertTransformerLayerInfer
from light_tts.models.sovits_gpt.tokenizer import ReqTextSplitInfo

torch.backends.cudnn.enabled = False

class BertTpPartBaseModel:
    # weight class
    pre_and_post_weight_class = BertPreAndPostLayerWeight
    transformer_weight_class = BertTransformerLayerWeight

    # infer class
    pre_layer_infer_class = BertPreLayerInfer
    transformer_layer_infer_class = BertTransformerLayerInfer

    # infer state class
    infer_state_class = BertInferStateInfo

    def __init__(self, weight_dir):
        self.tp_rank_ = 0
        self.world_size_ = 1
        self.weight_dir_ = weight_dir

        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_weights()
        self._init_infer_layer()
        self._init_some_value()
        return
    
    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return
    
    @final
    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _verify_params(self):
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=[])
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, torch.float16, network_config=self.config, mode=[])
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            "fp16",
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=None)
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return 
    
    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=[])
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=[]) for i in range(
                self.config["n_layer"] - 2)] # bert 只需要取倒数两层的feature
        return
    
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return
    
    @torch.no_grad()
    def prefill(self, input_ids:torch.Tensor, b_start_loc:torch.Tensor, b_seq_len:torch.Tensor):
        infer_state = self.infer_state_class()
        infer_state.max_len_in_batch = int(b_seq_len.max().item())
        infer_state.b_seq_len = b_seq_len
        infer_state.b_start_loc = b_start_loc
        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                        for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        infer_state.position_ids = position_ids

        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num - 2):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])

        return input_embs
    
    @torch.no_grad()
    def prefill_ReqTextSplitInfo_list(self, reqs: List[ReqTextSplitInfo]):
        inputs = []
        b_start_loc = [0,]
        b_seq_len = []
        for req in reqs:
            for info in req.splitphones_list:
                if info.input_ids is not None:
                    inputs.append(info.input_ids)
                    b_start_loc.append(b_start_loc[-1] + len(info.input_ids))
                    b_seq_len.append(len(info.input_ids))
                else:
                    # 直接初始化结果
                    bert = torch.zeros((len(info.phones), 1024), dtype=torch.float16, device="cpu")
                    info.bert_feature = bert

        if len(inputs) != 0:
            input_ids = torch.cat(inputs, dim=0).cuda()
            del b_start_loc[-1]
            b_start_loc = torch.tensor(b_start_loc, dtype=torch.int64, device="cuda")
            b_seq_len = torch.tensor(b_seq_len, dtype=torch.int64, device="cuda")
            all_bert_feature = self.prefill(input_ids, b_start_loc, b_seq_len).detach().cpu()
            for req in reqs:
                for info in req.splitphones_list:
                    if info.input_ids is not None:
                        seq_len = len(info.input_ids)
                        cur_bert_feature = all_bert_feature[0:seq_len, :]
                        all_bert_feature = all_bert_feature[seq_len:, :]
                        cur_bert_feature = cur_bert_feature[1:-1]
                        select_index = []
                        for i in range(len(info.word2ph)):
                            select_index.extend([i] * info.word2ph[i])
                        info.bert_feature = cur_bert_feature[select_index, :]
        
        for req in reqs:
            bert_features = []
            for info in req.splitphones_list:
                # 加一个检查
                if info.input_ids is not None:
                    assert len(info.word2ph) == len(info.norm_text)
                bert_features.append(info.bert_feature)
            req.bert_feature = torch.cat(bert_features, dim=0)
            assert len(req.bert_feature) == len(req.phones)
        return 


        


        

        
    


    
