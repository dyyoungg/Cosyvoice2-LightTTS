import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from light_tts.models.bert.layer_weights.transformer_layer_weight import BertTransformerLayerWeight
from light_tts.common.basemodel import TransformerLayerInferTpl
from light_tts.models.bert.triton_kernel.context_flashattention_nopad import context_attention_fwd
from light_tts.models.bert.infer_struct import BertInferStateInfo


class BertTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"]
        self.tp_k_head_num_ = network_config["num_attention_heads"]
        self.tp_v_head_num_ = network_config["num_attention_heads"]
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        return
    
    def context_forward(self, input:torch.Tensor, infer_state: BertInferStateInfo, layer_weight: BertTransformerLayerWeight):
        #att
        q = torch.addmm(
            layer_weight.query_bias_, input.view(-1, self.embed_dim_), layer_weight.query_weight_, beta=1.0, alpha=1.0
        )
        k = torch.addmm(
            layer_weight.key_bias_, input.view(-1, self.embed_dim_), layer_weight.key_weight_, beta=1.0, alpha=1.0
        )
        v = torch.addmm(
            layer_weight.value_bias_, input.view(-1, self.embed_dim_), layer_weight.value_weight_, beta=1.0, alpha=1.0
        )

        if infer_state.mid_o is None:
            infer_state.mid_o = torch.empty_like(q)
        o = infer_state.mid_o
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o.view(-1, self.tp_o_head_num_, self.head_dim_), 
                              infer_state.b_start_loc, 
                              infer_state.b_seq_len, 
                              infer_state.max_len_in_batch)
        
        out = torch.addmm(
            layer_weight.o_bias_, o.view(-1, self.embed_dim_), layer_weight.o_weight_, beta=1.0, alpha=1.0
        )

        input.add_(out)
        input = F.layer_norm(input.view(-1, self.embed_dim_), (self.embed_dim_,), layer_weight.att_norm_weight_, layer_weight.att_norm_bias_, eps=self.eps_)
        
        ffn1_out = torch.addmm(layer_weight.up_proj_bias, input.view(-1, self.embed_dim_), layer_weight.up_proj)
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate="tanh")
        ffn1_out = None
        ffn2_out = torch.addmm(layer_weight.down_proj_bias, gelu_out, layer_weight.down_proj)
        gelu_out = None
        input.add_(ffn2_out)
        input = F.layer_norm(input.view(-1, self.embed_dim_), (self.embed_dim_,), layer_weight.ffn_norm_weight, layer_weight.ffn_norm_bias, eps=self.eps_)
        return input



    
