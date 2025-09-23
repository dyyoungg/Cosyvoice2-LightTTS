import torch
from functools import partial
from torch import nn
from light_tts.models.sovits_gpt.layer_weights.transformer_layer_weight import TtsTransformerLayerWeight
from light_tts.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from light_tts.models.sovits_gpt.infer_struct import TtsInferStateInfo
from light_tts.common.basemodel import TransformerLayerInferTpl
from cuda_copy_kv import copy_kv
from light_tts_ppl_fp16_kernel import fp16_decode_attention
from torch.nn import functional as F
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

class TtsTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self._bind_func()
        return

    def _bind_func(self):
        self._bind_norm()
        self._bind_attention()
        return

    def _bind_norm(self):
        self._att_norm = partial(TtsTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(TtsTransformerLayerInfer._ffn_norm, self)
        return

    def _bind_attention(self):
        self._context_attention_kernel = partial(TtsTransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(TtsTransformerLayerInfer._token_decode_attention_normal, self)
        self._copy_kv_to_mem_cache = partial(TtsTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        return

    def _att_norm(
        self, input, infer_state: TtsInferStateInfo, layer_weight: TtsTransformerLayerWeight
    ) -> None:
        out = F.layer_norm(input,
                    (input.shape[-1],),
                    weight=layer_weight.att_norm_weight_,
                    bias=layer_weight.att_norm_bias_,
                    eps=self.eps_)
        input[:, :] = out
        return

    def _ffn_norm(
        self, input, infer_state: TtsInferStateInfo, layer_weight: TtsTransformerLayerWeight
    ) -> None:
        out = F.layer_norm(input,
            (input.shape[-1],),
            weight=layer_weight.ffn_norm_weight_,
            bias=layer_weight.ffn_norm_bias_,
            eps=self.eps_)
        input[:, :] = out
        return

    def _get_qkv(
        self, input, cache_kv, infer_state: TtsInferStateInfo, layer_weight: TtsTransformerLayerWeight
    ) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_) + layer_weight.q_bias_
        torch.addmm(
            layer_weight.kv_bias_,
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            beta=1.0,
            alpha=1.0,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        return q, cache_kv

    def _context_attention_kernel(
        self, q, kv, infer_state: TtsInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        # o_tensor = torch.empty_like(q) if out is None else out
        o_tensor = infer_state.o_tensor
        context_attention_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.bid_seq_len,
            infer_state.max_len_in_batch,
        )
        return o_tensor

    def _get_o(
        self, input, infer_state: TtsInferStateInfo, layer_weight: TtsTransformerLayerWeight
    ) -> torch.Tensor:
        o_tensor = torch.addmm(layer_weight.o_bias_, input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _ffn(self, input, infer_state: TtsInferStateInfo, layer_weight: TtsTransformerLayerWeight) -> torch.Tensor:
        ffn1_out = torch.addmm(layer_weight.ffn_1_bias_, input.view(-1, self.embed_dim_), layer_weight.ffn_1_weight_)
        gelu_out = F.relu(ffn1_out)
        input = None
        ffn1_out = None
        ffn2_out = torch.addmm(layer_weight.ffn_2_bias_, gelu_out, layer_weight.ffn_2_weight_)
        gelu_out = None
        return ffn2_out

    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        copy_kv(mem_manager.kv_buffer[self.layer_num_], buffer, mem_index, 1)
        return
    

    def _token_decode_attention_normal(self, q, infer_state: TtsInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        # o_tensor = torch.empty_like(q) if out is None else out
        o_tensor = infer_state.o_tensor
        fp16_decode_attention(
            o_tensor.view(calcu_shape1),
            1.0 / (self.head_dim_ ** 0.5),
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            infer_state.supported_max_input_len, # cuda graph 功能开启后，该值需要一直是固定的最大长度值。与模型位置向量能支持的最大长度有关。当模型变化后需要定制修改
        )
        return o_tensor