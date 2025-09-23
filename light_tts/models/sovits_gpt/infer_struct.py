import torch
import numpy as np
from light_tts.common.basemodel import InferStateInfo
from light_tts.common.req_manager import ReqManager
import pdb

class TtsInferStateInfo(InferStateInfo):
    def __init__(self):
        super().__init__()
        self.pos_emb = None
    
    def init_some_extra_state(self, model, input_ids : torch.Tensor):
        if self.is_prefill:
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            b_semantic_len_numpy = self.b_semantic_len.cpu().numpy()
            pos_ans = []
            mask_ans = []
            for i in range(len(b_seq_len_numpy)):
                text_token_len = b_seq_len_numpy[i] - b_semantic_len_numpy[i]
                pos_ans.append(np.arange(0, text_token_len))
                mask_ans.append(np.ones(shape=(text_token_len,)))
                # print("model text ", model.max_text_position)
                pos_ans.append(np.arange(0, b_semantic_len_numpy[i]) + model.max_text_position)
                mask_ans.append(np.zeros(shape=(b_semantic_len_numpy[i],)))
            
            position_ids = torch.from_numpy(np.concatenate(pos_ans)).cuda()
            self.pos_emb = torch.index_select(model.text_semantic_pos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.mask_tensor = torch.from_numpy(np.concatenate(mask_ans)).cuda()
        else:
            position_ids =  self.b_semantic_len - 1
            self.pos_emb = torch.index_select(model.semantic_pos_cached, 0, position_ids).view(self.b_semantic_len.shape[0], -1)
            self.supported_max_input_len = model.supported_max_input_len
        
        self.o_tensor = torch.empty((input_ids.shape[0], model.config["hidden_size"]), dtype=model.dtype, device="cuda")

        return
    
    def copy_(self, new_infer_state):
        self.b_req_idx.copy_(new_infer_state.b_req_idx)
        self.b_start_loc.copy_(new_infer_state.b_start_loc)
        self.b_seq_len.copy_(new_infer_state.b_seq_len)
        self.b_semantic_len.copy_(new_infer_state.b_semantic_len)
        self.pos_emb.copy_(new_infer_state.pos_emb)
        self.mem_index.copy_(new_infer_state.mem_index)
        self.temperature.copy_(new_infer_state.temperature)
        self.has_exist_tokens.copy_(new_infer_state.has_exist_tokens)
