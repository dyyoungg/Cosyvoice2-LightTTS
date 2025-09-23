import time
import torch
import numpy as np
import collections
from light_tts.server.sampling_params import SamplingParams
from dataclasses import dataclass, field
from typing import List, Dict
from light_tts.common.req_manager import ReqManager
from light_tts.common.mem_manager import MemoryManager
from light_tts.utils.infer_utils import mark_start, mark_end
from light_tts.server.io_struct import ReqRunStatus
import numpy as np
import copy


requests_mapping = {}

class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        win_size: int = 10,
        tau_r: float = 0.5,
        max_new_tokens: int = 500,
        min_new_tokens: int = 1,
    ) -> None:
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.win_size = win_size
        self.tau_r = tau_r
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        return


class InferReq:

    def __init__(
        self,
        r_id,
        input_token_ids=[],
        sampling_param : InferSamplingParams=None,
        req_idx=-1,
        prompt_len=0,
        req_status=None,
        semantic_len: int = 0,
        bistream=False,
        ignore_eos=False,
        vocab_size=1,
    ) -> None:
        self.r_id = r_id
        self.sampling_param = sampling_param
        self.req_idx = req_idx
        self.prompt_len = prompt_len
        self.input_token_ids = input_token_ids
        self.output_token_ids = []
        self.req_status = req_status
        self.cur_kv_len = 0 # 当前已经占用掉 token 现存的 kv len 长度
        self.ignore_eos = ignore_eos
        self.semantic_len = semantic_len
        self.bistream=bistream
        self.next_fill_index = -1
        return


@dataclass
class InferBatch:
    batch_id: int
    request_ids: List
    req_manager: ReqManager
    
    @classmethod
    @torch.no_grad()
    def init_batch(cls, batch_id, requests, dtype: torch.dtype, device: torch.device, req_manager:ReqManager, text_vob_size, vocab_size: int):

        request_ids = []
        need_alloc_size = len([r for r in requests if r['request_id'] not in requests_mapping])
        nopad_b_req_idx = req_manager.alloc(need_alloc_size)
        nopad_b_req_idx = nopad_b_req_idx.cpu().numpy()
        
        index = 0
        for r in requests:
            # request id -> idx in list mapping
            r_id = r['request_id']

            if r_id not in requests_mapping.keys():
                tokenized_input = copy.deepcopy(r['input_id'])
                semantic_len=r["semantic_len"]
                # # 2 for <sos> and <task_id>
                input_length = len(tokenized_input)
                # postprocessor
                sampling_param = r["sampling_param"]
                assert r['req_status'] == ReqRunStatus.WAIT_IN_QUEUE
                r_obj = InferReq(r_id, 
                                input_token_ids=tokenized_input,
                                sampling_param=InferSamplingParams(**sampling_param), 
                                req_idx=nopad_b_req_idx[index], 
                                prompt_len=input_length,
                                req_status=r['req_status'],
                                semantic_len=semantic_len,
                                bistream=r["bistream"],
                                ignore_eos=r["ignore_eos"],
                                vocab_size=vocab_size)
                # 初始化存在惩罚项目
                requests_mapping[r_id] = r_obj
                index += 1
            else:
                assert r['req_status'] == ReqRunStatus.WAIT_APPEND_PREFILL
                r_obj = requests_mapping[r_id]
                new_token_start = r["new_token_start"]
                r_obj.input_token_ids += r["input_id"][new_token_start:]
                r_obj.ignore_eos = r["ignore_eos"]
            
            request_ids.append(r_id)
            
            # 初始化之后 所有请求状态置换为 RUNNING 状态
            r_obj.req_status = ReqRunStatus.RUNNING

        return cls(
            batch_id=batch_id,
            request_ids=request_ids,
            req_manager=req_manager,
        )
    
    @torch.no_grad()
    def free_self(self, reversed_req_ids=[]):
        free_req_index = []
        free_token_index = []
        for request_id in self.request_ids:
            if request_id not in reversed_req_ids:
                req : InferReq = requests_mapping.pop(request_id)
                free_req_index.append(req.req_idx)
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.cur_kv_len])
        if len(free_req_index) > 0:
            free_token_index = torch.cat(free_token_index, dim=-1)
            self.req_manager.free(free_req_index, free_token_index)
        if len(requests_mapping) == 0:
            requests_mapping.clear()
        return
    
    @torch.no_grad()
    def filter(self, request_ids: List[str], finished_request_ids: List[str], reversed_req_ids=[]):
        if len(requests_mapping) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self
        if len(request_ids) == 0:
            self.free_self()
            return InferBatch(
                batch_id=self.batch_id,
                request_ids=[],
                req_manager=self.req_manager
            )
        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req : InferReq = requests_mapping.pop(request_id)
            free_req_index.append(req.req_idx)
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][:req.cur_kv_len])
        
        if len(free_req_index) > 0:
            free_token_index = torch.cat(free_token_index, dim=-1)
            self.req_manager.free(free_req_index, free_token_index)
        
        return InferBatch(
            batch_id=self.batch_id,
            request_ids=request_ids,
            req_manager=self.req_manager,
        )

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        request_ids = batch1.request_ids + batch2.request_ids
        
        return InferBatch(
            batch_id=batch1.batch_id,
            request_ids=request_ids,
            req_manager=batch1.req_manager,
        )

    def __len__(self):
        return len(self.request_ids)
    
