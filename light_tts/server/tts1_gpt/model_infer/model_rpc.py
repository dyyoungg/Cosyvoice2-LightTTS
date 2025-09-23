import asyncio
import os
import numpy as np
import rpyc
import torch
import traceback
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from .infer_batch import InferBatch
from light_tts.models.cosyvoice2.model import CosyVoice2TpPartModel
from light_tts.utils.infer_utils import set_random_seed
from light_tts.utils.infer_utils import calculate_time, mark_start, mark_end
from .pre_process import prepare_decode_inputs, prepare_prefill_inputs
from .post_process import sampling_ids
from .infer_batch import requests_mapping
from .infer_batch import InferReq
from light_tts.server.io_struct import ReqRunStatus
from light_tts.utils.log_utils import init_logger
from functools import partial
from light_tts.server.shm_tools.shm_objs import SharedSpeechManager

class ModelRpcServer(rpyc.Service):

    @torch.no_grad()
    def exposed_init_model(self, kvargs):
        import torch
        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.load_way = kvargs["load_way"]
        self.style_name = kvargs["style_name"]
        self.mode = kvargs["mode"]
        self.speech_token_size = kvargs["speech_token_size"]
        self.mix_ratio = kvargs.get("mix_ratio", [5, 15])

        self.cache = {}
        self.logger = init_logger(__name__)

        weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]
        
        torch.cuda.set_device(0)        

        model_kvargs = {
            "tp_rank": self.tp_rank,
            "world_size": self.world_size,
            "weight_dir": os.path.join(weight_dir, 'Qwen2-0.5B-CosyVoice-BlankEN'),
            "pt_dir": os.path.join(weight_dir, 'llm.pt'),
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "style_name": self.style_name,
            "speech_token_size": self.speech_token_size,
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
        }

        port = kvargs["port"]
        self.fill_token_id = self.speech_token_size + 2

        try:
            self.model = CosyVoice2TpPartModel(model_kvargs)
        except Exception as e:
            self.logger.error(f"load model error: {str(e)} {e} {type(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
        torch.cuda.empty_cache()
        set_random_seed(2147483647)
        return
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    @torch.no_grad()
    def exposed_add_batch(self, batch_id, reqs, dtype):
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, 
                                           torch.cuda.current_device(), self.model.req_manager, 
                                           self.model.text_vob_size,
                                           self.model.speech_token_size)
        self.cache[batch_id] = batch_data

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj : InferReq  = requests_mapping[req_id]
            ans[req_id] = (req_obj.req_status, req_obj.cur_kv_len)
        return ans
    
    @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    # @calculate_time(show=True, min_cost_ms=0.1)
    @torch.no_grad()
    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list, reversed_req_ids):
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list, reversed_req_ids)
        del batch
        self.cache[batch_id] = filter_batch
        return
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    @torch.no_grad()
    def exposed_remove_batch(self, batch_id, reversed_req_ids):
        batch = self.cache.pop(batch_id)
        batch.free_self(reversed_req_ids)
        del batch
        return
    
    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def forward(self, batch_id, is_prefill):
        
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(batch)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch)
            # print(kwargs)
        
        # 2 for <sos> and <task_id>
        text_vocab = self.model.text_vob_size + 2
        logits = self.model.forward(**kwargs)
        
        mask = ~kwargs["b_next_fill"]
        try:
            next_token_ids = sampling_ids(
                logits[mask], kwargs["output_token_ids"][mask], 
                kwargs["ignore_eos"][mask],
                kwargs["bistream"][mask],
                self.model.speech_token_size,
                self.model.speech_token_size + 2
            )
        except Exception as e:
            print("sampling_ids error")
            print(e)
            traceback.print_exc()
            print(logits[mask].shape)
            print(kwargs["output_token_ids"][mask])
            print(kwargs["ignore_eos"][mask])
            print(kwargs["bistream"][mask])
            print(self.model.speech_token_size)
            print(self.model.speech_token_size + 2)
            print("sampling_ids error")
            raise e
        next_token_ids = next_token_ids.detach().cpu().numpy()
        
        index = 0
        for req_obj in run_reqs:
            # prefill and decode is same
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            if mask[index]:
                next_token_id = next_token_ids[index]
                index += 1
            else:
                next_token_id = self.fill_token_id
                req_obj.next_fill_index += self.mix_ratio[1] + 1
            if next_token_id == self.fill_token_id:
                req_obj.next_fill_index = len(req_obj.output_token_ids) + self.mix_ratio[1] + 1
            else:
                req_obj.input_token_ids.append(next_token_id + text_vocab) # 生成的token id 需要偏移
            req_obj.output_token_ids.append(next_token_id)
            metadata = None
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict
    
    
class ModelRpcClient:
    def __init__(self, model_rpc):
        self.model: ModelRpcServer = model_rpc
        self._init_model = self.model.exposed_init_model
        self._add_batch = self.model.exposed_add_batch
        self._prefill_batch = self.model.exposed_prefill_batch
        self._decode_batch = self.model.exposed_decode_batch
        self._filter_batch = self.model.exposed_filter_batch
        self._merge_batch = self.model.exposed_merge_batch
        self._remove_batch = self.model.exposed_remove_batch
        return

    async def init_model(self, kvargs):
        ans : rpyc.AsyncResult = self._init_model(kvargs)
        return

    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        return ans

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        return ans

    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        return ans

    async def filter_batch(self, batch_id, req_id_list, finished_req_id_list, append_prefill_req_ids):
        ans = self._filter_batch(batch_id, req_id_list, finished_req_id_list, append_prefill_req_ids)
        return 

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        return

    async def remove_batch(self, batch_id, append_prefill_req_ids):
        ans = self._remove_batch(batch_id, append_prefill_req_ids)
        return

async def start_model_process():
    return ModelRpcClient(ModelRpcServer())
    

