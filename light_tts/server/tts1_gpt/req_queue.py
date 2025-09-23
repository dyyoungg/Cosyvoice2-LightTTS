import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from light_tts.utils.infer_utils import calculate_time
from light_tts.server.io_struct import Req
from light_tts.server.io_struct import ReqRunStatus, FinishStatus
from prometheus_client import Summary
gpt1_queue_size = Summary("lightllm_gpt1_queue_size", "Waiting requests in gpt1")

class ReqQueue:

    def __init__(self, args) -> None:
        self.max_total_tokens = args.max_total_token_num
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.waiting_req_bistream_list: List[Req] = []
        self.router_token_ratio = args.router_token_ratio
        self.router_max_new_token_len = args.router_max_new_token_len
        return
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def append_bistream(self, req):
        self.waiting_req_bistream_list.append(req)
        return

    def _init_cache_list(self, current_batch:Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs]
        else:
            self.cache_len_list = []

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req:Req, is_busy):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()

        ok_token_num = need_max_token_num < self.max_total_tokens
        ok_req_num = len(self.cache_len_list) <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False
    
    #@calculate_time(show=True, min_cost_ms=10)
    async def generate_new_batch(self, current_batch:Batch):

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = 0
        exist_req_num += 0 if current_batch is None else len(current_batch.reqs)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None
        
        # 计算当前所有的token使用量，包括当前使用和暂停的
        cur_all_used_tokens = 0 if current_batch is None else current_batch.batch_used_tokens
        
        # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
        cur_token_ratio = cur_all_used_tokens / self.max_total_tokens
        is_busy = cur_token_ratio >= self.router_token_ratio

        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        new_batch_first_router_need_tokens = 0 # 主要是对 prefill 或者 splitfuse 大块计算时候的限制
        aborted_count = 0

        new_waiting_bistream_list = []
        for req in self.waiting_req_bistream_list:
            if  req.req_status == ReqRunStatus.WAIT_FOR_TEXT:
                await req.try_to_fill_text()
            if req.req_status == ReqRunStatus.WAIT_IN_QUEUE or req.req_status == ReqRunStatus.WAIT_APPEND_PREFILL:
                self.waiting_req_list.append(req)
            else:
                new_waiting_bistream_list.append(req)
        self.waiting_req_bistream_list = new_waiting_bistream_list

        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE: 
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉. 
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token的泄漏
                aborted_count += 1
                continue
            req_first_router_need_tokens = req.get_first_router_need_tokens()
            if self._can_add_new_req(req, is_busy) and new_batch_first_router_need_tokens + req_first_router_need_tokens <= self.batch_max_tokens:
                can_run_list.append(req)
                new_batch_first_router_need_tokens += req_first_router_need_tokens
            else:
                break

        gpt1_queue_size.observe(len(self.waiting_req_list))
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None

