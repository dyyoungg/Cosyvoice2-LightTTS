from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import enum
import time

# for bistream
bistream_lock = asyncio.Lock()

class ReqRunStatus(enum.Enum):
    WAIT_IN_QUEUE = 0 # 在队列中等待
    RUNNING = 1 # 运行
    PAUSED_AND_KVKEEP = 2 # 暂停保留KV
    PAUSED_AND_OFFLOAD = 3 # 暂停卸载KV
    RERUNNING_FROM_KVKEEP = 4 # 从暂停中恢复
    RERUNNING_FROM_OFFLOAD = 5 # 从卸载KV中恢复
    WAIT_APPEND_PREFILL = 6 # 等待再次prefill
    WAIT_FOR_TEXT = 7 # 等待文本输入

class FinishStatus(enum.Enum):
    NO_FINISH = 0 # 没有结束
    FINISHED_STOP = 1 # 因为遇到了STOP token 而结束
    FINISHED_LENGTH = 2 # 因为长度达到了最大长度而结束
    FINISHED_ABORT = 3 # 因为请求被中止而结束

    def is_finished(self):
        return 1 <= self.value <= 3
    
    def is_aborted(self):
        return self == FinishStatus.FINISHED_ABORT

    def get_finish_reason(self):
        if self == FinishStatus.FINISHED_STOP:
            finish_reason = "stop"
        elif self == FinishStatus.FINISHED_LENGTH:
            finish_reason = "length"
        elif self == FinishStatus.FINISHED_ABORT:
            finish_reason = "abort"
        else:
            finish_reason = None
        return finish_reason

class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, semantic_len, speech_index):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.sample_params = sample_params
        self.speech_index = speech_index
        self.ignore_eos = True

        self.max_output_len = sample_params.max_new_tokens
        self.output_ids = []
        self.output_metadata_list = []

        self.req_status = ReqRunStatus.WAIT_IN_QUEUE
        self.finish_status = FinishStatus.NO_FINISH
        self.cur_kv_len = 0
        self.new_token_start = 0
        self.semantic_len = semantic_len
        self.start_time = time.time()
        return
    
    def to_rpc_obj(self):
        return {"request_id": self.request_id,
                "input_id": self.prompt_ids,
                "output_len": self.max_output_len,
                "sampling_param": self.sample_params.to_dict(),
                "semantic_len": self.semantic_len,
                "speech_index": self.speech_index,
                "bistream": self.bistream,
                "ignore_eos": self.ignore_eos,
                "new_token_start": self.new_token_start,
                "req_status": self.req_status}
    
    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"prompt_ids={self.prompt_ids}, ")
    
    def get_used_tokens(self):
        return max(0, self.cur_kv_len)

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        raise Exception("need to impl")
    
    def get_decode_need_tokens(self):
        raise Exception("need to impl")
    
    def get_first_router_need_tokens(self):
        raise Exception("need to impl")
    
class CosyVoice2BistreamReq(Req):
    def __init__(self, request_id, prompt_ids, text_ids, audio_ids, mix_ratio, sample_params: SamplingParams, speech_index, stream, sos_eos, task_id):
        super().__init__(request_id, [sos_eos], sample_params, len(audio_ids), speech_index)
        self.text_ids = text_ids
        self.audio_ids = audio_ids
        self.text_cache = prompt_ids + text_ids
        self.mix_ratio = mix_ratio
        self.stream = stream
        self.bistream = True
        self.sos_eos = sos_eos
        self.task_id = task_id
        self.token_offset = 0
        self.receiving_text = True
        self.first = True
        self.ignore_eos = True
        self.req_status = ReqRunStatus.WAIT_FOR_TEXT
        return
    
    async def try_to_fill_text(self):
        async with bistream_lock:
            if self.first:
                self.init_prompt()
            else:
                self.fill_token()

    def init_prompt(self):
        while len(self.audio_ids) > 0:
            if len(self.text_cache) >= self.mix_ratio[0]:
                self.prompt_ids += self.text_cache[:self.mix_ratio[0]] + self.audio_ids[:self.mix_ratio[1]]
                self.text_cache = self.text_cache[self.mix_ratio[0]:]
                self.audio_ids = self.audio_ids[self.mix_ratio[1]:]
            else:
                break

        self.input_len = len(self.prompt_ids)
        if len(self.audio_ids) > 0:
            if self.receiving_text:
                return
            else:
                self.prompt_ids = self.prompt_ids + self.text_cache + [self.task_id] + self.audio_ids
                self.text_cache = []
                self.audio_ids = []
                self.ignore_eos = False
                self.input_len = len(self.prompt_ids)
        self.first = False
        self.req_status = ReqRunStatus.WAIT_IN_QUEUE

    def fill_token(self):
        # 从text_cache中填充token
        if len(self.text_cache) > self.mix_ratio[0]:
            self.new_token_start = len(self.prompt_ids)
            self.prompt_ids += self.text_cache[:self.mix_ratio[0]]
            self.text_cache = self.text_cache[self.mix_ratio[0]:]
            self.req_status = ReqRunStatus.WAIT_APPEND_PREFILL
        else:
            if self.receiving_text:
                self.req_status = ReqRunStatus.WAIT_FOR_TEXT
            else:
                self.new_token_start = len(self.prompt_ids)
                self.prompt_ids = self.prompt_ids + self.text_cache + [self.task_id] + self.audio_ids
                self.text_cache = []
                self.audio_ids = []
                self.ignore_eos = False
                self.req_status = ReqRunStatus.WAIT_APPEND_PREFILL

        self.input_len = len(self.prompt_ids)
    
    def ready_for_infer(self):
        return self.req_status == ReqRunStatus.WAIT_IN_QUEUE or self.req_status == ReqRunStatus.WAIT_APPEND_PREFILL

    async def append_input(self, finish, text_ids):
        async with bistream_lock:
            if finish:
                self.receiving_text = False
            else:
                self.text_cache += text_ids
                self.max_output_len += 20 * len(text_ids)
    
    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        """
        普通continues batch调度模式, 先prefill 后 decode 的估计方式 的实现
        """
        has_out_len = len(self.output_ids)
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            # 用当前输出长度的 1.1 倍作为预估输出长度的另一个参考量，用于更新估计的最大输出长度量
            # 后续会更新为更合理的统计条件概率估计方式 to do
            cur_max_new_token_len = min(self.max_output_len, max(int(1.1 * has_out_len), router_max_new_token_len))

        if self.req_status == ReqRunStatus.RUNNING:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (self.input_len + 1, max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD or self.req_status == ReqRunStatus.WAIT_APPEND_PREFILL:
            return (self.input_len + has_out_len + 1, max(0, cur_max_new_token_len - has_out_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        else:
            assert False, "error state"
        return
    
    def get_decode_need_tokens(self):
        if self.req_status == ReqRunStatus.RUNNING:
            return 1
        else:
            assert False, "error state"
    
    def get_first_router_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE or self.req_status == ReqRunStatus.WAIT_APPEND_PREFILL:
            return self.input_len
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.input_len + len(self.output_ids)
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return 0
        else:
            assert False, "error state"

class CosyVoice2Req(Req):
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, semantic_len, speech_index, stream):
        super().__init__(request_id, prompt_ids, sample_params, semantic_len, speech_index)
        self.stream = stream
        self.bistream = False
        self.token_offset = 0
        return

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        """
        普通continues batch调度模式, 先prefill 后 decode 的估计方式 的实现
        """
        has_out_len = len(self.output_ids)
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            # 用当前输出长度的 1.1 倍作为预估输出长度的另一个参考量，用于更新估计的最大输出长度量
            # 后续会更新为更合理的统计条件概率估计方式 to do
            cur_max_new_token_len = min(self.max_output_len, max(int(1.1 * has_out_len), router_max_new_token_len))

        if self.req_status == ReqRunStatus.RUNNING:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (self.input_len + 1,  max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return (self.input_len + has_out_len + 1, max(0, cur_max_new_token_len - has_out_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        else:
            assert False, "error state"
        return

    def get_decode_need_tokens(self):
        if self.req_status == ReqRunStatus.RUNNING:
            return 1
        else:
            assert False, "error state"
    
    def get_first_router_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return self.input_len
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.input_len + len(self.output_ids)
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return 0
        else:
            assert False, "error state"

    

class NormalReq(Req):
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, semantic_len, bert_feature_ref_index, bert_feature_text_index):
        super().__init__(request_id, prompt_ids, sample_params, semantic_len, bert_feature_ref_index, bert_feature_text_index)
        return
    
    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        """
        普通continues batch调度模式, 先prefill 后 decode 的估计方式 的实现
        """
        has_out_len = len(self.output_ids)
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            # 用当前输出长度的 1.1 倍作为预估输出长度的另一个参考量，用于更新估计的最大输出长度量
            # 后续会更新为更合理的统计条件概率估计方式 to do
            cur_max_new_token_len = min(self.max_output_len, max(int(1.1 * has_out_len), router_max_new_token_len))

        if self.req_status == ReqRunStatus.RUNNING:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (self.input_len + 1,  max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return (self.input_len + has_out_len + 1, max(0, cur_max_new_token_len - has_out_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        else:
            assert False, "error state"
        return
    
    def get_decode_need_tokens(self):
        if self.req_status == ReqRunStatus.RUNNING:
            return 1
        else:
            assert False, "error state"
    
    def get_first_router_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return self.input_len
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.input_len + len(self.output_ids)
        elif self.req_status == ReqRunStatus.PAUSED_AND_KVKEEP:
            return 0
        else:
            assert False, "error state"



class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

        # 该参数只会在batch init， prefill， decode 后进行更新，并在剔除请求时减少
        # 在 batch rpc init 之后才会被填充正确的值，初始化为 None
        self.batch_decode_need_tokens = None
        self.batch_used_tokens = 0
        # init used tokens
        for req in self.reqs:
            self.batch_used_tokens += req.get_used_tokens()
        return

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def mark_and_get_finished_req_and_preupdate_status(self, eos_id, fill_token_id):
        unfinished_req_ids, finished_req_ids, append_prefill_req_ids = [], [], []
        for req in self.reqs:
            if len(req.output_ids) >= 1 and req.output_ids[-1] == eos_id:
                req.finish_status = FinishStatus.FINISHED_STOP
            elif len(req.output_ids) >= req.max_output_len:
                req.finish_status = FinishStatus.FINISHED_LENGTH
            elif req.bistream and req.output_ids[-1] == fill_token_id:
                req.output_ids.pop()
                req.req_status = ReqRunStatus.WAIT_FOR_TEXT

            if req.finish_status.is_finished():
                finished_req_ids.append(req.request_id)
                # 标记的时候，也同时更新一些这些请求被移除掉的更新量，有点dirty
                self.batch_used_tokens -= req.get_used_tokens()
                self.batch_decode_need_tokens -= req.get_decode_need_tokens()
            elif req.req_status == ReqRunStatus.WAIT_FOR_TEXT:
                append_prefill_req_ids.append(req.request_id)
            else:
                unfinished_req_ids.append(req.request_id)
    
        return unfinished_req_ids, finished_req_ids, append_prefill_req_ids
    
    def filter_out_finished_req(self, unfinished_req_ids, finished_req_ids, append_prefill_req_ids):
        if len(finished_req_ids) != 0 or len(append_prefill_req_ids) != 0:
            self.reqs = [self.id_to_reqs[req_id] for req_id in unfinished_req_ids]
            self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return
    
    def pop_req(self, req_id):
        self.reqs = [req for req in self.reqs if req.request_id != req_id]
        req = self.id_to_reqs[req_id]
        self.id_to_reqs.pop(req_id)
        self.batch_used_tokens -= req.get_used_tokens()
        self.batch_decode_need_tokens -= req.get_decode_need_tokens()
        return

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        self.batch_used_tokens += mini_batch.batch_used_tokens
        self.batch_decode_need_tokens += mini_batch.batch_decode_need_tokens
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                f"reqs={self.reqs}, ")
        
class AbortReq:
    def __init__(self, req_id, style):
        self.req_id = req_id
        self.style = style

class ReqError:
    def __init__(self, req_id, style, error:str):
        self.req_id = req_id
        self.style = style
        self.error = error
        
