import torch
import time
import uuid
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Dict, List, Optional
from ..sampling_params import SamplingParams
from ..io_struct import Req, CosyVoice2Req, Batch, CosyVoice2BistreamReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from light_tts.utils.infer_utils import calculate_time
from ..io_struct import AbortReq, ReqRunStatus, FinishStatus, ReqError
from .stats import Stats
from light_tts.utils.log_utils import init_logger
from light_tts.utils.config_utils import get_style_gpt_path, get_style_config
from light_tts.server.shm_tools.shm_objs import SharedSpeechManager
import multiprocessing as mp
from light_tts.utils.load_utils import load_yaml
from itertools import chain

logger = init_logger(__name__)


class RouterManager:

    def __init__(self, args, tts1_gpt_port, tts_decode_port, style_name, gpt_parall_lock:mp.Semaphore):
        self.args = args
        self.world_size = 1
        self.load_way = args.load_way
        self.style_name = style_name
        self.gpt_parall_lock = gpt_parall_lock
        self.has_lock = False
        self.parall_step_counter = 0
        self.parall_step_max_num = args.gpt_paral_step_num
        self.mode = args.mode
        
        self.running_batch: Batch = None
        self.new_mini_batch: Batch = None
        self.has_wait_tokens = 0
        self.max_wait_tokens = 5
        
        context = zmq.asyncio.Context(2)
        self.recv_from_tts1_encode = context.socket(zmq.PULL)
        self.recv_from_tts1_encode.bind(f"tcp://127.0.0.1:{tts1_gpt_port}")
        
        self.send_to_tts_decode = context.socket(zmq.PUSH)
        self.send_to_tts_decode.connect(f"tcp://127.0.0.1:{tts_decode_port}")

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
       
        # self.model_config = torch.load(get_style_gpt_path(args.model_dir, style_name), map_location="cpu")["config"]
        # self.style_config = get_style_config(args.model_dir, style_name)
        # 如果配置文件中存在token_num, 就用配置文件中的配置来初始化token的数量
        # if self.style_config.get("token_num", None) is not None:
            # self.args.max_total_token_num = self.style_config.get("token_num")
            # self.max_total_token_num = self.args.max_total_token_num
        # else:
            # self.max_total_token_num = args.max_total_token_num
        # self.eos_id = self.model_config["model"]["EOS"]
        # self.max_semantic_position = self.model_config["data"]["max_sec"] * 50
        # if self.args.sample_close is True:
        #     self.top_k = 1
        # else:
        #     self.top_k = self.model_config["inference"]["top_k"] 
        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)
        self.max_req_total_len = self.args.max_req_total_len
        self.max_total_token_num = args.max_total_token_num
        assert self.max_req_total_len <= self.max_total_token_num

        configs = load_yaml(args.model_dir)
        self.model_config = configs["llm"].llm.model.model.config
        self.speech_token_size = configs["llm"].speech_token_size
        self.mix_ratio = configs["llm"].mix_ratio
        self.decode_token_hop_len = 2 * configs["flow"].input_frame_rate # 2s or 1s ??
        self.flow_pre_lookahead_len = configs["flow"].pre_lookahead_len

        self.vocab_size = self.model_config.vocab_size
        self.eos_id = self.speech_token_size
        self.fill_token_id = self.eos_id + 2
        self.sos_eos = self.vocab_size
        self.task_id = self.vocab_size + 1
        self.max_semantic_position = self.model_config.max_position_embeddings
        del configs

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process()
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            kvargs = {
                "rank_id" : rank_id,
                "world_size" : self.world_size,
                "weight_dir" : self.args.model_dir,
                "load_way" : self.load_way,
                "max_total_token_num" : self.max_total_token_num,
                "mode" : self.mode,
                "max_req_num" : self.args.running_max_req_size + 8,
                "max_seq_length" : self.args.max_req_total_len + 8, # 留一点余量
                "style_name" : self.style_name,
                "cache_capacity": self.args.cache_capacity,
                "port": self.args.port,
                "speech_token_size": self.speech_token_size,
                "mix_ratio": self.mix_ratio,
                "disable_cudagraph": self.args.disable_cudagraph,
                "graph_max_batch_size": self.args.graph_max_batch_size,
                "graph_max_len_in_batch": self.args.graph_max_len_in_batch,
            }
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))

        await asyncio.gather(*init_model_ret)
        
        self.req_queue = ReqQueue(self.args)   
        return

    async def add_req(
        self,
        request_id : int, 
        prompt_ids : List[int],
        text_ids : List[int],
        audio_ids : List[int],
        speech_index : int,
        sampling_params : SamplingParams,
        stream : Optional[bool] = False,
        bistream : Optional[bool] = False,
        append : Optional[bool] = False,
        finish : Optional[bool] = False
    ):
        if append:
            
            if self.running_batch is not None:
                for req in self.running_batch.reqs:
                    if req.request_id == request_id and isinstance(req, CosyVoice2BistreamReq):
                        logger.debug(f"[append] to running_batch | req_id={request_id}")
                        await req.append_input(finish, text_ids)
                        return
            if self.new_mini_batch is not None:
                for req in self.new_mini_batch.reqs:
                    if req.request_id == request_id and isinstance(req, CosyVoice2BistreamReq):
                        logger.debug(f"[append] to new_mini_batch | req_id={request_id}")
                        await req.append_input(finish, text_ids)
                        return
            
            for req in self.req_queue.waiting_req_bistream_list:
                if req.request_id == request_id:
                    logger.debug(f"[append] to waiting_req_bistream_list | req_id={request_id}")
                    await req.append_input(finish, text_ids)
                    return
            for req in self.req_queue.waiting_req_list:
                if req.request_id == request_id and isinstance(req, CosyVoice2BistreamReq):
                    logger.debug(f"[append] to waiting_req_list | req_id={request_id}")
                    await req.append_input(finish, text_ids)
                    return
        if bistream:
            req = CosyVoice2BistreamReq(
                request_id, prompt_ids, text_ids, audio_ids, self.mix_ratio,
                sampling_params, speech_index, stream, self.sos_eos, self.task_id
            )
            self.req_queue.append_bistream(req)
            
            return
        semantic_len = len(audio_ids)
        all_ids = list(chain([self.sos_eos], prompt_ids, text_ids, [self.task_id], audio_ids))
        req = CosyVoice2Req(request_id, all_ids, sampling_params, semantic_len, speech_index, stream)
        self.req_queue.append(req)
        return
    
    def check_and_wait_to_has_lock(self):
        if self.has_lock == False:
            ans = self.gpt_parall_lock.acquire(block=True)
            assert ans == True
            self.has_lock = True
        self.parall_step_counter += 1
        return
    
    def check_and_release_lock(self):
        assert self.has_lock == True
        if self.has_lock == True and self.parall_step_counter >= self.parall_step_max_num:
            self.parall_step_counter = 0
            self.gpt_parall_lock.release()
            self.has_lock = False
        return
    
    def release_lock_when_all_finish(self):
        if self.has_lock == True:
            self.parall_step_counter = 0
            self.gpt_parall_lock.release()
            self.has_lock = False
        return

    async def abort(self, request_id, style):
        assert self.style_name == style
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.finish_status = FinishStatus.FINISHED_ABORT
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.finish_status = FinishStatus.FINISHED_ABORT
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        idle_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                idle_count = 1000
                if counter_count % 20 == 0:
                    total_used_tokens = self.running_batch.batch_used_tokens
                    token_ratio = total_used_tokens / self.max_total_token_num
                    logger.debug(
                        f"{self.style_name} current batch size: {len(self.running_batch.reqs)} " 
                        f"{self.style_name} token used ratio: {token_ratio} "
                        f"{self.style_name} gpt wait len {len(self.req_queue.waiting_req_list)}"
                    )
                    pass
                    self.stats_tool.print_stats()
             
                
            if self.running_batch is None:
                self.release_lock_when_all_finish()
                await asyncio.sleep(0.01)  # 10ms
                idle_count -= 1
                if idle_count == 0:
                    torch.cuda.empty_cache()

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = await self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                # logger.info(
                #     f"[schedule] create batch | batch_id={new_batch.batch_id} | reqs={[(r.request_id, r.req_status.name, r.text_cache) for r in new_batch.reqs]}"
                # )
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            self.new_mini_batch = await self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if self.new_mini_batch is not None:
                # logger.info(
                #     f"[schedule] create mini_batch | batch_id={self.new_mini_batch.batch_id} | reqs={[(r.request_id, r.req_status.name) for r in self.new_mini_batch.reqs]}"
                # )
                self.stats_tool.count_prompt_tokens(self.new_mini_batch)
                await self._prefill_batch(self.new_mini_batch)
                if not self.new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, self.new_mini_batch)
                    self.running_batch.merge(self.new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            assert False, "false state"

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        req_to_req_status = ans[0]
        
        self._update_init_status_to_batch(batch, req_to_req_status)
        return

    async def _prefill_batch(self, batch:Batch):
        # logger.debug(
        #     f"[prefill] enter | batch_id={batch.batch_id} | reqs={[(r.request_id, r.req_status.name, r.text_cache) for r in batch.reqs]}"
        # )
        self.check_and_wait_to_has_lock()
        t0 = time.time()
        await self._init_batch(batch)
        # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        req_to_out_status = ans[0]
        t1 = time.time()
        self.check_and_release_lock()

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids, append_prefill_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id, self.fill_token_id)
        # logger.info(
        #     f"[prefill] done | batch_id={batch.batch_id} | cost_ms={(t1 - t0)*1000:.2f} | "
        #     f"unfinished={len(unfinished_req_ids)} finished={len(finished_req_ids)} append_prefill={len(append_prefill_req_ids)} | "
        #     f"batch_used_tokens={batch.batch_used_tokens}"
        # )
        self._send_to_tts2_decodec_proc(batch, req_to_out_status)
        reqs = [batch.id_to_reqs[req_id] for req_id in append_prefill_req_ids]
        self.req_queue.waiting_req_bistream_list.extend(reqs)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids, append_prefill_req_ids)
        await self._handle_finish_and_append_req(batch, unfinished_req_ids, finished_req_ids, append_prefill_req_ids)
        return

    async def _decode_batch(self, batch:Batch):
        # logger.debug(
        #     f"[decode] enter | batch_id={batch.batch_id} | has_wait_tokens={self.has_wait_tokens} | "
        #     f"reqs={[(r.request_id, r.req_status.name, len(r.output_ids)) for r in batch.reqs]}"
        # )
        self.check_and_wait_to_has_lock()
        t0 = time.time()
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        req_to_out_status = ans[0]
        t1 = time.time()
        self.check_and_release_lock()

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids, append_prefill_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id, self.fill_token_id)
        self._send_to_tts2_decodec_proc(batch, req_to_out_status)
        reqs = [batch.id_to_reqs[req_id] for req_id in append_prefill_req_ids]
        self.req_queue.waiting_req_bistream_list.extend(reqs)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids, append_prefill_req_ids)
        await self._handle_finish_and_append_req(batch, unfinished_req_ids, finished_req_ids, append_prefill_req_ids)
        return

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List, append_prefill_req_ids):
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids, append_prefill_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch, append_prefill_req_ids):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id, append_prefill_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return
    
    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_and_append_req(self, batch: Batch, unfinished_req_ids, finished_req_ids, append_prefill_req_ids):
        if len(finished_req_ids) != 0 or len(append_prefill_req_ids) != 0:
            if batch.is_clear():
                await self._remove_batch(batch, append_prefill_req_ids)
            else:
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids, append_prefill_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return
    
    def _update_init_status_to_batch(self, batch: Batch, req_to_req_status):
        # 更新请求状态
        new_batch_used_tokens = 0
        new_batch_decode_need_tokens = 0 # 只有在 splitfuse 模式下有意义
        for req_id, (req_status, cur_kv_len) in req_to_req_status.items():
            r_obj = batch.id_to_reqs[req_id]
            r_obj.req_status = req_status
            r_obj.cur_kv_len = cur_kv_len
            new_batch_used_tokens += r_obj.get_used_tokens()
            new_batch_decode_need_tokens += r_obj.get_decode_need_tokens()
        
        batch.batch_used_tokens = new_batch_used_tokens
        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
    
    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status):
        new_batch_used_tokens = 0
        new_batch_decode_need_tokens = 0 # 只有在 splitfuse 模式下有意义
        for req_id, (req_status, cur_kv_len, new_token_id, new_gen_metadata) in req_to_out_status.items():
            req : Req = batch.id_to_reqs[req_id]
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            if new_token_id is not None:
                req.output_ids.append(new_token_id)
                # req.output_metadata_list.append(new_gen_metadata) # no use
            new_batch_used_tokens += req.get_used_tokens()
            new_batch_decode_need_tokens += req.get_decode_need_tokens()
        
        batch.batch_used_tokens = new_batch_used_tokens
        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
        
    def _can_decode(self, batch: Batch):
        return True
    
    def _send_to_tts2_decodec_proc(self, batch: Batch, req_to_out_status):
        for req_id, (req_status, cur_kv_len, new_token_id, new_gen_metadata) in req_to_out_status.items():
            req = batch.id_to_reqs[req_id]
            if req.finish_status.is_aborted():
                continue
            if req.stream == False:
                if req.finish_status.is_finished():
                    if req.output_ids[-1] == self.eos_id:
                        req.output_ids.pop()
                    logger.info(f"Send:    tts1_gpt_{self.style_name} | req_id: {req.request_id} | {len(req.output_ids)} token_ids bilv {len(req.output_ids) / max(1, req.input_len)}")
                    self.send_to_tts_decode.send_pyobj(
                        (req.output_ids, req.speech_index, req.request_id, 0, True, self.style_name)
                    )
                    cost_time = (time.time() - req.start_time) * 1000
                    logger.info(f"module tts_gpt req_id {req_id} cost_time {cost_time} ms")
            else:
                offset = self.decode_token_hop_len + self.flow_pre_lookahead_len + req.token_offset
                # logger.info(
                #     f"[stream] req_id={req.request_id} | output_ids={len(req.output_ids)} | token_offset={req.token_offset} | "
                #     f"decode_hop_len={self.decode_token_hop_len} | offset={offset}"
                # )
                if len(req.output_ids) >= offset:
                    if req.output_ids[-1] == self.eos_id:
                        req.output_ids.pop()
                    logger.info(f"Send:    tts1_gpt_{self.style_name} | req_id: {req.request_id} | {len(req.output_ids)} token_ids | offset {offset}")
                    self.send_to_tts_decode.send_pyobj(
                        (req.output_ids[:offset], req.speech_index, req.request_id, req.token_offset, False, self.style_name)
                    )
                    req.token_offset += self.decode_token_hop_len

                if req.finish_status.is_finished():
                    if req.output_ids[-1] == self.eos_id:
                        req.output_ids.pop()
                    logger.info(f"Send:    tts1_gpt_{self.style_name} | req_id: {req.request_id} | {len(req.output_ids)} token_ids bilv {len(req.output_ids) / max(1, req.input_len)}")
                    self.send_to_tts_decode.send_pyobj(
                        (req.output_ids, req.speech_index, req.request_id, req.token_offset, True, self.style_name)
                    )
                    cost_time = (time.time() - req.start_time) * 1000
                    logger.info(f"module tts_gpt req_id {req_id} cost_time {cost_time} ms")

        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_tts1_encode.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 9:
                request_id, style, prompt_text_ids, text_ids, speech_index, stream, bistream, append, finish = recv_req
                logger.info(f"Receive: tts1_gpt_{style.ljust(5)} | req_id: {request_id} | {len(prompt_text_ids)} {len(text_ids)} token_ids | speech_index: {speech_index} | finish {finish} | append:{append} ")
                if self.style_name != style:
                    logger.error(f"error style name should {self.style_name} but get {style}")
                    assert False, "error"
                if append:
                    
                    await self.add_req(request_id, prompt_text_ids, text_ids, audio_ids, speech_index, None, stream, bistream, append, finish)
                    continue

                audio_ids = self.shared_speech_manager.get_index_speech_token(speech_index).arr[0]
                audio_ids = [audio_id + self.vocab_size + 2 for audio_id in audio_ids]
                input_total_len = len(prompt_text_ids) + len(text_ids) + len(audio_ids) + 2
                # 为了能让短句的输出不那么长而影响调度做的操作。
                # 8192为ppl_fp16算子的上限
                max_new_tokens = min(len(text_ids) * 20, 8192 - input_total_len)
                if bistream:
                    min_new_tokens = 1
                else:
                    min_new_tokens = len(text_ids) * 2
                if max_new_tokens < 1:
                    logger.error(f"max_new_tokens is error,{self.max_semantic_position} - {len(audio_ids)}")
                    logger.error(f"give up req_id {request_id}, style {self.style_name}")
                    self.send_to_tts_decode.send_pyobj(ReqError(request_id, style, "prompt audio is too long"))
                elif (input_total_len + max_new_tokens > self.max_req_total_len) or (input_total_len == 0):
                    logger.error(f"input len {input_total_len} + max_new_tokens {max_new_tokens} > max_req_total_len {self.max_req_total_len}")
                    logger.error(f"give up req_id {request_id}, style {self.style_name}")
                    self.send_to_tts_decode.send_pyobj(ReqError(request_id, style, "text is too long to handle"))
                else:
                    sampling_params = SamplingParams(do_sample=True, top_k=25, top_p=0.8, win_size=10, tau_r=0.1, ignore_eos=False, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
                    
                    await self.add_req(request_id, prompt_text_ids, text_ids, audio_ids, speech_index, sampling_params, stream, bistream, append, finish)
                    logger.info("first text req appended")
            
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                assert self.style_name == abort_req.style
                await self.abort(request_id, recv_req.style)
                self.send_to_tts_decode.send_pyobj(abort_req)
            elif isinstance(recv_req, ReqError):
                # 直接转发
                self.send_to_tts_decode.send_pyobj(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        self.model_rpcs = None
        return

def start_tts1_gpt_process(args, tts1_gpt_port, tts_decode_port, style_name, gpt_parall_lock, pipe_writer):
    from light_tts.utils.graceful_utils import graceful_registry
    graceful_registry(f"tts1_gpt_{style_name}")
    try:
        router = RouterManager(
            args,
            tts1_gpt_port=tts1_gpt_port,
            tts_decode_port=tts_decode_port,
            style_name=style_name,
            gpt_parall_lock=gpt_parall_lock)
    
        asyncio.run(router.wait_to_model_ready())
    except Exception as e:
        import traceback
        import sys
        etype, evalue, tb = sys.exc_info()
        err_str = '\n'.join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
