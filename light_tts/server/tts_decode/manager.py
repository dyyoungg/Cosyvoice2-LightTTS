import light_tts.utils.infer_repair # import lightllm.utils.infer_repair 是为了hack 修改一些默认实现，不要删除
import torch
import torch.nn.functional as F
import time
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Union
import traceback
from .model_infer.model_rpc import start_model_process, TTS2DecodeModelRpcClient
from light_tts.utils.config_utils import get_config_json, get_style_config, get_style_index

from light_tts.utils.infer_utils import calculate_time, mark_start, mark_end
from light_tts.utils.log_utils import init_logger
from ..io_struct import AbortReq, ReqError
from .obj import DecodeReq
from typing import List

logger = init_logger(__name__)

class TTSDecodeManager:
    def __init__(
        self,
        args,
        tts_decode_port,
        httpserver_port,
        style_name,
        decode_parall_lock,
        decode_proc_index
    ):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.recv_from_tts2_gpt = context.socket(zmq.PULL)
        self.recv_from_tts2_gpt.bind(f"tcp://127.0.0.1:{tts_decode_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")
        self.req_id_to_out = {}
        self.style_name =  style_name
        self.waiting_reqs = []
        self.decode_parall_lock = decode_parall_lock
        self.decode_proc_index = decode_proc_index
        

    async def wait_to_model_ready(self):
        gpu_num = torch.cuda.device_count()
        self.gpu_id = self.decode_proc_index % gpu_num

        self.rpc_model = await start_model_process()

        kvargs = {
            "gpu_id": self.gpu_id,
            "model_dir": self.args.model_dir,
            "port": self.args.port,
            "shared_cache_capacity": self.args.cache_capacity
        }
        await self.rpc_model.init_model(kvargs)
        return
    
    async def infer_decodec_batch(self, batch):
        await self.rpc_model.decode(batch)
        return

    def get_batch(self):
        if len(self.waiting_reqs) == 0:
            return []
        batch = []
        while len(self.waiting_reqs) != 0:
            req : DecodeReq = self.waiting_reqs.pop(0)
            batch.append(req)
            # 同时进行推理的请求数量限制
            if len(batch) >= self.args.decode_max_batch_size:
                break
        
        return batch

    async def loop_for_fwd(self):
        module_name = "tts_decode"
        idle_count = 0
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
                idle_count -= 1
                if idle_count == 0:
                    torch.cuda.empty_cache()
            else:
                idle_count = 1000
                while len(self.waiting_reqs) > 0:
                    # batch: List[DecodeReq] = self.get_batch()
                    batch = [self.waiting_reqs.pop(0)]
                    try:
                        await self.infer_decodec_batch(batch)
                    except Exception as e:
                        logger.exception(str(e))
                        for req in batch:
                            req.has_exception = str(e)
                    
                    for req in batch:
                        output_ids, speech_index, request_id, token_offset, finalize, style_name, start_time = req.req_tuple
                        if req.has_exception is None:
                            logger.info(f"Send:    {module_name:<14} style_name {self.style_name} | req_id: {request_id}")
                            self.send_to_httpserver.send_pyobj((req.gen_sampling_rate, req.gen_audios, request_id, finalize))
                            cost_time = (time.time() - start_time) * 1000
                            logger.info(f"module tts_decode req_id {request_id} cost_time {cost_time} ms")
                        else:
                            self.send_to_httpserver.send_pyobj(ReqError(request_id, style_name, req.has_exception))

                    logger.debug(f"{self.style_name} current waiting queue in tts_decode: {len(self.waiting_reqs)}")
 

    async def abort(self, req_id, style):
        self.waiting_reqs = [req for req in self.waiting_reqs if req.req_tuple[2] != req_id]
        return
                    
    async def handle_loop(self):
        module_name = "tts_decode"
        while True:
            try:
                recv_obj = await self.recv_from_tts2_gpt.recv_pyobj() 
                if isinstance(recv_obj, AbortReq):
                    self.send_to_httpserver.send_pyobj(recv_obj)
                    await self.abort(recv_obj.req_id, recv_obj.style)
                elif isinstance(recv_obj, ReqError):
                    self.send_to_httpserver.send_pyobj(recv_obj)
                elif isinstance(recv_obj, tuple) and len(recv_obj) == 6:
                    output_ids, speech_index, request_id, token_offset, finalize, style_name = recv_obj
                    logger.info(f"Receive: {module_name:<14} style_name {self.style_name} | req_id: {request_id} | {len(output_ids)} token_ids")
                    recv_obj = (output_ids, speech_index, request_id, token_offset, finalize, style_name, time.time())

                    self.waiting_reqs.append(DecodeReq(req_tuple=recv_obj))
                else:
                    assert False, f"Error Req Inf {recv_obj}"

            except Exception as e:
                logger.error(f"detoken process has exception {str(e)}")
                traceback.print_exc()
                pass


def start_tts_decode_process(params_list, pipe_writer):
    from light_tts.utils.graceful_utils import graceful_registry
    graceful_registry("tts_decode")
    torch.backends.cudnn.enabled = True
    managers = []
    try:
        for params in params_list:
            args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index = params
        
            tts_decodec = TTSDecodeManager(
                args,
                tts_decode_port=tts_decode_port,
                httpserver_port=httpserver_port,
                style_name=style_name,
                decode_parall_lock=decode_parall_lock,
                decode_proc_index=decode_proc_index
            )
            asyncio.run(tts_decodec.wait_to_model_ready())
            managers.append(tts_decodec)
    except Exception as e:
        pipe_writer.send(str(e))
        raise

    pipe_writer.send('init ok')
    loop = asyncio.new_event_loop()
    for manager in managers:
        loop.create_task(manager.loop_for_fwd())
        loop.create_task(manager.handle_loop())

    loop.run_forever()
    return
