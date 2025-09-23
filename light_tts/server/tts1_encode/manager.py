import light_tts.utils.infer_repair # import lightllm.utils.infer_repair 是为了hack 修改一些默认实现，不要删除
import torch
import zmq
import zmq.asyncio
import asyncio
import time
import uvloop
from ..io_struct import AbortReq, ReqError
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from light_tts.utils.config_utils import get_config_json
from light_tts.utils.log_utils import init_logger
from light_tts.server.shm_tools.shm_objs import SharedSpeechManager
from light_tts.utils.load_utils import load_yaml
from light_tts.server.tts1_encode.model_infer.frontend import CosyVoiceFrontEnd

logger = init_logger(__name__)

class TTS1EncodeManager:
    def __init__(
        self,
        args,
        tts1_gpt_ports,
        tts1_encode_port,
        index_id,
        encode_parall_lock,
    ):
        self.index_id = index_id
        self.encode_parall_lock = encode_parall_lock

        context = zmq.asyncio.Context(2)

        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{tts1_encode_port}")
        self.waiting_reqs = [] 
        self.model_cfg = get_config_json(args.model_dir)
        self.send_to_tts1_gpts = {}
        for i, lora_w in enumerate(self.model_cfg["lora_info"]):
            # context = zmq.asyncio.Context()
            if i % args.bert_process_num == self.index_id: # 通过id分配他需要处理的lora
                send_to_tts1_gpt_i = context.socket(zmq.PUSH)
                send_to_tts1_gpt_i.connect(f"tcp://127.0.0.1:{tts1_gpt_ports[i]}")
                self.send_to_tts1_gpts[lora_w["style_name"]] =  send_to_tts1_gpt_i
        # 只能有一个进程刷新内部的标记值
        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)

        self.world_size = 1
        self.trust_remote_code=args.trust_remote_code
        self.args = args

        configs = load_yaml(args.model_dir)
        self.configs = configs
        self.resample_rate = configs['sample_rate']
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(args.model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(args.model_dir),
                                          configs['allowed_special'])
        del configs

    
    async def abort(self, request_id, style):
        abort_req = AbortReq(req_id=request_id, style=style)
        self.waiting_reqs = [req for req in self.waiting_reqs if req[1] != request_id]
        self.send_to_tts1_gpts[style].send_pyobj(abort_req)
        return
    
    # def get_batch(self):
    #     if len(self.waiting_reqs) == 0:
    #         return []
    #     batch = []
    #     total_input_len = 0
    #     while len(self.waiting_reqs) != 0:
    #         request_dict, request_id, req_split_info = self.waiting_reqs.pop(0)
    #         # 限制调度的token跑的长度
    #         input_token_len = req_split_info.get_infer_input_len()
    #         if total_input_len + input_token_len <= 6 * 1024:
    #             total_input_len += input_token_len
    #             batch.append((request_dict, request_id, req_split_info))
    #         else:
    #             # 将请求退回，跳出循环
    #             self.waiting_reqs = [(request_dict, request_id, req_split_info),] + self.waiting_reqs
    #             break
            
    #         # 同时进行推理的请求数量限制
    #         if len(batch) >= 16:
    #             break

    #     return batch

    async def loop_for_fwd(self):
        module_name = "tts1_encoder"
        idle_count = 0
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
                idle_count -= 1
                if idle_count == 0:
                    torch.cuda.empty_cache()
            else:
                idle_count = 1000
                n = len(self.waiting_reqs)
                while n > 0:
                    request_dict, request_id, text_token, prompt_token = self.waiting_reqs.pop(0)
                    tts_model_name = request_dict["tts_model_name"]
                    speech_index = request_dict["speech_index"]
                    stream = request_dict["stream"]
                    bistream = request_dict.get("bistream", False)
                    append = request_dict.get("append", False)
                    finish = request_dict.get("finish", False)

                    n -= 1

                    if request_dict["need_extract_speech"]:
                        prompt_speech_16k = self.shared_speech_manager.get_index_data(speech_index)
                        if prompt_speech_16k is None:
                            raise RuntimeError(f"In encode, get_index_data {speech_index} not found")
                        prompt_speech_16k = torch.from_numpy(prompt_speech_16k.arr)
                        speech_token, speech_feat, embedding = self.frontend.frontend_zero_shot_speech(text_token, prompt_token, prompt_speech_16k, self.resample_rate)
                        self.shared_speech_manager.set_index_speech(request_dict["speech_index"], speech_token, speech_feat, embedding)
                    else:
                        if not self.shared_speech_manager.speech_data_ready(speech_index):
                            self.waiting_reqs.append((request_dict, request_id, text_token, prompt_token))
                            continue

                    logger.info(f"Send:    {module_name:<14} | req_id: {request_id} | prompt length {len(prompt_token)} | text length {len(text_token)} to tts_gpt | with speech | append:{append}")
                    self.send_to_tts1_gpts[tts_model_name].send_pyobj((request_id, tts_model_name, prompt_token, text_token, speech_index, stream, bistream, append, finish))
                    cost_time = (time.time() - request_dict['time'])*1000
                    logger.info(f"module {module_name} req_id {request_id} cost_time {cost_time} ms")
                    
                        
    async def loop_for_netio_req(self):
        module_name = "tts1_encoder"
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 2:
                request_dict, request_id = recv_req
                text = request_dict["text"]
                prompt_text = request_dict["prompt_text"]
                tts_model_name = request_dict["tts_model_name"]
                request_dict['time'] = time.time()
                logger.info(f"Receive: {tts_model_name:<14} | req_id: {request_id} | {len(text)} chars")
                
                try:
                    text_token = self.frontend.tokenizer.encode(text, allowed_special=self.frontend.allowed_special)
                    prompt_token = self.frontend.tokenizer.encode(prompt_text, allowed_special=self.frontend.allowed_special)
                    self.waiting_reqs.append((request_dict, request_id, text_token, prompt_token))
                    logger.info(f"encode module {list(self.send_to_tts1_gpts.keys())} waiting len {len(self.waiting_reqs)}")
                except Exception as e:
                    self.send_to_tts1_gpts[tts_model_name].send_pyobj(ReqError(request_id, tts_model_name, str(e))) 
                    logger.error(f"Send:    {module_name:<14} | req_id: {request_id} | tts_model_name: {tts_model_name} | text: {text} error: {str(e)}")
                
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                style = abort_req.style
                await self.abort(request_id, style)
            elif isinstance(recv_req, ReqError):
                self.send_to_tts1_gpts[recv_req.style].send_pyobj(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        self.model_rpcs = None
        return

def start_tts1_encode_process(args, tts1_gpt_ports, tts1_encode_port, index_id, encode_parall_lock, pipe_writer):
    from light_tts.utils.graceful_utils import graceful_registry
    graceful_registry("tts1_encode")
    try: 
        visualserver = TTS1EncodeManager(
            args,
            tts1_gpt_ports,
            tts1_encode_port,
            index_id,
            encode_parall_lock)
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        visualserver.clean_up()
        raise
    pipe_writer.send('init ok')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(visualserver.loop_for_fwd())
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
