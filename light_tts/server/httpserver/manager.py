import zmq
import zmq.asyncio
import asyncio
import uvloop
import time

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from light_tts.utils.config_utils import get_config_json
from light_tts.utils.log_utils import init_logger
from ..io_struct import AbortReq, ReqError
from light_tts.server.shm_tools.shm_objs import SharedSpeechManager

logger = init_logger(__name__)


class HttpServerManager:
    def __init__(
        self,
        args,
        httpserver_port,
        tts1_encode_ports
    ):
        self.args = args
        self.total_config = get_config_json(args.model_dir)

        self.send_to_tts1_encode_dict = {}

        context = zmq.asyncio.Context(2)

        for index, lora_item in enumerate(self.total_config["lora_info"]):
            tts_encode_port = tts1_encode_ports[index % args.bert_process_num]
            style_name = lora_item["style_name"]
            self.send_to_tts1_encode_dict[style_name] = context.socket(zmq.PUSH)
            self.send_to_tts1_encode_dict[style_name].connect(f"tcp://127.0.0.1:{tts_encode_port}")

        self.recv_from_tts_decode = context.socket(zmq.PULL)
        self.recv_from_tts_decode.bind(f"tcp://127.0.0.1:{httpserver_port}")

        self.req_id_to_out_inf = {}
         
        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)
        return

    def alloc_speech_mem(self, speech_md5, prompt_wav):
        index, have_alloc = self.shared_speech_manager.alloc(speech_md5)
        if not have_alloc:
            self.shared_speech_manager.set_index_data(index, prompt_wav.shape, prompt_wav)
        return index, have_alloc
    
    async def append_bistream(self, request_dict, request_id):
        # 如果 req_status 还没有创建，将其加入等待队列
        if request_id not in self.req_id_to_out_inf:
            await asyncio.sleep(0.01)
            
        # req_status 已存在，直接发送
        style = request_dict["tts_model_name"]
        self.send_to_tts1_encode_dict[style].send_pyobj((request_dict, request_id))
        text = request_dict["text"]
        logger.info(f"appendbitstream send2tts_encode: {request_id}, text: {text} ")

    
    async def generate(self, request_dict, request_id, request=None):
        style = request_dict["tts_model_name"]
        req_status = ReqStatus(request_id)
        event = req_status.event
        self.req_id_to_out_inf[request_id] = req_status
        self.send_to_tts1_encode_dict[style].send_pyobj((request_dict, request_id))  # 1 代表的是 semantic_len, 最开始只有一个起始token
      

        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=1) # 时间信息
            except asyncio.TimeoutError:
                pass
                
            if request is not None and await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.abort(request_id, style)
                yield AbortReq(req_id=request_id, style=style)
                return

            # 说明请求已经被 aborted 掉了。
            if request_id not in self.req_id_to_out_inf.keys():
                yield ReqError(request_id, style, "not in dict")
                return

            async with req_status.lock:
                event.clear()
                ans = None
                if req_status.error is not None:
                    ans = req_status.error
                if len(req_status.audio_datas) > 0:
                    audio_data = req_status.audio_datas.pop(0)
                    ans = {"tts_speech": audio_data}
                
                if ans is None:
                    continue
                
                yield ans

                if req_status.finished:
                    try:
                        del self.req_id_to_out_inf[request_id]
                    except:
                        pass
                    return
        
    

    async def abort(self, request_id, style):
        abort_req = AbortReq(req_id=request_id, style=style)
        logger.info(f"Abort:   http_server    | req_id: {request_id}")
        self.send_to_tts1_encode_dict[style].send_pyobj(abort_req)

        try:
            del self.req_id_to_out_inf[request_id]
        except:
            pass

        return

    async def handle_loop(self):
        while True:
            recv_ans = await self.recv_from_tts_decode.recv_pyobj()
            try:
                if isinstance(recv_ans, (AbortReq, ReqError)):
                    req_status: ReqStatus = self.req_id_to_out_inf.get(recv_ans.req_id)
                    async with req_status.lock: 
                        req_status.error = recv_ans
                        req_status.finished = True
                        req_status.event.set()
                else:
                    sampling_rate, audio_data, req_id, finished = recv_ans
                    req_status: ReqStatus = self.req_id_to_out_inf.get(req_id)

                    async with req_status.lock: 
                        req_status.audio_datas.append(audio_data)
                        req_status.sampling_rate = sampling_rate
                        req_status.finished = finished
                        req_status.event.set()
            except:
                pass
        return

class ReqStatus:
    def __init__(self, req_id) -> None:
        self.req_id = req_id
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.sampling_rate = None 
        self.audio_datas = []
        self.finished = False
        self.error = None
        # 新增：标记第一个文本请求是否已经发送
        self.first_text_sent = False
        self.first_text_sent_event = asyncio.Event()

