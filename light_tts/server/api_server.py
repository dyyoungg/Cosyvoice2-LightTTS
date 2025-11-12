# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import asyncio
import torch
import uvloop
import hashlib
import base64
import time
import copy
import os
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
from http import HTTPStatus
import multiprocessing as mp
import numpy as np
from fastapi import FastAPI, UploadFile, Form, File, BackgroundTasks, Request, WebSocketDisconnect, WebSocket
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import soundfile as sf
from io import BytesIO
import multiprocessing as mp
import logging
import sys
import os
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# 设置多个库的日志级别为 ERROR，减少调试输出
logging.getLogger("librosa").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("Pillow").setLevel(logging.ERROR)
# 设置根日志级别为 WARNING，过滤掉所有 DEBUG 信息
logging.getLogger().setLevel(logging.WARNING)
import librosa

from light_tts.server.httpserver.manager import HttpServerManager
from light_tts.server.tts1_encode.manager import start_tts1_encode_process
from light_tts.server.tts1_gpt.manager import start_tts1_gpt_process
from light_tts.server.tts_decode.manager import start_tts_decode_process
from light_tts.server.req_id_generator import ReqIDGenerator

from light_tts.utils.net_utils import alloc_can_use_network_port
from light_tts.utils.start_utils import start_submodule_processes
from light_tts.utils.config_utils import get_config_json, check_config, get_style_index
from light_tts.utils.param_utils import check_request
from light_tts.utils.health_utils import health_check
from light_tts.static_config import dict_language
from light_tts.server.io_struct import AbortReq, ReqError
from light_tts.utils.common import text_normalize
from light_tts.utils.text_norm_lib.text_normlization import TextNormalizer
from cosyvoice.utils.file_utils import load_wav

from light_tts.server.metrics import histogram_timer
from prometheus_client import Counter, Histogram, generate_latest
all_request_counter = Counter("lightllm_request_count", "The total number of requests")
failure_request_counter = Counter("lightllm_request_failure", "The total number of requests")
sucess_request_counter = Counter("lightllm_request_success", "The total number of requests")
request_latency_histogram = Histogram("lightllm_request_latency", "Request latency", ['route'])

from light_tts.server.health_monitor import start_health_check_process
from light_tts.utils.log_utils import init_logger
from light_tts.server.prompt_loader import prompt_config


# Placeholder logger; will be configured and re-initialized in main()/normal_start()
# logger = logging.getLogger(__name__)
logger = init_logger(__name__)
TIMEOUT_KEEP_ALIVE = 5  # seconds.

g_id_gen = ReqIDGenerator()
app = FastAPI()
server = uvicorn.Server(uvicorn.Config(app))

isFirst = True
lora_styles = []

# 添加锁来保护 prompt 相关全局变量的并发访问
prompt_lock = asyncio.Lock()

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


# 建议探测频率为60s的间隔，同时认为探测失败的超时时间为60s.连续3次探测失败则重启容器。
@app.get("/healthz")
@app.get("/health")
async def healthcheck():
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=404)

    if await health_check(httpserver_manager, g_id_gen, lora_styles):
        return JSONResponse({"message": "Ok"}, status_code=200)
    else:
        return JSONResponse({"message": "Error"}, status_code=404)
    
@app.get("/liveness")
def liveness():
    return {"status": "ok"}

@app.get("/readiness")
def readiness():
    return {"status": "ok"}

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'] * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

async def generate_data_stream(generate_objs):
    audio_buffer = bytearray()
    first_flag = True
    sampling_rate = 24000
    time_thresh = 0.8
    silence_threshold = -40

    for generator in generate_objs:
        async for i in generator:
            tts_audio = (i['tts_speech'] * (2 ** 15)).astype(np.int16).tobytes()
            audio_buffer.extend(tts_audio)
            buffer_duration = len(audio_buffer) / (2 * sampling_rate)  # 2 bytes per sample, sample_rate samples per second
            
            if buffer_duration >= time_thresh:
                
                audio = AudioSegment(
                    data=bytes(audio_buffer),
                    sample_width=2,
                    frame_rate=sampling_rate,
                    channels=1
                )
                
                if first_flag:
                   
                    start_trim = detect_leading_silence(audio, silence_threshold=silence_threshold)
                    
                    safe_start_trim = max(0, start_trim - 48)  # 2ms缓冲
                    safe_start_trim = min(safe_start_trim, len(audio) - 1)  # 不超过音频长度
                    trimmed_audio = audio[safe_start_trim:]
                    
                    resampled_audio_bytes = trimmed_audio.raw_data
                    first_flag = False
                    print(f'first chunk in loop!!!!! audio duration: {buffer_duration}')
                    
                    audio_buffer = bytearray(resampled_audio_bytes)
               
                yield bytes(audio_buffer)
                audio_buffer = bytearray()
            
    if len(audio_buffer) > 0: 
        if first_flag:  # 如果从未处理过静音（整个音频< time thresh秒）
            audio = AudioSegment(
                data=bytes(audio_buffer),
                sample_width=2,
                frame_rate=sampling_rate,
                channels=1
            )
            start_trim = detect_leading_silence(audio, silence_threshold=silence_threshold)
            safe_start_trim = max(0, start_trim - 48)
            trimmed_audio = audio[safe_start_trim:]
            audio_buffer = bytearray(trimmed_audio.raw_data)
            first_flag = False
        yield bytes(audio_buffer)
       

def calculate_md5(file: UploadFile) -> str:
    hash_md5 = hashlib.md5()
    while chunk := file.read(8192):  # 分块读取以支持大文件
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def send_wav(websocket: WebSocket, generator):
    try:
        async for result in generator:
            # 检查WebSocket连接状态
            if websocket.client_state.name != "CONNECTED":
                logger.info("WebSocket connection closed, stopping audio generation")
                break
                
            # 处理音频数据
            if isinstance(result, dict) and "tts_speech" in result:
                try:
                    audio_bytes = (result['tts_speech'] * (2 ** 15)).astype(np.int16).tobytes()
                    await websocket.send_bytes(audio_bytes)
                    logger.info(f"send tts speech:{len(audio_bytes)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to send audio data: {e}")
                    break
                continue

            # 处理错误情况
            if isinstance(result, AbortReq):
                try:
                    await websocket.send_json({
                        "error": f"Generation error: abort request",
                    })
                except Exception as e:
                    logger.warning(f"Failed to send abort error: {e}")
                break
            elif isinstance(result, ReqError):
                try:
                    await websocket.send_json({
                        "error": f"Generation error: {result.error}",
                    })
                except Exception as e:
                    logger.warning(f"Failed to send request error: {e}")
                break
            else:
                # 处理其他未知类型的结果
                logger.warning(f"Unknown result type from generator: {type(result)}")
                break
    except Exception as e:
        logger.warning(f"Error in send_wav: {e}")
    finally:
        try:
            await websocket.close()
        except Exception as e:
            pass
    
# 初始化全局变量，但不立即加载音频文件
prompt_cache = {}
prompt_md5_cache = {}
prompt_config_global = prompt_config

print("Global variables initialized")

@app.websocket("/inference_bistream")
async def inference_zero_shot_bistream(websocket: WebSocket):
    global isFirst
    global lora_styles, prompt_cache, prompt_md5_cache, prompt_config_global
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False
    # 接受 WebSocket 连接
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    # 接收初始化参数
    init_params = await websocket.receive_json()

    # 解析固定参数
    spk_id = init_params.get("spk_id")
    prompt_text = init_params.get("prompt_text")
    prompt_wav_b64 = init_params.get("prompt_wav")
    tts_model_name = init_params.get("tts_model_name", "default")

    prompt_speech_16k = None
    speech_md5 = None

    if spk_id is not None:
        if spk_id in prompt_config_global:
           
            styles = prompt_config_global[spk_id]
            config = styles["中性"] # 使用默认 style 
            prompt_wav_path = config["prompt_wav"]
            if prompt_wav_path not in prompt_cache:
                logger.info(f"Dynamically loading audio: {prompt_wav_path}")
                try:
                    speech, _ = librosa.load(prompt_wav_path, sr=16000)
                    prompt_cache[prompt_wav_path] = speech
                    with open(prompt_wav_path, "rb") as f:
                        md5 = calculate_md5(f)
                    prompt_md5_cache[prompt_wav_path] = md5
                    logger.info(f"Successfully loaded audio: {prompt_wav_path}, md5:{md5}")
                except Exception as e:
                    print(f"Error loading audio: {e}")
                    try:
                        await websocket.send_json({
                            "error": f"Failed to load audio file: {e}"
                        })
                        await websocket.close(code=400)
                    except Exception as send_error:
                        logger.warning(f"Failed to send error message: {send_error}")
                    return
            else:
                logger.info(f"{prompt_wav_path} already exist.Use prompt cache!")

            
            prompt_speech_16k = prompt_cache[prompt_wav_path].copy()
            prompt_speech_16k = torch.from_numpy(prompt_speech_16k).float().unsqueeze(0)
            prompt_text_val = config["prompt_text"]

            speech_md5 = prompt_md5_cache[prompt_wav_path]
            
            
        else:
            # spk_id存在但在配置中找不到
            try:
                await websocket.send_json({
                    "error": f"spk_id '{spk_id}' not found in prompt_config. Please use a valid spk_id or provide prompt_wav and prompt_text."
                })
                await websocket.close(code=400)
            except Exception as send_error:
                logger.warning(f"Failed to send error message: {send_error}")
            return
    else:
        if prompt_wav_b64 is None or prompt_text is None:
            try:
                await websocket.send_json({
                    "error": "If spk_id is not provided, prompt_wav and prompt_text are required."
                })
                await websocket.close(code=400)
            except Exception as send_error:
                logger.warning(f"Failed to send error message: {send_error}")
            return

        try:
            wav_bytes = base64.b64decode(prompt_wav_b64)
            wav_bytes_io = BytesIO(wav_bytes)
            prompt_speech_16k = load_wav(wav_bytes_io, 16000)
            wav_bytes_io.seek(0)
            speech_md5 = calculate_md5(wav_bytes_io)
            prompt_text_val = prompt_text

        except Exception as e:
            try:
                await websocket.send_json({
                    "error": f"Invalid prompt_wav: {e}"
                })
                await websocket.close(code=400)
            except Exception as send_error:
                logger.warning(f"Failed to send error message: {send_error}")
            return
    
    # # 处理音频文件（假设客户端发送 base64 编码的音频）
    # prompt_wav_data = await websocket.receive_bytes()
    # wav_bytes_io = BytesIO(prompt_wav_data)
    # prompt_speech_16k = load_wav(wav_bytes_io, 16000)
    # wav_bytes_io.seek(0)
    # speech_md5 = calculate_md5(wav_bytes_io)
    
    if tts_model_name == "default":
        tts_model_name = lora_styles[0]
    
    speech_index, have_alloc = httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)
    request_id = g_id_gen.generate_id()
    first_text = True
    prompt_text = text_normalize(prompt_text_val, split=False)
    process_task = None
    # print("speech index", speech_index, have_alloc, request_id)
    zh_normalizer = TextNormalizer()
    try:
        while True:
            input_data = await websocket.receive_json()
            tts_text = input_data.get("tts_text", "")
            if not input_data.get("finish", False):
                tts_text = zh_normalizer.normalize(tts_text)
            print("receive text", tts_text)
            cur_req_dict = {
                "text": tts_text,
                "prompt_text": prompt_text,
                "tts_model_name": tts_model_name,
                "speech_md5": speech_md5,
                "need_extract_speech": first_text and not have_alloc,
                "stream": True,
                "speech_index": speech_index,
                "bistream": True,
                "append": not first_text
            }
            if input_data.get("finish", False):
                cur_req_dict["finish"] = True
                cur_req_dict["append"] = True
                await httpserver_manager.append_bistream(cur_req_dict, request_id)
                break
            elif first_text:
                generator = httpserver_manager.generate(cur_req_dict, request_id)
                process_task = asyncio.create_task(send_wav(websocket, generator))
                first_text = False
            else:
                await httpserver_manager.append_bistream(cur_req_dict, request_id)
        await process_task
    except WebSocketDisconnect:
        # 处理客户端断开连接
        logger.info("WebSocket client disconnected")
        await httpserver_manager.abort(request_id, tts_model_name)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            # 检查连接状态再发送错误消息
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "error": f"Server error: {str(e)}"
                })
        except Exception as send_error:
            logger.warning(f"Failed to send error message: {send_error}")
    finally:
        try:
            await websocket.close()
        except Exception as close_error:
            pass
        if process_task is not None:
            process_task.cancel()


@histogram_timer(request_latency_histogram)
@app.post("/inference_zero_shot")
async def inference_zero_shot(
        request: Request,
        tts_text: str = Form(),
        prompt_text: str = Form(None),
        prompt_wav: UploadFile = File(None),
        spk_id: str = Form(None),
        stream: bool = Form(default=False),
        tts_model_name: str = Form(default="default"),
    ):
    all_request_counter.inc()
    global isFirst
    global lora_styles
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    if tts_model_name == "default":
        tts_model_name = lora_styles[0]
    
    if spk_id is not None:
        if spk_id in prompt_config_global:

            styles = prompt_config_global[spk_id]
            config = styles["中性"] # 使用默认 style 
            prompt_wav_path = config["prompt_wav"]
            if prompt_wav_path not in prompt_cache:
                logger.info(f"Dynamically loading audio: {prompt_wav_path}")
                try:
                    speech, _ = librosa.load(prompt_wav_path, sr=16000)
                    prompt_cache[prompt_wav_path] = speech
                    with open(prompt_wav_path, "rb") as f:
                        md5 = calculate_md5(f)
                    prompt_md5_cache[prompt_wav_path] = md5
                    logger.info(f"Successfully loaded audio: {prompt_wav_path}, md5:{md5}")
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load audio file: {e}"}
                    )
            else:
                logger.info(f"{prompt_wav_path} already exist.Use prompt cache!")

            
            prompt_speech_16k = prompt_cache[prompt_wav_path].copy()
            prompt_speech_16k = torch.from_numpy(prompt_speech_16k).float().unsqueeze(0)
            prompt_text_val = config["prompt_text"]

            speech_md5 = prompt_md5_cache[prompt_wav_path]
            
        else:
            # spk_id存在但在配置中找不到
           return JSONResponse(
                status_code=400,
                content={"error": f"spk_id '{spk_id}' not found in prompt_config. Please use a valid spk_id or provide prompt_wav and prompt_text."}
            )
    else:
        if prompt_wav is None or prompt_text is None:
            return JSONResponse(
                status_code=400,
                content={"error": "If spk_id is not provided, prompt_wav and prompt_text are required."}
            )
            
        try:
            prompt_speech_16k = load_wav(prompt_wav.file, 16000)
            prompt_wav.file.seek(0)
            speech_md5 = calculate_md5(prompt_wav.file)
            prompt_text_val = prompt_text

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid prompt_wav: {e}"}
            )
    
    prompt_text = text_normalize(prompt_text_val, split=False)
    tts_texts = text_normalize(tts_text, split=True)

    generate_objs = []
    need_extract_speech=True
    speech_index, have_alloc = httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)

    for text in tts_texts:
        cur_req_dict = {
            "text": text,
            "prompt_text": prompt_text,
            "tts_model_name": tts_model_name,
            "speech_md5": speech_md5,
            "need_extract_speech": need_extract_speech and not have_alloc,
            "stream": stream,
            "speech_index": speech_index
        }
        need_extract_speech = False
        request_id = g_id_gen.generate_id()
        results_generator = httpserver_manager.generate(
            cur_req_dict, request_id, request=request
        )
        generate_objs.append(results_generator)

    if stream:
        try:
            return StreamingResponse(generate_data_stream(generate_objs))
        except Exception as e:
            logger.error("An error occurred: %s", str(e), exc_info=True)
            return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))
    else:
        ans_objs = []
        for generator in generate_objs:
            async for result in generator:
                ans_objs.append(result)

        # 如果有错误返回就直接返回错误
        for e in ans_objs:
            if isinstance(e, AbortReq):
                failure_request_counter.inc()
                return create_error_response(HTTPStatus.BAD_REQUEST, "req aborted error")
            if isinstance(e, ReqError):
                failure_request_counter.inc()
                return create_error_response(HTTPStatus.BAD_REQUEST, f"req error:{e.error}")

        return StreamingResponse(generate_data(ans_objs))    

@app.post("/query_tts_model")
@app.get("/query_tts_model")
async def show_available_styles(request: Request) -> Response:
    data = {"tts_models": lora_styles}
    json_data = json.dumps(data)
    return Response(content=json_data, media_type="application/json")

@app.get("/metrics")
async def metrics() -> Response:
    metrics_data = generate_latest()
    response = Response(metrics_data)
    response.mimetype = 'text/plain'
    return response

@app.post("/add_spk")
async def add_spk(
    spk_id: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    style: str = Form("中性")
):
    global prompt_cache, prompt_md5_cache, prompt_config_global

    prompt_wav.file.seek(0)
    md5 = calculate_md5(prompt_wav.file)
    wav_path = f"/dev/shm/{md5}.wav"

    async with prompt_lock:
        # 检查是否已存在
        if spk_id in prompt_config_global and style in prompt_config_global[spk_id]:
            return {"success": False, "msg": f"spk_id {spk_id} with style {style} already exists"}
        
        if os.path.exists(wav_path):
            # 已有文件，无需重复写入
            pass
        else:
            prompt_wav.file.seek(0)
            with open(wav_path, "wb") as f:
                f.write(await prompt_wav.read())

        try:
            speech, _ = librosa.load(wav_path, sr=16000)
        except Exception as e:
            return {"success": False, "msg": f"Invalid prompt_wav: {e}"}

        
        prompt_cache[wav_path] = speech
        prompt_md5_cache[wav_path] = md5

        if spk_id not in prompt_config_global:
            prompt_config_global[spk_id] = {}
        prompt_config_global[spk_id][style] = {"prompt_wav": wav_path, "prompt_text": prompt_text}
        
        print(f"Successfully added spk_id: {spk_id}, style: {style}, wav_path: {wav_path}")
        return {"success": True, "msg": "added", "md5": md5}


@app.get("/spk_list")
async def spk_list():
    result = []
    # 使用锁保护并发读取
    async with prompt_lock:
        for spk_id, styles in prompt_config_global.items():
            for style, info in styles.items():
                if isinstance(info, dict):
                    prompt_text = info.get("prompt_text", "")
                    wav_path = info.get("prompt_wav", "")
                else:
                    prompt_text = ""
                    wav_path = info
               
                wav_b64 = ""
                if wav_path and os.path.exists(wav_path):
                    with open(wav_path, "rb") as f:
                        wav_b64 = base64.b64encode(f.read()).decode("utf-8")
                result.append({
                    "spk_id": spk_id,
                    "style": style,
                    "prompt_text": prompt_text,
                    "prompt_wav": wav_b64
                })
    return result



@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    await asyncio.sleep(3)
    logger.info("Graceful shutdown completed.")

    # 杀掉所有子进程
    import psutil
    import signal
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)

    server.should_exit = True
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=["tts", "llm"], default="tts", help="选择要启动的服务类型")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--model_dir", type=str, default=None,
                        help="the model weight dir path, the app will load config, weights and tokenizer from this dir")
    parser.add_argument("--tokenizer_mode", type=str, default="slow",
                        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test, 
                        when you want to get best performance, try auto mode""")
    parser.add_argument("--load_way", type=str, default="HF",
                        help="the way of loading model weights, the default is HF(Huggingface format), llama also supports DS(Deepspeed)")
    parser.add_argument("--max_total_token_num", type=int, default=6000,
                        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)")
    parser.add_argument("--batch_max_tokens", type=int, default=None,
                        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM")
    parser.add_argument("--running_max_req_size", type=int, default=100,
                        help="the max size for forward requests in the same time")
    parser.add_argument("--max_req_input_len", type=int, default=2048,
                        help="the max value for req input tokens num")
    parser.add_argument("--max_req_total_len", type=int, default=32 * 1024,
                        help="the max value for req_input_len + req_output_len")
    parser.add_argument("--bert_process_num", type=int, default=1)
    parser.add_argument("--decode_process_num", type=int, default=1)
    parser.add_argument("--encode_paral_num", type=int, default=100)
    parser.add_argument("--gpt_paral_num", type=int, default=100)
    parser.add_argument("--gpt_paral_step_num", type=int, default=200)
    parser.add_argument("--decode_paral_num", type=int, default=1)
    parser.add_argument("--decode_max_batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default=["triton_flashdecoding"], nargs='+',
                        help="""Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding 
                        | triton_gqa_attention | triton_gqa_flashdecoding] 
                        [triton_int8weight | triton_int4weight | lmdeploy_int4weight | ppl_int4weight], 
                        triton_flashdecoding mode is for long context, current support llama llama2 qwen;
                        triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
                        triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
                        ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
                        ppl_fp16 mode use ppl fast fp16 decode attention kernel;
                        triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode use int8 and int4 to store weights;
                        you need to read source code to make sure the supported detail mode for all models""")
    parser.add_argument("--trust_remote_code", action='store_true',
                        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.")
    parser.add_argument("--disable_log_stats", action='store_true',
                        help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10,
                        help="log stats interval in second.")
    parser.add_argument("--log_path_or_dir", type=str, default=None,
                        help="Directory or .log file path for LightLLM logs")
    
    parser.add_argument("--router_token_ratio", type=float, default=0.0,
                        help="token ratio to control router dispatch")
    parser.add_argument("--router_max_new_token_len", type=int, default=1024,
                        help="the request max new token len for router")
    parser.add_argument("--cache_capacity", type=int, default=200,
                    help="cache server capacity for multimodal resources")
    parser.add_argument("--cache_reserved_ratio", type=float, default=0.5,
                    help="cache server reserved capacity ratio after clear")
    parser.add_argument("--sample_close",  action='store_true',
                    help="close sample function for tts_gpt")
    parser.add_argument("--health_monitor",  action='store_true',
                        help="health check time interval")
    parser.add_argument("--disable_cudagraph", action="store_true", help="Disable the cudagraph of the decoding stage")
    parser.add_argument(
        "--graph_max_batch_size",
        type=int,
        default=16,
        help="""Maximum batch size that can be captured by the cuda graph for decodign stage.
                The default value is 8. It will turn into eagar mode if encounters a larger value.""",
    )
    parser.add_argument(
        "--graph_max_len_in_batch",
        type=int,
        default=8192,
        help="""Maximum sequence length that can be captured by the cuda graph for decodign stage.
                The default value is 8192. It will turn into eagar mode if encounters a larger value. """,
    )

    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()

    # Configure logging as early as possible
    # Recreate module logger to attach shared handlers
    # global logger
    # logger = init_logger(__name__)

    assert args.max_req_input_len < args.max_req_total_len
    assert args.max_req_total_len <= args.max_total_token_num
    
    # 普通模式下
    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert (
            args.batch_max_tokens >= args.max_req_total_len
        ), "batch_max_tokens must >= max_req_total_len"

    all_config = get_config_json(args.model_dir)
    num_loras = len(all_config["lora_info"])
    for lora_w in all_config["lora_info"]:
        lora_styles.append(lora_w['style_name'])
    
    assert args.bert_process_num <= num_loras
    assert args.decode_process_num <= num_loras

    can_use_ports = alloc_can_use_network_port(
        num=num_loras * 2 + args.bert_process_num + 100, used_nccl_port=None
    )

    httpserver_port = can_use_ports[0]
    del can_use_ports[0]
    tts1_encode_ports = can_use_ports[0:args.bert_process_num]
    del can_use_ports[0:args.bert_process_num]
    
    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args,
        httpserver_port=httpserver_port,
        tts1_encode_ports=tts1_encode_ports)
    
    tts1_gpt_ports = can_use_ports[0 : num_loras]
    del can_use_ports[0 : num_loras]

    
    # 第一个实列需要先初始化，解决一些同步的问题
    funcs = []
    start_args = []
    encode_parall_lock = mp.Semaphore(args.encode_paral_num)
    for index_id in range(args.bert_process_num):
        funcs.append(start_tts1_encode_process)
        start_args.append((args, tts1_gpt_ports, tts1_encode_ports[index_id], index_id, encode_parall_lock))
    start_submodule_processes(start_funcs=funcs[0:1], start_args=start_args[0:1])
    if len(start_args) > 1:
        start_submodule_processes(start_funcs=funcs[1:], start_args=start_args[1:])
    
    tts_decode_ports = can_use_ports[0 : num_loras]
    del can_use_ports[0 : num_loras]

    funcs = []
    start_args = []
    gpt_parall_lock = mp.Semaphore(args.gpt_paral_num)
    for style_name, tts1_gpt_port, tts_decode_port in zip([item["style_name"] for item in all_config["lora_info"]], tts1_gpt_ports, tts_decode_ports): 
        funcs.append(start_tts1_gpt_process)
        start_args.append((args, tts1_gpt_port, tts_decode_port, style_name, gpt_parall_lock))
    start_submodule_processes(start_funcs=funcs, start_args=start_args)


    decode_parall_lock = mp.Semaphore(args.decode_paral_num)
    funcs = []
    start_args = []
    for decode_proc_index in range(args.decode_process_num):
        tmp_args = []
        for style_name, tts_decode_port in zip([item["style_name"] for item in all_config["lora_info"]], tts_decode_ports): 
            if (get_style_index(args.model_dir, style_name) % args.decode_process_num) == decode_proc_index:
                tmp_args.append((args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index))
        funcs.append(start_tts_decode_process)
        start_args.append((tmp_args,))
    start_submodule_processes(start_funcs=funcs, start_args=start_args)

    if args.health_monitor:
        start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    server.install_signal_handlers()
    logger.info(f"start args {args}")
    logger.info(f"start args {args}")
    logger.debug('#'*100)
    time.sleep(10)
    logger.debug(f'Initial complete')
    time.sleep(10)
    logger.debug('#'*100)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )


def normal_start(args):
    # Configure logging as early as possible in this start path too
    global logger
    logger = init_logger(__name__)
    assert args.max_req_input_len < args.max_req_total_len
    assert args.max_req_total_len <= args.max_total_token_num
    
    # 普通模式下
    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert (
            args.batch_max_tokens >= args.max_req_total_len
        ), "batch_max_tokens must >= max_req_total_len"

    all_config = get_config_json(args.model_dir)
    num_loras = len(all_config["lora_info"])
    for lora_w in all_config["lora_info"]:
        lora_styles.append(lora_w['style_name'])
    
    assert args.bert_process_num <= num_loras
    assert args.decode_process_num <= num_loras

    can_use_ports = alloc_can_use_network_port(
        num=num_loras * 2 + args.bert_process_num + 100, used_nccl_port=None
    )

    httpserver_port = can_use_ports[0]
    del can_use_ports[0]
    tts1_encode_ports = can_use_ports[0:args.bert_process_num]
    del can_use_ports[0:args.bert_process_num]
    
    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args,
        httpserver_port=httpserver_port,
        tts1_encode_ports=tts1_encode_ports)
    
    tts1_gpt_ports = can_use_ports[0 : num_loras]
    del can_use_ports[0 : num_loras]

    
    # 第一个实列需要先初始化，解决一些同步的问题
    funcs = []
    start_args = []
    encode_parall_lock = mp.Semaphore(args.encode_paral_num)
    for index_id in range(args.bert_process_num):
        funcs.append(start_tts1_encode_process)
        start_args.append((args, tts1_gpt_ports, tts1_encode_ports[index_id], index_id, encode_parall_lock))
    start_submodule_processes(start_funcs=funcs[0:1], start_args=start_args[0:1])
    if len(start_args) > 1:
        start_submodule_processes(start_funcs=funcs[1:], start_args=start_args[1:])
    
    tts_decode_ports = can_use_ports[0 : num_loras]
    del can_use_ports[0 : num_loras]

    funcs = []
    start_args = []
    gpt_parall_lock = mp.Semaphore(args.gpt_paral_num)
    for style_name, tts1_gpt_port, tts_decode_port in zip([item["style_name"] for item in all_config["lora_info"]], tts1_gpt_ports, tts_decode_ports): 
        funcs.append(start_tts1_gpt_process)
        start_args.append((args, tts1_gpt_port, tts_decode_port, style_name, gpt_parall_lock))
    start_submodule_processes(start_funcs=funcs, start_args=start_args)


    decode_parall_lock = mp.Semaphore(args.decode_paral_num)
    funcs = []
    start_args = []
    for decode_proc_index in range(args.decode_process_num):
        tmp_args = []
        for style_name, tts_decode_port in zip([item["style_name"] for item in all_config["lora_info"]], tts_decode_ports): 
            if (get_style_index(args.model_dir, style_name) % args.decode_process_num) == decode_proc_index:
                tmp_args.append((args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index))
        funcs.append(start_tts_decode_process)
        start_args.append((tmp_args,))
    start_submodule_processes(start_funcs=funcs, start_args=start_args)

    if args.health_monitor:
        start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    server.install_signal_handlers()
    logger.info(f"start args {args}")
    logger.info(f"start args {args}")
    logger.debug('#'*100)
    time.sleep(10)
    logger.debug(f'Initial complete')
    time.sleep(10)
    logger.debug('#'*100)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True), # this code will not be ok for settings to fork to subprocess
    main()
