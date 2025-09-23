import rpyc
import torch
import json
import os
import numpy as np
from light_tts.utils.infer_utils import set_random_seed
from light_tts.models.sovits_gpt.utils.audio_utils import get_spepc
from light_tts.utils.log_utils import init_logger
from typing import List
from ..obj import DecodeReq
from light_tts.utils.load_utils import load_yaml
from .model import CosyVoice2Model
from light_tts.server.shm_tools.shm_objs import SharedSpeechManager

logger = init_logger(__name__)

class TTS2DecodeModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        gpu_id = kvargs["gpu_id"]
        model_dir = kvargs["model_dir"]
        torch.cuda.set_device(gpu_id)
        
        configs = load_yaml(model_dir)
        self.model = CosyVoice2Model(configs['flow'], configs['hift'], fp16=True)
        self.model.load('{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        self.fp16 = True
        load_jit = False
        load_trt = True
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            capability = torch.cuda.get_device_capability(0)
            self.model.load_trt('{}/flow.decoder.estimator.{}.sm{}{}.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32', capability[0], capability[1]),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
            # self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu_.10.8.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
            #                     '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
            #                     self.fp16)
        
        set_random_seed(2147483647)
        self.shared_speech_manager = SharedSpeechManager(f"{kvargs['port']}_cosyvoice", kvargs["shared_cache_capacity"])
        del configs
    
    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def forward(self, batch:List[DecodeReq]):
        for req in batch:
            output_ids, speech_index, request_id, token_offset, finalize, style_name, start_time = req.req_tuple
            speech_token, speech_feat, spk_embedding = self.shared_speech_manager.get_index_speech(speech_index)
            speech_token, speech_feat, spk_embedding = speech_token.arr, speech_feat.arr, spk_embedding.arr
            tts_speech = self.model.token2wav(
                torch.tensor(output_ids, device="cuda").unsqueeze(0),
                torch.as_tensor(speech_token, device="cuda"),
                torch.as_tensor(speech_feat, device="cuda").unsqueeze(0),
                torch.as_tensor(spk_embedding, device="cuda"),
                request_id,
                token_offset,
                finalize=finalize
            )
            req.gen_audios = tts_speech.view(-1).cpu().numpy()
        return

    def exposed_decode(self, batch:List[DecodeReq]):
        return self.forward(batch)


class TTS2DecodeModelRpcClient:
    def __init__(self, model_rpc):
        self.model: TTS2DecodeModelRpcServer = model_rpc
        self._init_model = self.model.exposed_init_model
        self._decode = self.model.exposed_decode
        return

    async def init_model(self, kvargs):
        ans : rpyc.AsyncResult = self._init_model(kvargs)
        return

    async def decode(self, batch:List[DecodeReq]):
        return self._decode(batch)

async def start_model_process():
    return TTS2DecodeModelRpcClient(TTS2DecodeModelRpcServer())

