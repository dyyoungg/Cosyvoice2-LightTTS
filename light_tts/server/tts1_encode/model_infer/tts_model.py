import torch
import numpy as np
import librosa
from torch.nn import functional as F
from light_tts.models.sovits_gpt.module.vq_model import SynthesizerTrn as Decodec
from light_tts.models.sovits_gpt.module.ssl_model import CNHubert
from light_tts.models.sovits_gpt.utils import *
from light_tts.models.sovits_gpt.utils.audio_utils import get_spepc
from light_tts.models.sovits_gpt.utils.text.cleaner import clean_text
from light_tts.models.sovits_gpt.utils.text import cleaned_text_to_sequence
from light_tts.models.sovits_gpt.tokenizer import TTSTokenizer, ReqTextSplitInfo
from transformers import AutoTokenizer
from light_tts.models.bert.model import BertTpPartBaseModel


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)

class TtsModel:
    def __init__(self, model_config):
        self.is_half = True
        self.device = "cuda"
        self.dict_language = {"中文": "zh","英文": "en","日文": "ja","ZH": "zh","EN": "en","JA": "ja","zh": "zh","en": "en","ja": "ja"}

        # total_config: ['lora_info', 'tokenizer', 'encoder', 'model_type', 'torch_dtype', 'sampling_rate']
        cnhubert_path = model_config["encoder"]["path"]
        sovit_model_path = model_config["sovit_model_path"]
        dict_s2 = torch.load(sovit_model_path, map_location="cpu")
        # dict_s2: ['weight', 'config', 'info']
        hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(hps)
        self.hps.model.semantic_frame_rate = "25hz"

        # vq_model + ssl_model  629.8MB->732.4->919.32
        self.vq_model = self.load_vq_model(dict_s2) # 过完gpt的pred_semantic->audio 输出
        self.ssl_model = self.load_ssl_model(cnhubert_path) # 输入音频编码
        return
    
    def load_vq_model(self, dict_s2):
        vq_model = Decodec(self.hps)
        if self.is_half:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        # 需要手动调用repair_weight_norm_weights
        # 修复 weight norm 权重的问题。
        vq_model.repair_weight_norm_weights()
        return vq_model

    def load_ssl_model(
            self,
            cnhubert_base_path,
    ):
        ssl_model = CNHubert(cnhubert_base_path)
        ssl_model.eval()

        if self.is_half:
            ssl_model = ssl_model.half().to(self.device)
        else:
            ssl_model = ssl_model.to(self.device)
        
        # 需要手动调用repair_weight_norm
        # 修复 weight norm 权重的问题。
        ssl_model.repair_weight_norm_weights()
        return ssl_model

    def get_refer_feature(self, refer_wav_path):
        zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3), dtype=np.float16 if self.is_half == True else np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(refer_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if (self.is_half == True):
                wav16k = wav16k.half().to(self.device)
                zero_wav_torch = zero_wav_torch.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
                zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]

        refer = get_spepc(self.hps, refer_wav_path)
        if (self.is_half == True):
            refer = refer.half().to(self.device)
        else:
            refer = refer.to(self.device)

        return prompt_semantic, refer, zero_wav

    def get_prompt_feature(self, prompt_text, tts_tokenizer: TTSTokenizer, bert_model:BertTpPartBaseModel, prompt_language="zh"):
        prompt_text = prompt_text.strip("\n")
        prompt_language = self.dict_language[prompt_language]
        req_split_info = tts_tokenizer.get_split_phones(prompt_text, prompt_language)
        bert_model.prefill_ReqTextSplitInfo_list([req_split_info,])
        return req_split_info.bert_feature.numpy(), req_split_info.phones
