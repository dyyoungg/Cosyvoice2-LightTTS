from .utils.text.symbols import *
import torch
from light_tts.utils.log_utils import init_logger
from light_tts.models.sovits_gpt.utils.text.cleaner import clean_text_inf
from transformers import AutoModelForMaskedLM
import LangSegment
from typing import List, Tuple
from dataclasses import dataclass, field

logger = init_logger(__name__)

@dataclass
class SplitPhones:
    norm_text: str = None
    phones: List[str] = None
    word2ph: list = None
    language: str = None
    input_ids: torch.Tensor = None
    bert_feature: torch.Tensor = None

@dataclass
class ReqTextSplitInfo:
    origin_prompt: str = None
    origin_language: str = None
    norm_text: str = None
    phones: List[str] = None
    bert_feature: torch.Tensor = None
    splitphones_list: List[SplitPhones] = field(default_factory=list)
    has_exception: str = None

    def get_infer_input_len(self):
        input_len = 0
        for info in self.splitphones_list:
            if info.input_ids is not None:
                input_len += len(info.input_ids)
        return input_len
    
    def encode_all_input_ids(self, tts_tokenizer):
        for info in self.splitphones_list:
            if info.input_ids is None:
                if info.language in ["zh", "all_zh"]:
                    info.input_ids = tts_tokenizer.encode(info.norm_text)
        return
    
    def verify_infer_input_len(self):
        input_len = 0
        for info in self.splitphones_list:
            if info.input_ids is not None:
                input_len += len(info.input_ids)
                if len(info.input_ids) > 512:
                    raise Exception(f"bert input is too long {len(info.input_ids)} > 512")
        if input_len > 1024:
            raise Exception(f"bert total input is too long {input_len} > 1024")
        return
class TTSTokenizer:

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.device = "cuda"
        self.dtype = torch.float16 if config["torch_dtype"] == "float16" else torch.float32

    def get_split_phones(self, text, language) -> ReqTextSplitInfo:
        if language in {"en","all_zh","all_ja"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = clean_text_inf(formattext, language)
            # print('phonse: ', phones)

            req_split_info = ReqTextSplitInfo(origin_prompt=text, 
                                origin_language=language, 
                                norm_text=norm_text, 
                                phones=phones)
            req_split_info.splitphones_list.append(
                SplitPhones(norm_text=norm_text,
                            phones=phones,
                            word2ph=word2ph,
                            language=language))
            req_split_info.encode_all_input_ids(self)
            return req_split_info

        elif language in {"zh", "ja","auto"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        langlist.append(language)
                    textlist.append(tmp["text"])
            req_split_info = ReqTextSplitInfo(origin_prompt=text, origin_language=language)
            phones_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
                req_split_info.splitphones_list.append(
                    SplitPhones(norm_text=norm_text, phones=phones, word2ph=word2ph, language=lang))
                phones_list.append(phones)
                norm_text_list.append(norm_text)

            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)
            req_split_info.phones = phones
            req_split_info.norm_text = norm_text
            req_split_info.encode_all_input_ids(self)
            logger.info("normal text: {}".format(norm_text))
            logger.info('phonse: {}'.format(phones))
        return req_split_info
    

    def encode(self, text):
        assert len(text) != 0
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].view(-1)

    def __getattr__(self, name):
        if name != 'encode':
            return getattr(self.tokenizer, name)
        return self.encode