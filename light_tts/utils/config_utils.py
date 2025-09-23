import os
import json
from schema import Schema, And, Use, Optional, SchemaError
from light_tts.utils.path_utils import trans_relative_to_abs_path
from light_tts.utils.log_utils import init_logger
import sys

logger = init_logger(__name__)

def get_config_json(model_dir):
    return {
        "lora_info": [
            {
                "style_name": "CosyVoice2",
            }
        ],
    }

def check_config(config):
    # 定义配置的规则
    config_schema = Schema({
        "lora_info": [
            {
                "style_name": And(str, len),
                "gpt_model_path": And(str, len),
                "sovit_model_path": And(str, len),
                "refer_wav_path": And(str, len),
                "prompt_text_path": And(str, len),
                Optional("refer_wav_path_sovits"): And(str, len), # 空，单个文件路径，一个文件夹路径
                Optional("token_num"): Use(int) # 用于为热点lora 提供更多的token初始化，支持更好并发。
            }],

        "tokenizer": {
            "path": And(str, len)
        },
        "encoder": {
            "path": And(str, len)
        },
        "model_type": Optional("tts", default="tts"),
        "torch_dtype":  Optional("float16", default="float16"),
        "sampling_rate": And(Use(int), lambda n: 0 <= n <= 100000),
    })

    # 读取并解析配置文件
    try:
        # 校验配置文件内容
        config_schema.validate(config)
        logger.info("配置文件校验成功！")

    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误：{e}")
        sys.exit(1)
    except SchemaError as e:
        logger.error(f"配置文件校验失败：{e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"配置文件未找到：{e}")
        sys.exit(1)

    logger.info("lora style check success")
    return config

def get_style_gpt_path(weight_dir, style_name):
    total_config = get_config_json(weight_dir)
    for item in total_config["lora_info"]:
        if style_name == item["style_name"]:
            return item["gpt_model_path"]
    assert False, f"can not find {style_name} gpt weight path"

def get_style_config(weight_dir, style_name):
    total_config = get_config_json(weight_dir)
    for item in total_config["lora_info"]:
        if style_name == item["style_name"]:
            return item
    assert False, f"can not find {style_name} gpt weight path"

def get_style_index(weight_dir, style_name):
    total_config = get_config_json(weight_dir)
    for index, item in enumerate(total_config["lora_info"]):
        if style_name == item["style_name"]:
            return index
    assert False, f"can not find {style_name} gpt weight path"