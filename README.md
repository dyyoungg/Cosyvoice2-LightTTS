
- [模型文件介绍](#模型文件介绍)
  - [模型目录](#模型目录)
  - [models/config.json 内容格式](#modelsconfigjson-内容格式)
- [部署](#部署)
  - [本地部署](#本地部署)
  - [构建容器](#构建容器)
  - [拉取镜像](#拉取镜像)
  - [容器启动](#容器启动)
  - [CCI 容器启动](#cci-容器启动)

# 模型文件介绍
## 模型目录

models
```
├── chinese-hubert-base
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── chinese-roberta-wwm-ext-large
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
├── config.json
└── lora
    ├── bzt.ckpt
    ├── bzt.pth
    ├── bzt.txt
    ├── bzt.wav
    ├── mb.ckpt
    ├── mb.pth
    ├── mb.txt
    ├── mb.wav
    ├── silk_man.ckpt
    ├── silk_man.pth
    ├── silk_man.txt
    ├── silk_man.wav
    ├── tc.ckpt
    ├── tc.pth
    ├── tc.txt
    ├── tc.wav
    ├── wmy.ckpt
    ├── wmy.pth
    ├── wmy.txt
    └── wmy.wav
```
## models/config.json 内容格式
```json
{
    "lora_info": [
        {
            "style_name": "bzt",
            "gpt_model_path": "./lora/bzt.ckpt",
            "sovit_model_path": "./lora/bzt.pth",
            "refer_wav_path": "./lora/bzt.wav",
            "prompt_text_path": "./lora/bzt.txt"
        },
        {
            "style_name": "mb",
            "gpt_model_path": "./lora/mb.ckpt",
            "sovit_model_path": "./lora/mb.pth",
            "refer_wav_path": "./lora/mb.wav",
            "prompt_text_path": "./lora/mb.txt"
        },
        {
            "style_name": "silk_man",
            "gpt_model_path": "./lora/silk_man.ckpt",
            "sovit_model_path": "./lora/silk_man.pth",
            "refer_wav_path": "./lora/silk_man.wav",
            "prompt_text_path": "./lora/silk_man.txt"
        },
        {
            "style_name": "tc",
            "gpt_model_path": "./lora/tc.ckpt",
            "sovit_model_path": "./lora/tc.pth",
            "refer_wav_path": "./lora/tc.wav",
            "prompt_text_path": "./lora/tc.txt"
        },
        {
            "style_name": "wmy",
            "gpt_model_path": "./lora/wmy.ckpt",
            "sovit_model_path": "./lora/wmy.pth",
            "refer_wav_path": "./lora/wmy.wav",
            "prompt_text_path": "./lora/wmy.txt"
        }
    ],
    "tokenizer": {
        "path": "./chinese-roberta-wwm-ext-large"
    },
    "encoder": {
        "path": "./chinese-hubert-base"
    },
    "model_type": "tts",
    "torch_dtype": "float16",
    "sampling_rate": 16000
  }
```
# 部署

## 本地部署
在仓库根目录下的启动命令
先执行这条指令，可以大幅度的优化多进程操作一张显卡时的性能
```
  nvidia-cuda-mps-control -d 
  CUDA_VISIBLE_DEVICES=7 python -m lightllm.server.api_server --model_dir /nvme/wzj/voice/xiangfaliu/models     \
                                     --host 0.0.0.0                 \
                                     --port 8017                 \
                                     --bert_process_num 1 \
                                     --decode_process_num 1 \
                                     --max_total_token_num 10000 \
                                     --encode_paral_num 5 \
                                     --gpt_paral_num 10 \
                                     --decode_paral_num 1
```
## 构建容器
docker build -t lightllm-tts-sovits .

## 拉取镜像
```
docker pull registry.cn-sh-01.sensecore.cn/lm4science-ccr/sense_tts_server:v2.0.0
```
## 容器启动
```
docker run -v /nvme:/nvme -it -p 8081:8017 --gpus "device=0" --shm-size=4g lightllm-tts-sovits:latest --model_dir /nvme/wzj/voice/xiangfaliu/models     \
    --host 0.0.0.0                 \
    --port 8017                 \
    --bert_process_num 1 \
    --decode_process_num 1 \
    --max_total_token_num 10000 \
    --encode_paral_num 5 \
    --gpt_paral_num 10 \
    --decode_paral_num 1
```
## CCI 容器启动


```
容器镜像名称：sense_tts_server
版本：v2.0.0	
```
如果是在cci容器中运行，启动参数中最好加上 --health_monitor， 该参数会在内部主动监控整个服务的健康程度，出了问题会杀掉所有进程，然后cci容器就会重启。
```
cd lightllm
model_dir="/mnt/cache/zhangxingyan/gitlab/chat/server/lightllm/models_v2"
port=20052
python -m lightllm.server.api_server --model_dir model_dir  \
    --host 0.0.0.0 \
    --port $port            \
    --max_req_input_len 1024  \
    --max_req_total_len 2048 \
    --max_total_token_num 96000    \
    --tokenizer_mode slow \
    --trust_remote_code \
    --health_monitor
     

```