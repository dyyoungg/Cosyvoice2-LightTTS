# LightLLM CosyVoice

> **⚠️ Internal Modified Version**  
> This is an internal customized version based on [ModelTC/LightTTS](https://github.com/ModelTC/LightTTS).  
> Original repository: https://github.com/ModelTC/LightTTS

A high-performance text-to-speech (TTS) service built on LightLLM framework, supporting real-time voice synthesis with customizable speaker voices and streaming audio generation.

## Features

### 🎯 Core Capabilities
- **Real-time TTS**: WebSocket-based streaming audio generation
- **Zero-shot Voice Cloning**: Upload reference audio to clone any voice
- **Multi-speaker Support**: Pre-configured speaker voices
- **Streaming Audio**: Real-time audio streaming via WebSocket connections
- **RESTful API**: Standard HTTP endpoints for TTS requests

### 🚀 Performance Features
- **High Throughput**: Optimized for concurrent request handling
- **Memory Efficient**: Smart caching of audio prompts and models
- **GPU Acceleration**: CUDA-optimized inference with configurable batch sizes


## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory recommended
- Linux/Ubuntu 20.04+ (tested on Ubuntu)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd lightllm-cosyvoice
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights**
```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```
```
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```


## Quick Start

### Basic Launch
```bash
# Using the provided launcher script
bash ./launcher.sh

# Or run directly
python -m light_tts.server.api_server \
    --model_dir /path/to/CosyVoice2-0.5B \
    --host 0.0.0.0 \
    --port 8088
```

### Advanced Configuration
```bash
export LIGHTLLM_LOG_DIR="path to log"
python -m light_tts.server.api_server \
    --model_dir /path/to/CosyVoice2-0.5B \
    --host 0.0.0.0 \
    --port 8089 \
    --bert_process_num 1 \
    --decode_process_num 1 \
    --max_total_token_num 60000 \
    --encode_paral_num 50 \
    --gpt_paral_num 50 \
    --decode_paral_num 1 \
    --mode triton_flashdecoding \
```

## API Usage

LightLLM CosyVoice provides two main interfaces for text-to-speech:

### 1. HTTP REST API (`inference_zero_shot`)
**Use for**: Simple TTS requests, batch processing, integration with web applications

- **Endpoint**: `POST /inference_zero_shot`
- **Input**: Text + Speaker ID (or custom audio upload)
- **Output**: Complete audio file
- **Features**: Non-streaming and streaming modes

### 2. WebSocket API (`inference_bistream`)
**Use for**: Real-time streaming, interactive conversations, low-latency applications

- **Endpoint**: `WebSocket /inference_bistream`
- **Input**: Streaming text chunks
- **Output**: Real-time audio streaming
- **Features**: Continuous conversation, append mode

**HTTP REST API (inference_zero_shot)**
```python
import requests

# Basic TTS request
response = requests.post("http://localhost:8088/inference_zero_shot", data={
    'tts_text': '你好，这是测试',
    'spk_id': 'tangwei',
    'stream': False
})

# Save audio
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

**WebSocket API (inference_bistream)**
```python
import websockets
import asyncio
import json
from websockets.exceptions import ConnectionClosed
import soundfile as sf
import numpy as np
import base64


def encode_wav_to_base64(wav_path):
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    return base64.b64encode(wav_bytes).decode("utf-8")
    
async def sender(ws):
    prompt_wav_path = "example.wav"
    prompt_text = ""
    prompt_wav_b64 = encode_wav_to_base64(prompt_wav_path)

    # 发送初始参数
    await ws.send(json.dumps({
        # "spk_id": ""  # 可选参数
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav_b64
    }))

    # 按句发送 TTS 文本内容
    await ws.send(json.dumps({"tts_text": "今天"}))
    await asyncio.sleep(0.05)

    await ws.send(json.dumps({"tts_text": "天气很不错，"}))
    await asyncio.sleep(0.05)

    await ws.send(json.dumps({"tts_text": "我们一起"}))
    await asyncio.sleep(0.05)

    await ws.send(json.dumps({"tts_text": "去郊游吧。"}))

    # 最后发送结束信号
    await ws.send(json.dumps({"finish": True}))

    print("[sender] 发送完毕")

async def receiver(ws, pcm_chunks):
    chunk_count = 0
    try:
        async for message in ws:
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                chunk_count += 1
                print(f"[receiver] 收到音频chunk {chunk_count}, 大小: {len(audio_bytes)} bytes")
            else:
                try:
                    error_data = json.loads(message)
                    if "error" in error_data:
                        print(f"❌ 服务器错误: {error_data['error']}")
                        return
                    else:
                        print(f"📨 收到JSON消息: {error_data}")
    except ConnectionClosed as e:
        print(f"[receiver] 连接关闭: {e}")
    print(f"[receiver] 接收结束，共收到 {chunk_count} 个chunk")

async def tts_bitstream_inference():
    uri = "ws://localhost:8080/inference_bistream"
    pcm_chunks = []
    async with websockets.connect(uri) as ws:
        send_task = asyncio.create_task(sender(ws))
        recv_task = asyncio.create_task(receiver(ws, pcm_chunks))
        results = await asyncio.gather(send_task, recv_task, return_exceptions=True)

    if pcm_chunks:
        pcm_bytes = b"".join(pcm_chunks)
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        samplerate = 24000  # 默认22050
        sf.write("output.wav", pcm_array, samplerate, subtype="PCM_16")
        print("✅ 音频保存成功：output.wav")
    else:
        print("❌ 没有收到音频数据")

 asyncio.run(tts_bitstream_inference())
 ```

## Adding New Speakers

### Method 1: Runtime Addition via API
Add speakers dynamically while the server is running:

```bash
# Add a new speaker with custom audio
curl -X POST "http://localhost:8088/add_spk" \
  -F "spk_id=my_voice" \
  -F "prompt_text=这是参考音频的文本内容" \
  -F "prompt_wav=@my_voice.wav" \
  -F "style=中性"
```

**Requirements:**
- Audio file: 16kHz, mono, WAV format
- Reference text: Clear, noise-free speech
- Speaker ID: Unique identifier

### Method 2: Static Configuration
Add speakers permanently by modifying `light_tts/server/prompt_loader.py`:

```python
prompt_config = {
    "tangwei": {
        "中性": {
            "prompt_wav": find_prompt_audio_path("tangwei2.wav"),
            "prompt_text": "其实最后选上我去拍电视剧的那几位导演..."
        }
    },
    # Add your new speaker
    "your_speaker": {
        "中性": {
            "prompt_wav": "/path/to/your/audio.wav",
            "prompt_text": "这是参考音频的文本内容"
        },
        "开心": {  # Optional: different emotional styles
            "prompt_wav": "/path/to/happy_audio.wav", 
            "prompt_text": "这是开心语调的参考文本"
        }
    }
}
```

Then place audio files in the directory `asset/prompt_audio`