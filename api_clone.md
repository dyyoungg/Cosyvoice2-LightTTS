# Clone 语音生成API接口(支持克隆语音)

该API接口支持基于LORA风格生成语音,同时也支持语音克隆功能。请求体中需要提供待生成语音的文本输入、LORA风格ID、生成参数,以及用于语音克隆的音频数据和对应文本。API将返回Base64编码后的PCM格式语音数据,以及相关的音频参数。使用示例代码展示了如何发送请求(包含语音克隆数据)、解码响应数据并将其写入WAV文件。

## 基本信息
- 请求方式: POST
- 请求URL: http://0.0.0.0:8083/generate/

## 请求头参数
| 参数名称 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| Content-Type | 是 | string | 请求体数据格式,应为application/json |

## 请求体参数(Content-Type=application/json)
| 参数名称 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| inputs | 是 | string | 待生成语音的文本输入 |
| style | 是 | string | LORA风格ID |
| parameters | 是 | dict | 生成参数对象 |
| clone_params | 否 | dict | 克隆语音参数对象,用于语音克隆 |

### parameters对象参数
| 参数名称 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| do_sample | 是 | boolean | 是否开启采样 |
| temperature | 是 | float | 采样温度 |

### clone_params对象参数
| 参数名称 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| speech | 是 | dict | 克隆语音对象 |

#### speech对象参数
| 参数名称 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| type | 是 | string | 传输数据格式,如base64 |
| data | 是 | string | Base64编码后的克隆语音数据, 原始文件流格式为.wav |
| p_phones | 是 | string | 克隆语音数据对应的文本内容 |

## 请求示例
```json
{
  "inputs": "苏醒这两组宣传画本就敏感有争议性，再加上那位一直蹭苏醒热度的画师不停的蹦跶，她的微博热度正高。",
  "style": "0",
  "parameters": {
    "do_sample": true,
    "temperature": 0.8
  },
  "clone_params": {
    "speech": {
      "type": "base64",
      "data": "xxxxxxxxxxx",
      "p_phones": "恶魔与人类的关系刻不容缓"
    }
  }
}
```

## 返回参数
| 参数名称 | 类型 | 说明 |
| --- | --- | --- |
| pcm_base64 | string | Base64编码后的PCM格式语音数据 |
| sampling_rate | integer | 采样频率 |
| bit_depth | integer | 采样位深度 |
| channels | integer | 声道数 |
| byte_order | string | 字节序,little或big |

## 响应示例
```json
{
  "pcm_base64": "xxxxxxxxxxxxxx",
  "sampling_rate": 16000,
  "bit_depth": 16,
  "channels": 1,
  "byte_order": "little"
}
```

## 使用示例
```python
import time
import requests
import json
import soundfile as sf
from io import BytesIO
import numpy as np

url = 'http://0.0.0.0:8083/generate/'
headers = {'Content-Type': 'application/json'}
inputs = f"苏醒这两组宣传画本就敏感有争议性，再加上那位一直蹭苏醒热度的画师不停的蹦跶，她的微博热度正高。"
clone_wav_path = "/nvme/baishihao/tts/tts_clones/clones/0000_20240116_140013.815476_prompt_final_denoise_segmentation.wav"
with open(clone_wav_path, 'rb') as fileObj:
    audio_data = fileObj.read()
    base64_data = base64.b64encode(audio_data).decode("utf-8")

data = {
    'inputs': inputs,
    'style': '0',
    "parameters": {
        "do_sample": True,
        "temperature": 0.8,
    },
    "clone_params": {
        "speech": {
            "type" : "base64",
            "data" : base64_data,
            "p_phones" : "恶魔与人类的关系刻不容缓"
        }
    }
}

start = time.time()
response = requests.post(url, headers=headers, data=json.dumps(data))
result_json = response.json()
print("cost time:", time.time() - start)

pcm_data = base64.b64decode(result_json["pcm_base64"])
sampling_rate = result_json["sampling_rate"]
bit_depth = result_json["bit_depth"]
channels = result_json["channels"]
byte_order = result_json["byte_order"]

if bit_depth == 16:
    dtype = np.int16 if byte_order == 'little' else np.int16.newbyteorder('>')
elif bit_depth == 8:
    dtype = np.int8 if byte_order == 'little' else np.int8.newbyteorder('>')

pcm_data = np.frombuffer(pcm_data, dtype=dtype)

sf.write('output.wav', pcm_data, sampling_rate, subtype='PCM_16')
```