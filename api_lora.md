# LORA语音生成API接口

该API接口用于生成基于LORA风格的语音数据。请求体中需要提供待生成语音的文本输入、LORA风格ID和一些生成参数。API将返回Base64编码后的PCM格式语音数据,以及相关的音频参数。使用示例代码展示了如何发送请求、解码响应数据并将其写入WAV文件。

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
| parameters | 是 | dict | 生成参数对象 包含 do_sample 和 temperature 项 |

### parameters对象参数
| 参数名称 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| do_sample | 是 | boolean | 是否开启采样 |
| temperature | 是 | float | 采样温度，需要 > 0.0 |

## 请求示例
```json
{
  "inputs": "苏醒这两组宣传画本就敏感有争议性，再加上那位一直蹭苏醒热度的画师不停的蹦跶，她的微博热度正高。",
  "style": "base",
  "parameters": {
    "do_sample": true,
    "temperature": 0.8
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
  "pcm_base64": "xxxxxxxxxxxxxxxxxx",
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
import base64
import soundfile as sf
import numpy as np

url = 'http://0.0.0.0:8083/generate/'
headers = {'Content-Type': 'application/json'}
inputs = f"苏醒这两组宣传画本就敏感有争议性，再加上那位一直蹭苏醒热度的画师不停的蹦跶，她的微博热度正高。"

data = {
    'inputs': inputs,
    'style': 'base',
    "parameters": {
        "do_sample": True,
        "temperature": 0.8,
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
‵‵‵