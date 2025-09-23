import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
from typing import List, Union

class TTSBitstreamClient:
    def __init__(self, uri: str = "ws://localhost:50000/inference_bistream"):
        self.uri = uri
        self.pcm_chunks: List[bytes] = []
        self.error_occurred = False
        self.error_message = ""
    
    async def sender(self, ws):
        """发送初始化参数和文本数据"""
        try:
            # 发送初始化参数
            init_params = {
                "spk_id": "test_speaker",  # 或者使用具体的spk_id
                "tts_model_name": "default"
            }
            await ws.send(json.dumps(init_params))
            print("📤 发送初始化参数")
            
            # 发送文本数据
            texts = ["你好，这是一个测试。", "这是第二句话。"]
            for i, text in enumerate(texts):
                await ws.send(json.dumps({"tts_text": text}))
                print(f"📤 发送文本 {i+1}: {text}")
                await asyncio.sleep(0.1)  # 稍微延迟
            
            # 发送结束信号
            await ws.send(json.dumps({"finish": True}))
            print("📤 发送结束信号")
            
        except Exception as e:
            print(f"❌ 发送数据时出错: {e}")
            self.error_occurred = True
            self.error_message = str(e)

    async def receiver(self, ws):
        """接收音频数据和错误信息"""
        try:
            async for message in ws:
                # 尝试解析为JSON（错误消息）
                try:
                    error_data = json.loads(message)
                    if "error" in error_data:
                        print(f"❌ 服务器错误: {error_data['error']}")
                        self.error_occurred = True
                        self.error_message = error_data['error']
                        return
                    else:
                        print(f"📨 收到JSON消息: {error_data}")
                except json.JSONDecodeError:
                    # 不是JSON，应该是音频数据
                    if isinstance(message, bytes):
                        self.pcm_chunks.append(message)
                        print(f"📦 收到音频数据块: {len(message)} bytes")
                    else:
                        print(f"⚠️ 收到未知类型数据: {type(message)} - {message}")
                        
        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 WebSocket连接关闭: {e}")
            if e.code != 1000:  # 1000是正常关闭
                self.error_occurred = True
                self.error_message = f"连接异常关闭: {e}"
        except Exception as e:
            print(f"❌ 接收数据时出错: {e}")
            self.error_occurred = True
            self.error_message = str(e)

    async def run(self):
        """运行TTS推理"""
        try:
            async with websockets.connect(self.uri) as ws:
                print("🔌 WebSocket连接已建立")
                
                # 创建发送和接收任务
                send_task = asyncio.create_task(self.sender(ws))
                recv_task = asyncio.create_task(self.receiver(ws))
                
                # 等待任务完成
                results = await asyncio.gather(send_task, recv_task, return_exceptions=True)
                
                # 检查任务结果
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"❌ 任务 {i} 出现异常: {result}")
                        self.error_occurred = True
                        self.error_message = str(result)
                        
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            self.error_occurred = True
            self.error_message = str(e)
            return

        # 处理结果
        if self.error_occurred:
            print(f"❌ 推理失败: {self.error_message}")
            return False
        elif self.pcm_chunks:
            pcm_bytes = b"".join(self.pcm_chunks)
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            samplerate = 24000  # 默认24000
            sf.write("output.wav", pcm_array, samplerate, subtype="PCM_16")
            print(f"✅ 音频保存成功：output.wav (总长度: {len(pcm_array)} 样本)")
            return True
        else:
            print("❌ 没有收到音频数据")
            return False

async def main():
    client = TTSBitstreamClient()
    success = await client.run()
    if success:
        print("🎉 TTS推理完成！")
    else:
        print("💥 TTS推理失败！")

if __name__ == "__main__":
    asyncio.run(main())
