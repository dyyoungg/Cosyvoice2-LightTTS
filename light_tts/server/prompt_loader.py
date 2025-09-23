import os
import hashlib
import torchaudio

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 查找prompt音频文件的相对路径
def find_prompt_audio_path(filename):
    """查找prompt音频文件路径"""
    # 首先检查环境变量
    if 'COSYVOICE_PROMPT_AUDIO' in os.environ:
        env_path = os.path.join(os.environ['COSYVOICE_PROMPT_AUDIO'], filename)
        if os.path.exists(env_path):
            return env_path
    
    # 从当前文件位置开始查找
    possible_paths = [
        # 在项目根目录下查找
        os.path.join(current_dir, '..', '..', 'assets', 'prompt_audio', filename),
        os.path.join(current_dir, '..', '..', 'prompt_audio', filename),
        # 在cosyvoice目录下查找
        os.path.join(current_dir, '..', '..', 'cosyvoice', 'assets', 'prompt_audio', filename),
        # 在打包的exe中查找
        os.path.join(current_dir, '..', '..', 'assets', 'prompt_audio', filename),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    # 如果都没找到，返回默认路径（用于错误提示）
    return os.path.join(current_dir, '..', '..', 'assets', 'prompt_audio', filename)

prompt_config = {
    "tangwei":{
                "中性": {"prompt_wav": find_prompt_audio_path("tangwei2.wav"),
                        "prompt_text": "其实最后选上我去拍电视剧的那几位导演，他们都是看我的简历，看了简历以后，不用见我人，直接就定了我。"
                        }
    },
    "中文女1":{
                "中性": {"prompt_wav": find_prompt_audio_path("asaki2.wav"),
                        "prompt_text": "难道一直没有发现主播不太喜欢这个梗么，然后他就一直评论一直评论，然后我就把他删了，你再发一次，我就把你拉黑了。"
                        }
    },
    "中文女":{
                "中性": {"prompt_wav": find_prompt_audio_path("asaki_short.wav"),
                        "prompt_text": "难道一直没有发现主播不太喜欢这个梗么，然后他就一直评论一直评论，然后我就把他删了。"
                        }
    }
}