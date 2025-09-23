import ttsfrd
import os
import sys
import json
from cosyvoice.utils.frontend_utils import is_only_punctuation


frd = ttsfrd.TtsFrontendEngine()

# 获取当前文件所在目录，考虑打包环境
def get_root_dir():
    """获取根目录，兼容打包环境"""
    if getattr(sys, 'frozen', False):
        # 在打包的exe中
        if hasattr(sys, '_MEIPASS'):
            # ONEFILE模式：使用临时解压目录
            return sys._MEIPASS
        else:
            # DIR模式：使用exe文件所在目录
            return os.path.dirname(os.path.abspath(sys.argv[0]))
    else:
        # 开发环境：使用当前文件所在目录
        return os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = get_root_dir()

# 使用相对路径查找resource目录
def find_resource_path():
    """查找ttsfrd resource目录"""
    # 首先检查环境变量
    if 'COSYVOICE_PRETRAINED_MODELS' in os.environ:
        env_path = os.path.join(os.environ['COSYVOICE_PRETRAINED_MODELS'], 'CosyVoice-ttsfrd', 'resource')
        if os.path.exists(env_path):
            return env_path
    
    # 在打包环境中，优先使用环境变量设置的路径
    if getattr(sys, 'frozen', False):
        # 在打包环境中，如果没有环境变量，尝试从当前工作目录查找
        current_dir = os.getcwd()
        possible_paths = [
            os.path.join(current_dir, 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'),
            os.path.join(current_dir, '..', 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'),
            os.path.join(current_dir, '..', "..", 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                print("tts frd resource path", abs_path)
                return abs_path
    
    # 开发环境：从当前文件位置开始向上查找
    current_dir = ROOT_DIR
    possible_paths = [
        # 在项目根目录下查找
        os.path.join(current_dir, '..', '..', 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'),
        # 在项目根目录的上级目录查找
        os.path.join(current_dir, '..', '..', '..', 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'),
        os.path.join(current_dir, 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    # 如果都没找到，返回默认路径（用于错误提示）
    return os.path.abspath(os.path.join(current_dir, '..', '..', "..", 'pretrained_models', 'CosyVoice-ttsfrd', 'resource'))

# 延迟初始化，避免在导入时执行
def _lazy_init():
    """延迟初始化ttsfrd"""
    global frd
    resource_path = find_resource_path()
    if not os.path.exists(resource_path):
        raise FileNotFoundError(f'ttsfrd resource directory not found. Tried: {resource_path}')
    
    if not frd.initialize(resource_path):
        raise RuntimeError(f'failed to initialize ttsfrd resource at {resource_path}')
    
    frd.set_lang_type('pinyinvg')
    return True

# 延迟初始化标志
_lazy_init()
initialized = True

def text_normalize(text, split=True, text_frontend=True):
    if text_frontend is False:
        return [text] if split is True else text
    
    text = text.strip()
    texts = [i["text"] for i in json.loads(frd.do_voicegen_frd(text))["sentences"]]
    text = ''.join(texts)
    texts = [i for i in texts if not is_only_punctuation(i)]
    return texts if split is True else text