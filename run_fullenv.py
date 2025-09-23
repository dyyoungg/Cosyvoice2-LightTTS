#!/usr/bin/env python3
"""
CosyVoice 全环境自包含启动入口
仅需用户提供 --model_dir，其余环境与依赖均随可执行程序打包。
"""

import os
import sys
import argparse
import logging
import warnings
import subprocess

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 避免 typeguard 在冻结环境中尝试读取源码导致失败
os.environ.setdefault("TYPEGUARD_DISABLE", "true")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosyvoice_fullenv")


def _get_exe_base_dir() -> str:
    """返回冻结/开发环境下的基准目录。

    - ONEFILE: sys._MEIPASS（临时解压目录）
    - ONEDIR:  可执行文件所在目录
    - 开发环境: 本文件所在目录
    """
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


def _prepare_runtime_env(env: dict) -> dict:
    base_dir = _get_exe_base_dir()

    # 预训练资源：打包内提供 ttsfrd resource
    # 放置位置：
    # - ONEFILE:   <_MEIPASS>/pretrained_models/CosyVoice-ttsfrd/resource
    # - ONEDIR:    <exe_dir>/pretrained_models/CosyVoice-ttsfrd/resource
    pretrained_models_path = os.path.join(base_dir, 'pretrained_models')
    env["COSYVOICE_PRETRAINED_MODELS"] = os.path.abspath(pretrained_models_path)

    # 提示音频资源（如存在）
    assets_prompt_audio = os.path.join(base_dir, 'assets', 'prompt_audio')
    if os.path.exists(assets_prompt_audio):
        env["COSYVOICE_PROMPT_AUDIO"] = os.path.abspath(assets_prompt_audio)

    # 抑制常见警告
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"

    # 禁用 flash-attn / triton 相关快速路径，避免冻结环境下源码/编译依赖
    env.setdefault("HF_USE_FLASH_ATTENTION_2", "0")
    env.setdefault("USE_FLASH_ATTENTION", "0")
    env.setdefault("FLASH_ATTENTION_SKIP", "1")
    env.setdefault("TRITON_DISABLE_JIT", "1")
    env.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
    env.setdefault("DISABLE_TORCHINDUCTOR", "1")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    return env


def _is_multiprocessing_child() -> bool:
    # PyInstaller/multiprocessing 子进程常见标记
    if any(arg.startswith('--multiprocessing') for arg in sys.argv[1:]):
        return True
    if os.environ.get('PYI_CHILD_PROCESS') == '1':
        return True
    if os.environ.get('PYTHONMULTIPROCESSING_SPAWN') == '1':
        return True
    return False


def main() -> int:
    # 子进程不执行顶层参数解析
    if _is_multiprocessing_child():
        return 0
    parser = argparse.ArgumentParser(description='CosyVoice 全环境自包含启动器')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8089, help='服务器端口')

    # 高级参数保留默认，用户一般无需关心；如需暴露可追加参数
    parser.add_argument('--bert_process_num', type=int, default=1)
    parser.add_argument('--decode_process_num', type=int, default=1)
    parser.add_argument('--max_total_token_num', type=int, default=60000)
    parser.add_argument('--encode_paral_num', type=int, default=50)
    parser.add_argument('--gpt_paral_num', type=int, default=50)
    parser.add_argument('--decode_paral_num', type=int, default=1)
    parser.add_argument('--mode', type=str, default='triton_flashdecoding')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        return 1

    # 运行环境变量（供底层依赖使用）
    env = _prepare_runtime_env(os.environ)
    # 保留 triton JIT，但把缓存指向可写目录
    try:
        from pathlib import Path as _Path
        cache_root = _Path(os.environ.get('XDG_CACHE_HOME', _Path.home() / '.cache')) / 'cosyvoice_fullenv'
        triton_cache = cache_root / 'triton'
        inductor_cache = cache_root / 'torchinductor'
        triton_cache.mkdir(parents=True, exist_ok=True)
        inductor_cache.mkdir(parents=True, exist_ok=True)
        env.setdefault('TRITON_CACHE_DIR', str(triton_cache))
        env.setdefault('TORCHINDUCTOR_CACHE_DIR', str(inductor_cache))
        env.setdefault('PYTORCH_KERNEL_CACHE_PATH', str(inductor_cache))
    except Exception:
        pass

    # 直接调用服务器入口函数，避免使用 -m
    try:
        # 设置多进程启动方式（与原 __main__ 保持一致）
        try:
            import torch.multiprocessing as mp
            mp.set_start_method('spawn', force=True)
        except Exception:
            pass

        # 传递参数给下游 argparse
        import sys as _sys
        _old_argv = list(_sys.argv)
        _sys.argv = [
            'lightllm.server.api_server',
            '--model_dir', args.model_dir,
            '--host', args.host,
            '--port', str(args.port),
            '--bert_process_num', str(args.bert_process_num),
            '--decode_process_num', str(args.decode_process_num),
            '--max_total_token_num', str(args.max_total_token_num),
            '--encode_paral_num', str(args.encode_paral_num),
            '--gpt_paral_num', str(args.gpt_paral_num),
            '--decode_paral_num', str(args.decode_paral_num),
            '--mode', args.mode,
        ]

        # 导入并启动
        from light_tts.server import api_server as _api
        logger.info("正在启动 CosyVoice 服务(自包含环境)...")
        logger.info(f"模型目录: {args.model_dir}")
        logger.info(f"服务地址: http://{args.host}:{args.port}")
        _api.main()
        _sys.argv = _old_argv
        return 0
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务器...")
        return 0
    except Exception as e:
        logger.exception(f"启动失败: {e}")
        return 1


if __name__ == '__main__':
    try:
        import multiprocessing as _mp
        _mp.freeze_support()
    except Exception:
        pass
    sys.exit(main())


