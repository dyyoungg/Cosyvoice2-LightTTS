#!/usr/bin/env python3
"""
LightLLM CosyVoice Server - 主入口文件
用于打包成exe文件，隐藏源码
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 设置环境变量
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LightLLM CosyVoice Server')
    
    # 添加所有必要的参数
    parser.add_argument('--model_dir', type=str, 
                       default='/mnt/afs/share/CosyVoice2-0.5B',
                       help='模型目录路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8089,
                       help='服务器端口')
    parser.add_argument('--bert_process_num', type=int, default=1,
                       help='BERT进程数量')
    parser.add_argument('--decode_process_num', type=int, default=1,
                       help='解码进程数量')
    parser.add_argument('--max_total_token_num', type=int, default=60000,
                       help='最大总token数量')
    parser.add_argument('--encode_paral_num', type=int, default=50,
                       help='编码并行数量')
    parser.add_argument('--gpt_paral_num', type=int, default=50,
                       help='GPT并行数量')
    parser.add_argument('--decode_paral_num', type=int, default=1,
                       help='解码并行数量')
    parser.add_argument('--mode', type=str, default='triton_flashdecoding',
                       help='运行模式')
    
    args = parser.parse_args()
    
    # 检查模型目录是否存在
    if not os.path.exists(args.model_dir):
        logger.error(f"模型目录不存在: {args.model_dir}")
        logger.info("请确保模型文件已正确放置，或使用 --model_dir 参数指定正确的路径")
        return 1
    
    try:
        # 导入并启动服务器
        logger.info("正在启动 LightLLM CosyVoice 服务器...")
        logger.info(f"模型目录: {args.model_dir}")
        logger.info(f"服务地址: http://{args.host}:{args.port}")
        
        # 动态导入lightllm模块
        from light_tts.server.api_server import main as server_main
        
        # 设置sys.argv以传递给原始服务器
        sys.argv = [
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
            '--mode', args.mode
        ]
        
        # 启动服务器
        server_main()
        
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.error("请确保所有依赖已正确安装")
        return 1
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
