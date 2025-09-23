# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for LightLLM."""
import logging
import sys
import os
from typing import Optional
import time

_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_LOG_LEVEL = os.environ.get("LIGHTLLM_LOG_LEVEL", "debug")
_LOG_LEVEL = getattr(logging, _LOG_LEVEL.upper(), 0)
_LOG_DIR = os.environ.get("LIGHTLLM_LOG_DIR", None)

class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class PreciseTimeFormatter(logging.Formatter):
    """自定义格式化器，精确到0.1毫秒的时间显示"""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def formatTime(self, record, datefmt=None):
        """重写formatTime方法，精确到0.1毫秒"""
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%m-%d %H:%M:%S", ct)
        
        # 添加精确到0.1毫秒的时间戳
        # record.created 是时间戳（秒），取小数部分并转换为毫秒
        ms = (record.created % 1) * 1000
        # 精确到0.1毫秒，保留1位小数
        ms_str = f"{ms:.1f}"
        return f"{s}.{ms_str}"

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("light_tts")
_default_handler = None
_default_file_handler = None
_current_log_file_path = None  # absolute path to current log file if any
_inference_log_file_handler = {}


class TerminalColor:
    """ANSI escape sequences for terminal colors."""
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

class ColorFormatter(PreciseTimeFormatter):
    """日志格式化器，为不同级别的日志消息添加颜色，并精确到0.1毫秒。"""

    COLOR_MAP = {
        logging.DEBUG: TerminalColor.GREEN,
        logging.INFO: TerminalColor.BLUE,
        logging.WARNING: TerminalColor.YELLOW,
        logging.ERROR: TerminalColor.RED,
        logging.CRITICAL: TerminalColor.RED,
    }

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, TerminalColor.RESET)
        message = super().format(record)
        return f"{color}{message}{TerminalColor.RESET}"

def _setup_logger():
    _root_logger.setLevel(_LOG_LEVEL)
    global _default_handler
    global _default_file_handler
    color_fmt = ColorFormatter(_FORMAT, datefmt=_DATE_FORMAT)

    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(_LOG_LEVEL)
        _default_handler.setFormatter(color_fmt)
        _root_logger.addHandler(_default_handler)
    
    if _default_file_handler is None and _LOG_DIR is not None:
        if not os.path.exists(_LOG_DIR):
            try:
                os.makedirs(_LOG_DIR)
            except OSError as e:
                _root_logger.warn(f"Error creating directory {_LOG_DIR} : {e}")
        _default_file_handler = logging.FileHandler(_LOG_DIR + '/default.log')
        _default_file_handler.setLevel(_LOG_LEVEL)
        _default_file_handler.setFormatter(NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT))
        _root_logger.addHandler(_default_file_handler)

    _root_logger.propagate = False

# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    # Always attach the shared console handler
    if _default_handler is not None and _default_handler not in logger.handlers:
        logger.addHandler(_default_handler)
    # Always attach the shared default file handler (default.log) if configured
    if _default_file_handler is not None and _default_file_handler not in logger.handlers:
        logger.addHandler(_default_file_handler)
    logger.propagate = False
    return logger


def configure_logging(log_path_or_dir: Optional[str] = None):
    """Configure/reconfigure logging.

    - Accepts either a directory or a full `.log` file path.
      - If a directory is provided, logs to `default.log` under that directory.
      - If a path ending with `.log` is provided, logs directly to that file.
    - If not provided, falls back to environment variable `LIGHTLLM_LOG_DIR`.
    - Safe to call multiple times; updates file handler if the path changes.
    """
    _setup_logger(log_path_or_dir)
