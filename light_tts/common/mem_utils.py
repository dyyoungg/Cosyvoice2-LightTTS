from light_tts.common.mem_manager import MemoryManager
from light_tts.common.int8kv_mem_manager import INT8KVMemoryManager
from light_tts.common.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

def select_mem_manager_class(mode):
    logger.info(f"mode setting params: {mode}")
    if "ppl_int8kv" in mode:
        memory_manager_class = PPLINT8KVMemoryManager
        logger.info("Model kv cache using mode ppl int8kv")
    elif "triton_int8kv" in mode:
        memory_manager_class = INT8KVMemoryManager
        logger.info("Model kv cache using mode triton int8kv")
    else:
        memory_manager_class = MemoryManager
        logger.info("Model kv cache using mode normal")
    return memory_manager_class