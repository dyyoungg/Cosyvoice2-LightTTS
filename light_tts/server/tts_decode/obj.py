from dataclasses import dataclass
import numpy as np

@dataclass
class DecodeReq:
    req_tuple: tuple = None
    gen_audios: np.array = None
    gen_sampling_rate: int = None 
    has_exception: str = None