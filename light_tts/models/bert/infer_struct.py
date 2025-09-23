import torch
import numpy as np

class BertInferStateInfo:

    def __init__(self):
        self.b_start_loc = None
        self.b_seq_len = None
        self.max_len_in_batch = None
        self.position_ids = None
        self.mid_o = None
