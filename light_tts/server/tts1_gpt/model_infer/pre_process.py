import torch
import numpy as np
from .infer_batch import requests_mapping, InferReq, InferBatch
from light_tts.server.io_struct import ReqRunStatus
from light_tts.utils.infer_utils import calculate_time
from light_tts.server.shm_tools.shm_objs import SharedTensorManager
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

#@calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(batch:InferBatch):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    temperature_list = []
    ignore_eos = []
    bistream_list = []
    win_size = 10
    pad_token = -1
    batch_size = len(batch.request_ids)
    padded_output = torch.full((batch_size, win_size), pad_token, dtype=torch.int64, device='cuda')
    b_ready_cache_len = []
    b_next_fill = []

    for i, request_id in enumerate(batch.request_ids):
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        temperature_list.append(req.sampling_param.temperature)

        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        
        seq_len = len(req.input_token_ids)
        input_token_len = seq_len - req.cur_kv_len
        input_id = req.input_token_ids[req.cur_kv_len:]
        
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len
        ignore_eos.append((not req.bistream) or (req.ignore_eos))
        b_next_fill.append(req.bistream and (req.next_fill_index == len(req.output_token_ids)))
        bistream_list.append(req.bistream)
        output_token_ids = torch.tensor(req.output_token_ids, dtype=torch.int64, device='cuda')
        if len(output_token_ids) > 0:
            length = min(win_size, output_token_ids.shape[0])
            padded_output[i, -length:] = output_token_ids[-length:]
        
    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    temperature = torch.tensor(temperature_list, dtype=torch.float32, device='cuda')
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
    ignore_eos = torch.tensor(ignore_eos, dtype=torch.bool, device='cuda')
    b_next_fill = torch.tensor(b_next_fill, dtype=torch.bool, device='cuda')
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device='cuda')
    bistream_list = torch.tensor(bistream_list, dtype=torch.bool, device='cuda')

    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "temperature": temperature,
        "b_ready_cache_len": b_ready_cache_len,
        "output_token_ids": padded_output,
        "is_prefill": True,
        "ignore_eos": ignore_eos,
        "b_next_fill": b_next_fill,
        "bistream": bistream_list
    }
    
    return kwargs, run_reqs
    
#@calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch:InferBatch):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    temperature_list = []
    ignore_eos = []
    win_size = 10
    pad_token = -1
    batch_size = len(batch.request_ids)
    padded_output = torch.full((batch_size, win_size), pad_token, dtype=torch.int64, device='cuda')
    b_next_fill = []
    bistream_list = []

    for i, request_id in enumerate(batch.request_ids):
        req : InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        temperature_list.append(req.sampling_param.temperature)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_id = req.input_token_ids[-1]
        seq_len = len(req.input_token_ids)
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len

        output_token_ids = torch.tensor(req.output_token_ids, dtype=torch.int64, device='cuda')
        length = min(win_size, output_token_ids.shape[0])
        padded_output[i, -length:] = output_token_ids[-length:]

        ignore_eos.append(
            (not req.bistream and (len(req.output_token_ids) < req.sampling_param.min_new_tokens))
            or 
            (req.bistream and req.ignore_eos)
        )
        b_next_fill.append(req.bistream and (req.next_fill_index == len(req.output_token_ids)))
        bistream_list.append(req.bistream)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    temperature = torch.tensor(temperature_list, dtype=torch.float32, device='cuda')
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device='cuda')
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device='cuda')
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device='cuda')
    ignore_eos = torch.tensor(ignore_eos, dtype=torch.bool, device='cuda')
    b_next_fill = torch.tensor(b_next_fill, dtype=torch.bool, device='cuda')
    bistream_list = torch.tensor(bistream_list, dtype=torch.bool, device='cuda')

    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "temperature": temperature,
        "output_token_ids": padded_output,
        "is_prefill": False,
        "ignore_eos": ignore_eos,
        "b_next_fill": b_next_fill,
        "bistream": bistream_list
    }
    return kwargs, run_reqs
