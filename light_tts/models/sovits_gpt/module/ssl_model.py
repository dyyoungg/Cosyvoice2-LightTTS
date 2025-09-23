import torch
import logging
import torch.nn as nn
logging.getLogger("numba").setLevel(logging.WARNING)
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

class CNHubert(nn.Module):
    def __init__(self, base_path:str=None):
        super().__init__()
        self.model = HubertModel.from_pretrained(base_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats

def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1, 2)
