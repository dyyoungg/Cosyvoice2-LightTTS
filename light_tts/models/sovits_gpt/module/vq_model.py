import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from light_tts.models.sovits_gpt.module import commons
from light_tts.models.sovits_gpt.module import modules
from light_tts.models.sovits_gpt.module.models import (
    TextEncoder,
    Generator,
    PosteriorEncoder,
    ResidualCouplingBlock,
)
from light_tts.models.sovits_gpt.module.quantize import ResidualVectorQuantizer
from torch.cuda.amp import autocast
from light_tts.utils.infer_utils import calculate_time, mark_start, mark_end


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.spec_channels = params.data.filter_length // 2 + 1
        self.inter_channels = params.model.inter_channels
        self.segment_size = params.train.segment_size // params.data.hop_length

        self.hidden_channels = params.model.hidden_channels
        self.filter_channels = params.model.filter_channels
        self.n_heads = params.model.n_heads
        self.n_layers = params.model.n_layers
        self.kernel_size = params.model.kernel_size
        self.p_dropout = params.model.p_dropout
        self.resblock = params.model.resblock
        self.resblock_kernel_sizes = params.model.resblock_kernel_sizes
        self.resblock_dilation_sizes = params.model.resblock_dilation_sizes
        self.upsample_rates = params.model.upsample_rates
        self.upsample_initial_channel = params.model.upsample_initial_channel
        self.upsample_kernel_sizes = params.model.upsample_kernel_sizes
        self.n_speakers = params.data.n_speakers
        self.gin_channels = params.model.gin_channels

        params_dict = params.model.__dict__
        self.use_sdp = params_dict.get("use_sdp", False)
        self.semantic_frame_rate = params_dict.get("semantic_frame_rate", "25hz")
        self.freeze_quantizer = params_dict.get("freeze_quantizer", False)

        self.enc_p = TextEncoder(
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        )
        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            4,
            gin_channels=self.gin_channels,
        )

        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=self.gin_channels)

        ssl_dim = 768
        assert self.semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = self.semantic_frame_rate
        if self.semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        if self.freeze_quantizer:
            self.ssl_proj.requires_grad_(False)
            self.quantizer.requires_grad_(False)
            # self.quantizer.eval()
            # self.enc_p.text_embedding.requires_grad_(False)
            # self.enc_p.encoder_text.requires_grad_(False)
            # self.enc_p.mrte.requires_grad_(False)

    # def forward(self, ssl, y, y_lengths, text, text_lengths):
    #     y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
    #         y.dtype
    #     )
    #     ge = self.ref_enc(y * y_mask, y_mask)

    #     with autocast(enabled=False):
    #         ssl = self.ssl_proj(ssl)
    #         quantized, codes, commit_loss, quantized_list = self.quantizer(
    #             ssl, layers=[0]
    #         )

    #     if self.semantic_frame_rate == "25hz":
    #         quantized = F.interpolate(
    #             quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
    #         )

    #     x, m_p, logs_p, y_mask = self.enc_p(
    #         quantized, y_lengths, text, text_lengths, ge
    #     )
    #     z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
    #     z_p = self.flow(z, y_mask, g=ge)

    #     z_slice, ids_slice = commons.rand_slice_segments(
    #         z, y_lengths, self.segment_size
    #     )
    #     o = self.dec(z_slice, g=ge)
    #     return (
    #         o,
    #         commit_loss,
    #         ids_slice,
    #         y_mask,
    #         y_mask,
    #         (z, z_p, m_p, logs_p, m_q, logs_q),
    #         quantized,
    #     )

    # def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
    #     y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
    #         y.dtype
    #     )
    #     ge = self.ref_enc(y * y_mask, y_mask)

    #     ssl = self.ssl_proj(ssl)
    #     quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
    #     if self.semantic_frame_rate == "25hz":
    #         quantized = F.interpolate(
    #             quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
    #         )

    #     x, m_p, logs_p, y_mask = self.enc_p(
    #         quantized, y_lengths, text, text_lengths, ge, test=test
    #     )
    #     z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

    #     z = self.flow(z_p, y_mask, g=ge, reverse=True)

    #     o = self.dec((z * y_mask)[:, :, :], g=ge)
    #     return o, y_mask, (z, z_p, m_p, logs_p)

    def init_refer(self, refer):
        # if refer is not None:
        #     refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        #     refer_mask = torch.unsqueeze(
        #         commons.sequence_mask(refer_lengths, refer.size(2)), 1
        #     ).to(refer.dtype)
        #     self.ge = self.ref_enc(refer[:,:704] * refer_mask, refer_mask)
        # else:
        #     self.ge = None
        def get_ge(refer):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(
                    commons.sequence_mask(refer_lengths, refer.size(2)), 1
                ).to(refer.dtype)
                ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
            return ge
        ges = []
        for _refer in refer:
            ge = get_ge(_refer)
            ges.append(ge)
        self.ge = torch.stack(ges, 0).mean(0)
        del ges
        torch.cuda.empty_cache()
        del self.ref_enc
        torch.cuda.empty_cache()

    @calculate_time(show=True)
    @torch.no_grad()
    def decode(self, codes, text, noise_scale=0.5):
        ge = self.ge

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)

        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)
        # print('text')
        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o.cpu().numpy()

    @calculate_time(show=True)
    @torch.inference_mode()
    def decode_batch(
        self, codes: List[torch.Tensor], texts: List[torch.Tensor], noise_scale=0.5
    ):
        ge = self.ge
        y_lengths = []
        y_length_max = 0
        text_lengths = []
        text_length_max = 0

        for cur_code, cur_text in zip(codes, texts):
            y_lengths.append(cur_code.size(2) * 2)
            y_length_max = max(y_length_max, y_lengths[-1])
            text_lengths.append(cur_text.size(-1))
            text_length_max = max(text_length_max, text_lengths[-1])

        # 下面的pad 长度计算的代码，对于提升速度很重要，不同的shape，会导致很多卷积算子的性能下降
        import math

        y_length_max = 2 ** math.ceil(math.log2(y_length_max + 0.1))
        text_length_max = 2 ** math.ceil(math.log2(text_length_max + 0.1))

        y_lengths = torch.tensor(y_lengths, dtype=torch.int64, device="cuda")
        text_lengths = torch.tensor(text_lengths, dtype=torch.int64, device="cuda")

        # pad
        pad_codes = []
        pad_texts = []
        for cur_code, cur_text in zip(codes, texts):
            pad_codes.append(
                F.pad(
                    cur_code,
                    (0, (y_length_max // 2) - cur_code.size(-1)),
                    mode="constant",
                    value=0,
                )
            )
            pad_texts.append(
                F.pad(
                    cur_text,
                    (0, text_length_max - cur_text.size(-1)),
                    mode="constant",
                    value=0,
                )
            )

        pad_codes = torch.concat(pad_codes, dim=1)
        pad_texts = torch.concat(pad_texts, dim=0)

        quantized = self.quantizer.decode(pad_codes)

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, pad_texts, text_lengths, ge
        )

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge)

        o = self.dec((z * y_mask)[:, :, :], g=ge)

        o = o.cpu().numpy()
        o_list = []
        for i, cur_code in enumerate(codes):
            o_list.append(o[i : i + 1, :, 0 : (1280 * cur_code.size(-1))])

        return o_list

    # @torch.no_grad()
    # def batched_decode(self, codes, y_lengths, text, text_lengths, refer, noise_scale=0.5):
    #     ge = None
    #     if refer is not None:
    #         refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
    #         refer_mask = torch.unsqueeze(
    #             commons.sequence_mask(refer_lengths, refer.size(2)), 1
    #         ).to(refer.dtype)
    #         ge = self.ref_enc(refer * refer_mask, refer_mask)

    #     # y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, codes.size(2)), 1).to(
    #     #     codes.dtype
    #     # )
    #     y_lengths = (y_lengths * 2).long().to(codes.device)
    #     text_lengths = text_lengths.long().to(text.device)
    #     # y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
    #     # text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

    #     # 假设padding之后再decode没有问题, 影响未知，但听起来好像没问题？
    #     quantized = self.quantizer.decode(codes)
    #     if self.semantic_frame_rate == "25hz":
    #         quantized = F.interpolate(
    #             quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
    #         )

    #     x, m_p, logs_p, y_mask = self.enc_p(
    #         quantized, y_lengths, text, text_lengths, ge
    #     )
    #     z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

    #     z = self.flow(z_p, y_mask, g=ge, reverse=True)
    #     z_masked = (z * y_mask)[:, :, :]

    #     # 串行。把padding部分去掉再decode
    #     o_list:List[torch.Tensor] = []
    #     for i in range(z_masked.shape[0]):
    #         z_slice = z_masked[i, :, :y_lengths[i]].unsqueeze(0)
    #         o = self.dec(z_slice, g=ge)[0, 0, :].detach()
    #         o_list.append(o)

    #     # 并行（会有问题）。先decode，再把padding的部分去掉
    #     # o = self.dec(z_masked, g=ge)
    #     # upsample_rate = int(math.prod(self.upsample_rates))
    #     # o_lengths = y_lengths*upsample_rate
    #     # o_list = [o[i, 0, :idx].detach() for i, idx in enumerate(o_lengths)]

    #     return o_list

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)
