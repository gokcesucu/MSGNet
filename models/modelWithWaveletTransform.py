#MODEL 1
import numpy as np
import pywt  # Ensure pywt is installed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, Attention_Block, Predict


def FFT_for_Period(x, k=2):
    # [B, T, C]
   
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    
    # period = [24, 12, 8, 6, 4] based on top frequencies
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def wavelet_transform(x, wavelet='haar', level=1):
    B, T, C = x.size()
    coeffs = []

    for b in range(B):
        for c in range(C):
            # Perform a simplified wavelet decomposition
            coeff = pywt.wavedec(x[b, :, c].detach().cpu().numpy(), wavelet, level=level)
            # Flatten and store coefficients
            flattened_coeff = np.concatenate([c.flatten() for c in coeff])
            coeffs.append(flattened_coeff)

    # Find the maximum length and pad other sequences
    max_len = max(len(c) for c in coeffs)
    coeffs_padded = [np.pad(c, (0, max_len - len(c))) for c in coeffs]

    # Convert to tensor and reshape
    coeffs_tensor = torch.tensor(coeffs_padded).to(x.device)
    coeffs_tensor = coeffs_tensor.view(B, -1, C)
    return coeffs_tensor


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # Reduced number of attention heads for optimization
        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                    n_heads=max(1, configs.n_heads // 2), dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList([GraphBlock(configs.c_out, configs.d_model, configs.conv_channel,
                                               configs.skip_channel, configs.gcn_depth, configs.dropout,
                                               configs.propalpha, configs.seq_len, configs.node_dim)
                                    for _ in range(self.k)])

    def forward(self, x):
        B, T, N = x.size()

        # Apply FFT to detect periods
        scale_list, scale_weight = FFT_for_Period(x, self.k)

        # Optionally apply wavelet transform
        use_wavelet = True  # Toggle this to experiment with/without wavelet transform
        if use_wavelet:
            wavelet_coeffs = wavelet_transform(x, wavelet='haar', level=1)  # Simplified wavelet transform

        res = []
        for i in range(self.k):
            scale = scale_list[i]
            # Apply graph convolution
            x_gconv = self.gconv[i](x)
            
            # Padding for different scales
            if self.seq_len % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x_gconv.shape[0], (length - self.seq_len), x_gconv.shape[2]]).to(x.device)
                out = torch.cat([x_gconv, padding], dim=1)
            else:
                length = self.seq_len
                out = x_gconv
            
            out = out.reshape(B, length // scale, scale, N)

            # Apply optimized attention mechanism
            out = out.reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)
            
            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        
        # Adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        
        # Add residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reduced number of layers for potential performance gain
        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(max(1, configs.e_layers - 1))])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.seq2pred = Predict(configs.individual, configs.c_out,
                                configs.seq_len, configs.pred_len, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        
        for i in range(len(self.model)):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Project back to original dimensions
        dec_out = self.projection(enc_out)
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]
