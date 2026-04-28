"""
cnn_attention.py
────────────────
CNN with Attention mechanism — the core neural network from the paper.

Architecture:
  Input  : (batch, window_size, m_measurements)
  → transpose to (batch, m, window_size) for Conv1d (treat m as channels)
  → 3 × Conv1d blocks (filters: 32, 64, 128) with BN + ReLU + Dropout
  → Attention mechanism (eq. 9-10)
  → FC output → sigmoid (eq. 11)

Paper equations:
  F_l = φ(W_l * F_{l-1} + b_l)                  (8)
  α_i = exp(u_i) / Σ exp(u_j)  (softmax)         (9)
  u_i = wᵀ tanh(W_f f_i + b_f)                   (attention score)
  v   = Σ αᵢ fᵢ                                   (10)
  y   = σ(W_o v + b_o)                            (11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────────────────────
#  Attention module
# ─────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Bahdanau-style additive attention over temporal dimension.

    Input  : (batch, d, T)   feature maps from last conv layer
    Output : (batch, d)      context vector v = Σ αᵢ fᵢ
    """

    def __init__(self, d: int):
        super().__init__()
        self.W_f = nn.Linear(d, d, bias=True)   # W_f, b_f
        self.w   = nn.Linear(d, 1, bias=False)  # w

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        F : (batch, d, T)
        """
        # (batch, T, d)
        F_t = F.permute(0, 2, 1)
        # Attention scores u_i = wᵀ tanh(W_f f_i + b_f)
        u = self.w(torch.tanh(self.W_f(F_t)))     # (batch, T, 1)
        alpha = torch.softmax(u, dim=1)             # (batch, T, 1)
        # Context vector v = Σ αᵢ fᵢ
        v = (alpha * F_t).sum(dim=1)               # (batch, d)
        return v, alpha.squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────

class CNNAttentionDetector(nn.Module):
    """
    CNN + Attention FDIA detector.

    Parameters
    ----------
    m_measurements : number of measurement channels
    window_size    : temporal window length
    conv_filters   : list of filter counts per conv layer
    kernel_size    : convolution kernel size
    attention_dim  : d (output dim of last conv = attention_dim)
    dropout        : dropout probability
    """

    def __init__(
        self,
        m_measurements: int,
        window_size: int = 10,
        conv_filters: List[int] = None,
        kernel_size: int = 3,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128]

        self.m  = m_measurements
        self.w  = window_size

        # ── Convolutional layers ─────────────────────────────
        # Input: (batch, m, w)  — m as "channels", w as "length"
        conv_layers = []
        in_ch = m_measurements
        for out_ch in conv_filters:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_ch = out_ch
        self.convs = nn.Sequential(*conv_layers)

        # ── Attention over last conv output ──────────────────
        self.attention = TemporalAttention(d=in_ch)

        # ── Residual branch projection ────────────────────────
        # Takes residuals as secondary input (concatenated to context)
        # self.res_proj = nn.Sequential(
        #     nn.Linear(m_measurements * window_size, attention_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout),
        # )

        # ── Output layer (eq. 11) ────────────────────────────
        # Input: concat(context_v, res_features) → 1
        self.fc_out = nn.Linear(in_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window_size, m)   measurement window
        r : (batch, window_size, m)   residual window

        Returns
        -------
        y : (batch,)   attack probability ∈ [0, 1]
        """
        batch = x.size(0)

        # ── Conv branch on measurements ──────────────────────
        # (batch, m, w) — Conv1d expects (batch, channels, length)
        xc = x.permute(0, 2, 1)        # (batch, m, w)
        F  = self.convs(xc)             # (batch, last_filter, w)
        v, _ = self.attention(F)        # (batch, last_filter)

        # # ── Residual branch ──────────────────────────────────
        # r_flat = r.reshape(batch, -1)   # (batch, w*m)
        # r_feat = self.res_proj(r_flat)  # (batch, attention_dim)

        # ── Merge & classify ─────────────────────────────────
        # combined = torch.cat([v, r_feat], dim=-1)      # (batch, last_filter + attn_dim)
        logit    = self.fc_out(v).squeeze(-1)   # (batch,)
        y        = torch.sigmoid(logit)

        return y

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
