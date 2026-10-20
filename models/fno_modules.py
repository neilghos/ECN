import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Set


class SpectralConv1d(nn.Module):
    """
    1D Fourier Neural Operator spectral convolution.
    Applies parameterized complex weights in the Fourier frequency domain.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, in_channels, Seq_Len)
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        k_modes = min(self.modes, x_ft.size(-1))

        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :k_modes] = torch.einsum('bix, iox -> box', x_ft[:, :, :k_modes], self.weights[:, :, :k_modes])

        return torch.fft.irfft(out_ft, n=L, dim=-1)


class FNO1DBlock(nn.Module):
    """
    A single FNO-1D block:
    x -> SpectralConv1d(x) + Conv1d_bypass(x) -> GELU -> Dropout -> LayerNorm -> Residual (+ x)
    """
    def __init__(self, hidden_dim: int, modes: int = 12, dropout: float = 0.1):
        super(FNO1DBlock, self).__init__()
        self.spectral_conv = SpectralConv1d(hidden_dim, hidden_dim, modes)
        self.w = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden_dim, L)
        x_spec = self.spectral_conv(x)
        x_local = self.w(x)
        h = self.act(x_spec + x_local)
        h = self.dropout(h)
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        return h + x


class MultiHorizonSampler:
    """
    Extracts chronological interaction sequences for three evidence horizons:
      - Horizon 0 (Burst): Most recent 10% of historical time
      - Horizon 1 (Trend): Intermediate 30% of historical time
      - Horizon 2 (Anchor Backbone): Entire history with persistent anchor prioritization
    """
    def __init__(self, neighbor_sampler, anchor_node_ids: np.ndarray, num_total_nodes: int = 100000, seq_len: int = 32):
        self.neighbor_sampler = neighbor_sampler
        self.anchors_arr = np.asarray(anchor_node_ids, dtype=np.int64)
        self.seq_len = seq_len
        self.anchor_mask = np.zeros(num_total_nodes + 2, dtype=bool)
        if len(self.anchors_arr) > 0:
            valid_anchors = self.anchors_arr[self.anchors_arr < len(self.anchor_mask)]
            self.anchor_mask[valid_anchors] = True

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler

    def sample_multiscale_horizons(self, node_ids: np.ndarray,
                                   node_interact_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = len(node_ids)
        L = self.seq_len

        h_nbrs = np.zeros((3, B, L), dtype=np.int64)
        h_edges = np.zeros((3, B, L), dtype=np.int64)
        h_times = np.zeros((3, B, L), dtype=np.float32)

        for b in range(B):
            u = int(node_ids[b])
            t = float(node_interact_times[b])
            nbr_ids, edge_ids, nbr_times, _ = self.neighbor_sampler.find_neighbors_before(u, t)
            n = len(nbr_ids)

            if n == 0:
                continue

            span = max(t - float(nbr_times[0]), 1e-4)

            # Horizon 0: Short-term burst (last 10% of time)
            t_burst = t - 0.10 * span
            i0 = np.searchsorted(nbr_times, t_burst)
            k0 = min(n - i0, L)
            if k0 > 0:
                h_nbrs[0, b, -k0:] = nbr_ids[n - k0:]
                h_edges[0, b, -k0:] = edge_ids[n - k0:]
                h_times[0, b, -k0:] = nbr_times[n - k0:]

            # Horizon 1: Medium-term trend (last 30% of time)
            t_trend = t - 0.30 * span
            i1 = np.searchsorted(nbr_times, t_trend)
            m1 = n - i1
            if m1 > L:
                idx = np.linspace(i1, n - 1, L, dtype=int)
                h_nbrs[1, b] = nbr_ids[idx]
                h_edges[1, b] = edge_ids[idx]
                h_times[1, b] = nbr_times[idx]
            elif m1 > 0:
                h_nbrs[1, b, -m1:] = nbr_ids[i1:]
                h_edges[1, b, -m1:] = edge_ids[i1:]
                h_times[1, b, -m1:] = nbr_times[i1:]

            # Horizon 2: Long-term anchor backbone (entire history)
            if n > L:
                is_anc = self.anchor_mask[np.clip(nbr_ids, 0, len(self.anchor_mask) - 1)]
                idx_anc = np.where(is_anc)[0]
                n_anc = len(idx_anc)
                if n_anc >= L:
                    chosen = idx_anc[-L:]
                else:
                    rem = L - n_anc
                    idx_non = np.where(~is_anc)[0]
                    if len(idx_non) > 0:
                        step = np.linspace(0, len(idx_non) - 1, min(rem, len(idx_non)), dtype=int)
                        chosen = np.sort(np.concatenate([idx_anc, idx_non[step]]))
                    else:
                        chosen = idx_anc
                k2 = len(chosen)
                h_nbrs[2, b, -k2:] = nbr_ids[chosen]
                h_edges[2, b, -k2:] = edge_ids[chosen]
                h_times[2, b, -k2:] = nbr_times[chosen]
            else:
                h_nbrs[2, b, -n:] = nbr_ids
                h_edges[2, b, -n:] = edge_ids
                h_times[2, b, -n:] = nbr_times

        return h_nbrs, h_edges, h_times
