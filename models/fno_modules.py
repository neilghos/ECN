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


class STL5HorizonSampler:
    """
    STL-Decomposition-guided Multi-Horizon Sampler and Physical Prior Extractor.
    Extracts chronological interaction sequences for 5 physical regimes:
      - Horizon 0: Seasonality View (Diurnal/periodic phase match to query time t % 86400)
      - Horizon 1: Trend View (Macroscopic decimation across life-cycle + anchor backbone)
      - Horizon 2: Short Bursts View (< 1 hour or 5% span rapid-fire clusters)
      - Horizon 3: Medium Bursts View (< 1-2 days or 20% span episodic surge)
      - Horizon 4: Long Bursts View (< 7-14 days or 50% span sustained surge)
    Additionally extracts an 8-dimensional handcrafted physical prior vector:
      [trend_velocity, diurnal_sin, diurnal_cos, weekly_sin, weekly_cos, fano_factor, recency_decay, activity_density]
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

    def sample_stl_5views(self, node_ids: np.ndarray,
                          node_interact_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        B = len(node_ids)
        L = self.seq_len

        h_nbrs = np.zeros((5, B, L), dtype=np.int64)
        h_edges = np.zeros((5, B, L), dtype=np.int64)
        h_times = np.zeros((5, B, L), dtype=np.float32)
        priors = np.zeros((B, 8), dtype=np.float32)

        for b in range(B):
            u = int(node_ids[b])
            t = float(node_interact_times[b])

            nbr_ids, edge_ids, nbr_times, _ = self.neighbor_sampler.find_neighbors_before(u, t)
            n = len(nbr_ids)

            # Query-level cyclical priors (valid even for inductive/isolated nodes)
            diurnal_sin = np.sin(2.0 * np.pi * (t % 86400.0) / 86400.0)
            diurnal_cos = np.cos(2.0 * np.pi * (t % 86400.0) / 86400.0)
            weekly_sin = np.sin(2.0 * np.pi * (t % 604800.0) / 604800.0)
            weekly_cos = np.cos(2.0 * np.pi * (t % 604800.0) / 604800.0)

            if n == 0:
                priors[b, 1] = diurnal_sin
                priors[b, 2] = diurnal_cos
                priors[b, 3] = weekly_sin
                priors[b, 4] = weekly_cos
                continue

            span = max(t - float(nbr_times[0]), 1.0)

            # --- Physical Priors (8-dim) ---
            t_mid = t - 0.5 * span
            i_mid = np.searchsorted(nbr_times, t_mid)
            n_past = i_mid
            n_recent = n - i_mid
            trend_vel = (n_recent - n_past) / (n + 1e-5)

            if n >= 2:
                dt_arr = np.diff(nbr_times)
                fano = np.log1p(np.var(dt_arr) / (np.mean(dt_arr) + 1e-5))
            else:
                fano = 0.0

            dt_last = max(t - float(nbr_times[-1]), 0.0)
            recency_decay = np.exp(-min(dt_last, 604800.0) / 86400.0)
            act_density = np.log1p(float(n))

            priors[b] = [trend_vel, diurnal_sin, diurnal_cos, weekly_sin, weekly_cos, fano, recency_decay, act_density]

            # --- View 0: Seasonality View (Diurnal Phase Alignment) ---
            if n <= L:
                h_nbrs[0, b, -n:] = nbr_ids
                h_edges[0, b, -n:] = edge_ids
                h_times[0, b, -n:] = nbr_times
            else:
                phi_nbr = nbr_times % 86400.0
                phi_q = t % 86400.0
                dphi = np.abs(phi_nbr - phi_q)
                phase_dist = np.minimum(dphi, 86400.0 - dphi)
                top_idx = np.argpartition(phase_dist, L)[:L]
                chosen0 = np.sort(top_idx)
                h_nbrs[0, b] = nbr_ids[chosen0]
                h_edges[0, b] = edge_ids[chosen0]
                h_times[0, b] = nbr_times[chosen0]

            # --- View 1: Trend View (Macroscopic Decimation + Anchor Backbone) ---
            if n <= L:
                h_nbrs[1, b, -n:] = nbr_ids
                h_edges[1, b, -n:] = edge_ids
                h_times[1, b, -n:] = nbr_times
            else:
                safe_nbrs = np.clip(nbr_ids, 0, len(self.anchor_mask) - 1)
                is_anc = self.anchor_mask[safe_nbrs]
                idx_anc = np.where(is_anc)[0]
                n_anc = len(idx_anc)
                if n_anc >= L:
                    chosen1 = idx_anc[-L:]
                elif n_anc > 0:
                    rem = L - n_anc
                    idx_non = np.where(~is_anc)[0]
                    step = np.linspace(0, len(idx_non) - 1, min(rem, len(idx_non)), dtype=int)
                    chosen1 = np.sort(np.concatenate([idx_anc, idx_non[step]]))
                else:
                    chosen1 = np.linspace(0, n - 1, L, dtype=int)
                k1 = len(chosen1)
                h_nbrs[1, b, -k1:] = nbr_ids[chosen1]
                h_edges[1, b, -k1:] = edge_ids[chosen1]
                h_times[1, b, -k1:] = nbr_times[chosen1]

            # --- View 2: Short Bursts View (< 1 hour or 5% span) ---
            t_short = t - min(max(0.05 * span, 60.0), 3600.0)
            i2 = np.searchsorted(nbr_times, t_short)
            m2 = n - i2
            if m2 > L:
                chosen2 = np.linspace(i2, n - 1, L, dtype=int)
            elif m2 > 0:
                chosen2 = np.arange(i2, n)
            else:
                k2 = min(n, L)
                chosen2 = np.arange(n - k2, n)
            k2 = len(chosen2)
            h_nbrs[2, b, -k2:] = nbr_ids[chosen2]
            h_edges[2, b, -k2:] = edge_ids[chosen2]
            h_times[2, b, -k2:] = nbr_times[chosen2]

            # --- View 3: Medium Bursts View (< 1-2 days or 20% span) ---
            t_med = t - min(max(0.20 * span, 3600.0), 86400.0 * 2)
            i3 = np.searchsorted(nbr_times, t_med)
            m3 = n - i3
            if m3 > L:
                chosen3 = np.linspace(i3, n - 1, L, dtype=int)
            elif m3 > 0:
                chosen3 = np.arange(i3, n)
            else:
                k3 = min(n, L)
                chosen3 = np.arange(n - k3, n)
            k3 = len(chosen3)
            h_nbrs[3, b, -k3:] = nbr_ids[chosen3]
            h_edges[3, b, -k3:] = edge_ids[chosen3]
            h_times[3, b, -k3:] = nbr_times[chosen3]

            # --- View 4: Long Bursts View (< 7 days or 50% span) ---
            t_long = t - min(max(0.50 * span, 86400.0), 86400.0 * 7)
            i4 = np.searchsorted(nbr_times, t_long)
            m4 = n - i4
            if m4 > L:
                chosen4 = np.linspace(i4, n - 1, L, dtype=int)
            elif m4 > 0:
                chosen4 = np.arange(i4, n)
            else:
                k4 = min(n, L)
                chosen4 = np.arange(n - k4, n)
            k4 = len(chosen4)
            h_nbrs[4, b, -k4:] = nbr_ids[chosen4]
            h_edges[4, b, -k4:] = edge_ids[chosen4]
            h_times[4, b, -k4:] = nbr_times[chosen4]

        return h_nbrs, h_edges, h_times, priors

    def sample_multiscale_horizons(self, node_ids: np.ndarray,
                                   node_interact_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backwards compatibility helper returning 3 horizons."""
        h_nbrs, h_edges, h_times, _ = self.sample_stl_5views(node_ids, node_interact_times)
        return h_nbrs[:3], h_edges[:3], h_times[:3]


# Backwards compatibility alias
MultiHorizonSampler = STL5HorizonSampler
