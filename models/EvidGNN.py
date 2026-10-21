import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import find_anchor_nodes
from utils.DataLoader import Data
from models.modules import TimeEncoder
from models.fno_modules import FNO1DBlock, STL5HorizonSampler


class EvidGNN(nn.Module):
    """
    STL-Decomposition-Guided Dynamic Graph Transformer with Fourier Neural Operators.

    Architecture:
      1. STL Physical Prior Extractor:
         Extracts 8-dim physical priors (trend velocity, diurnal phase, weekly phase,
         Fano burstiness, recency decay, activity density).
      2. 5-Horizon Physical Partitioner:
         Partitions interaction streams into 5 physical regimes:
           - Horizon 0: Seasonality View (Diurnal/periodic phase alignment to query time)
           - Horizon 1: Trend View (Macroscopic decimation across life-cycle + anchor backbone)
           - Horizon 2: Short Bursts View (< 1 hour or 5% span rapid-fire clusters)
           - Horizon 3: Medium Bursts View (< 1-2 days or 20% span episodic surges)
           - Horizon 4: Long Bursts View (< 7-14 days or 50% span sustained surges)
      3. FNO-1D Spectral Operators:
         Continuous Fourier frequency filtering per horizon via complex mode multiplication.
      4. Dynamic Graph Transformer:
         Multi-Head Self-Attention over the 5 physical horizon tokens, conditioned on the
         STL physical priors and horizon positional embeddings.
      5. Multi-Scale Contrastive Learning:
         InfoNCE across canonical scale pairs (Season-Trend, Trend-Long, Short-Med, Med-Long, Season-Short).
      6. Output Node Representation:
         Fusion into node_feat_dim for downstream MergeLayer link prediction.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 neighbor_sampler = None,
                 train_data: Data = None,
                 src_node_ids: np.ndarray = None, dst_node_ids: np.ndarray = None, node_interact_times: np.ndarray = None,
                 edge_ids: np.ndarray = None,
                 time_feat_dim: int = 100,
                 embedding_dim: int = 172,
                 hidden_dim: int = 128,
                 seq_len: int = 32,
                 fno_modes: int = 16,
                 num_layers: int = 2,
                 num_time_windows: int = 20,
                 top_k_ratio: float = 0.10,
                 anchor_strategy: str = "composite",
                 alpha: float = 0.5,
                 temperature: float = 0.2,
                 contrastive_weight: float = 0.1,
                 dropout: float = 0.1,
                 transformer_layers: int = 1,
                 transformer_heads: int = 4,
                 device: str = "cpu",
                 **kwargs):
        super(EvidGNN, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.fno_modes = fno_modes
        self.num_layers = num_layers
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.current_contrastive_loss = torch.tensor(0.0, device=device)

        # 1. Retrieve empirical training data for anchor detection
        if train_data is not None:
            self.train_src = train_data.src_node_ids
            self.train_dst = train_data.dst_node_ids
            self.train_ts = train_data.node_interact_times
            self.train_edge_ids = train_data.edge_ids
        elif src_node_ids is not None and dst_node_ids is not None and node_interact_times is not None:
            self.train_src = src_node_ids
            self.train_dst = dst_node_ids
            self.train_ts = node_interact_times
            self.train_edge_ids = edge_ids
        else:
            raise ValueError("Must provide train_data or interaction arrays!")

        self.num_total_nodes = max(int(self.train_src.max()), int(self.train_dst.max())) + 1
        if node_raw_features is not None:
            self.num_total_nodes = max(self.num_total_nodes, node_raw_features.shape[0])

        self.has_edge_features = (edge_raw_features is not None) and (not np.all(edge_raw_features == 0))
        self.has_node_features = (node_raw_features is not None) and (not np.all(node_raw_features == 0))

        self.node_feat_dim = node_raw_features.shape[1] if node_raw_features is not None else embedding_dim
        self.edge_feat_dim = edge_raw_features.shape[1] if self.has_edge_features else 0

        print(f"[EvidGNN (Dynamic Graph Transformer + FNO-1D)] Initialized STL-Guided Architecture | Node Dim: {self.node_feat_dim} | Edge Dim: {self.edge_feat_dim} | Hidden Dim: {self.hidden_dim} | Modes: {self.fno_modes} | Transformer Heads: {transformer_heads}")

        # 2. Identify persistent backbone anchors
        self.anchor_node_ids, _ = find_anchor_nodes(
            src_node_ids=self.train_src,
            dst_node_ids=self.train_dst,
            node_interact_times=self.train_ts,
            num_time_windows=num_time_windows,
            top_k_ratio=top_k_ratio,
            strategy=anchor_strategy,
            alpha=alpha
        )
        print(f"  - Anchor backbone nodes identified: {len(self.anchor_node_ids)}")

        # 3. Setup STL 5-Horizon Sampler & Physical Prior Extractor
        self.neighbor_sampler = neighbor_sampler
        self.horizon_sampler = STL5HorizonSampler(
            neighbor_sampler=neighbor_sampler,
            anchor_node_ids=self.anchor_node_ids,
            num_total_nodes=self.num_total_nodes,
            seq_len=self.seq_len
        )

        # 4. Feature projection layers
        if self.has_node_features:
            self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
            self.node_proj = nn.Linear(self.node_feat_dim, self.hidden_dim)
        else:
            self.node_embedding = nn.Embedding(self.num_total_nodes + 1, self.hidden_dim)
            nn.init.normal_(self.node_embedding.weight, std=0.01)

        if self.has_edge_features:
            self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
            self.edge_proj = nn.Linear(self.edge_feat_dim, self.hidden_dim)
        else:
            self.edge_proj = None

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.time_proj = nn.Linear(time_feat_dim, self.hidden_dim)

        # 5. Fourier Neural Operator (FNO-1D) Spectral Layers
        self.fno_blocks = nn.Sequential(*[
            FNO1DBlock(hidden_dim=self.hidden_dim, modes=self.fno_modes, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        # 6. STL Physical Prior Projector and Horizon Positional Embeddings
        self.prior_proj = nn.Linear(8, self.hidden_dim)
        self.horizon_pos_emb = nn.Embedding(5, self.hidden_dim)

        # 7. Dynamic Graph Transformer (Self-Attention over the 5 Physical View Tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=2 * self.hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 8. Multi-Horizon Fusion Layer: [Attended 5 Tokens] -> Output Node Representation
        self.fusion = nn.Sequential(
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.node_feat_dim)
        )

        # 9. Contrastive Projection Head for Multi-Horizon InfoNCE
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        self.horizon_sampler.set_neighbor_sampler(neighbor_sampler)

    def resample_evidence_graphs(self):
        """Duck-type compatibility for DyGLib training epoch hooks."""
        pass

    def reset_step_cache(self, force_eval_reset: bool = False):
        self.current_contrastive_loss = torch.tensor(0.0, device=self.device)

    def get_contrastive_loss(self) -> torch.Tensor:
        return self.current_contrastive_loss

    def _encode_node_horizons(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Extracts 5 physical evidence horizons and passes each through the FNO-1D operator.
        Returns:
            tokens: Tensor of shape (Batch, 5, hidden_dim)
            priors: Tensor of shape (Batch, 8)
        """
        B = len(node_ids)
        h_nbrs, h_edges, h_times, priors_np = self.horizon_sampler.sample_stl_5views(node_ids, node_interact_times)
        priors = torch.from_numpy(priors_np).float().to(self.device)

        # Target node intrinsic embedding (for bypass & inductive nodes)
        target_tensor = torch.from_numpy(node_ids).long().to(self.device)
        if self.has_node_features:
            target_safe = torch.clamp(target_tensor, 0, self.node_raw_features.shape[0] - 1)
            target_emb = self.node_proj(self.node_raw_features[target_safe])
        else:
            target_safe = torch.clamp(target_tensor, 0, self.num_total_nodes)
            target_emb = self.node_embedding(target_safe)

        query_ts = torch.from_numpy(node_interact_times).float().to(self.device).unsqueeze(1) # (B, 1)

        view_tokens = []
        for m in range(5):
            nbr_t = torch.from_numpy(h_nbrs[m]).long().to(self.device)   # (B, L)
            edge_t = torch.from_numpy(h_edges[m]).long().to(self.device) # (B, L)
            time_t = torch.from_numpy(h_times[m]).float().to(self.device) # (B, L)

            valid_mask = (nbr_t > 0) # (B, L)

            if self.has_node_features:
                nbr_safe = torch.clamp(nbr_t, 0, self.node_raw_features.shape[0] - 1)
                node_emb = self.node_proj(self.node_raw_features[nbr_safe])
            else:
                nbr_safe = torch.clamp(nbr_t, 0, self.num_total_nodes)
                node_emb = self.node_embedding(nbr_safe)

            if self.has_edge_features and self.edge_proj is not None:
                edge_safe = torch.clamp(edge_t, 0, self.edge_raw_features.shape[0] - 1)
                edge_emb = self.edge_proj(self.edge_raw_features[edge_safe])
            else:
                edge_emb = 0.0

            dt = torch.clamp(query_ts - time_t, min=0.0) # (B, L)
            time_emb = self.time_proj(self.time_encoder(dt)) # (B, L, hidden_dim)

            x = node_emb + edge_emb + time_emb
            x = x * valid_mask.unsqueeze(-1).float()

            # Permute for 1D Fourier Neural Operator: (B, hidden_dim, L)
            x_fno = x.permute(0, 2, 1)
            h = self.fno_blocks(x_fno) # (B, hidden_dim, L)

            # Masked temporal pooling over L
            counts = valid_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0) # (B, 1)
            h_masked = h * valid_mask.unsqueeze(1).float()
            z_m = h_masked.sum(dim=-1) / counts # (B, hidden_dim)

            # Residual target node identity bypass
            has_history = valid_mask.any(dim=-1, keepdim=True).float() # (B, 1)
            z_m = (z_m * has_history) + target_emb

            view_tokens.append(z_m)

        tokens = torch.stack(view_tokens, dim=1) # (B, 5, hidden_dim)
        return tokens, priors

    def _attend_and_fuse(self, tokens: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """
        Conditions the 5 physical view tokens with STL physical priors and horizon positional embeddings,
        passes through the Dynamic Graph Transformer Self-Attention block, and fuses to node_feat_dim.
        """
        B = tokens.size(0)
        prior_emb = self.prior_proj(priors) # (B, hidden_dim)
        pos = self.horizon_pos_emb(torch.arange(5, device=self.device)) # (5, hidden_dim)

        # Condition tokens with horizon semantics + physical priors
        x = tokens + pos.unsqueeze(0) + prior_emb.unsqueeze(1) # (B, 5, hidden_dim)

        # Dynamic Graph Transformer Self-Attention
        tokens_attended = self.transformer(x) # (B, 5, hidden_dim)

        # Fusion to output node representation
        out = self.fusion(tokens_attended.reshape(B, 5 * self.hidden_dim)) # (B, node_feat_dim)
        return out

    def compute_multihorizon_contrastive_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes symmetric InfoNCE contrastive loss across canonical physical scale pairs:
          - (0, 1): Seasonality vs Trend
          - (1, 4): Trend vs Long Burst
          - (2, 3): Short vs Medium Burst
          - (3, 4): Medium vs Long Burst
          - (0, 2): Seasonality vs Short Burst
        """
        B = tokens.size(0)
        if B < 2:
            return torch.tensor(0.0, device=self.device)

        p = [F.normalize(self.proj_head(tokens[:, m]), dim=-1) for m in range(5)]
        targets = torch.arange(B, device=self.device)

        def pair_infonce(a, b):
            sim_ab = torch.mm(a, b.t()) / self.temperature
            sim_ba = torch.mm(b, a.t()) / self.temperature
            return 0.5 * (F.cross_entropy(sim_ab, targets) + F.cross_entropy(sim_ba, targets))

        loss = (pair_infonce(p[0], p[1]) +
                pair_infonce(p[1], p[4]) +
                pair_infonce(p[2], p[3]) +
                pair_infonce(p[3], p[4]) +
                pair_infonce(p[0], p[2])) / 5.0
        return loss

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, **kwargs):
        src_tokens, src_priors = self._encode_node_horizons(src_node_ids, node_interact_times)
        dst_tokens, dst_priors = self._encode_node_horizons(dst_node_ids, node_interact_times)

        if self.training:
            src_cl = self.compute_multihorizon_contrastive_loss(src_tokens)
            dst_cl = self.compute_multihorizon_contrastive_loss(dst_tokens)
            self.current_contrastive_loss = 0.5 * (src_cl + dst_cl)
        else:
            self.current_contrastive_loss = torch.tensor(0.0, device=self.device)

        src_embeddings = self._attend_and_fuse(src_tokens, src_priors)
        dst_embeddings = self._attend_and_fuse(dst_tokens, dst_priors)

        return src_embeddings, dst_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray, **kwargs):
        tokens, priors = self._encode_node_horizons(node_ids, node_interact_times)
        return self._attend_and_fuse(tokens, priors)
