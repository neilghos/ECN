import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from utils.utils import find_anchor_nodes
from utils.DataLoader import Data
from models.evidence_builder import EvidenceGraph, EvidenceGraphBuilder


class GINEEncoder(nn.Module):
    """
    GINE message passing encoder with residual LayerNorm connections and multi-scale layer readout.
    Computes:
    h_v^(k) = MLP^(k) ( (1 + eps) * h_v^(k-1) + sum_{u in N(v)} ReLU(h_u^(k-1) + W_e * e_{vu}) )
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super(GINEEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for l in range(num_layers):
            in_dim = node_dim if l == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(mlp, edge_dim=edge_dim, train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        layer_embeddings = [h]
        for l, conv in enumerate(self.convs):
            h_next = conv(h, edge_index, edge_attr)
            h = self.norms[l](h_next + h) if h.shape == h_next.shape else self.norms[l](h_next)
            h = self.dropout(h)
            layer_embeddings.append(h)
        return torch.stack(layer_embeddings, dim=0).mean(dim=0)


class EvidGNN(nn.Module):
    """
    EvidGNN: Evidence Graph Neural Network with GINE as the base encoder.
    Unified across both attributed graphs and pure topological graphs.
    """
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 neighbor_sampler = None,
                 train_data: Data = None,
                 src_node_ids: np.ndarray = None, dst_node_ids: np.ndarray = None, node_interact_times: np.ndarray = None,
                 edge_ids: np.ndarray = None,
                 time_feat_dim: int = 100,
                 embedding_dim: int = 172,
                 num_layers: int = 2,
                 num_evidence_graphs: int = 4,
                 peripheral_sample_ratio: float = 0.60,
                 edge_sample_ratio: float = 1.0,
                 num_time_windows: int = 20,
                 top_k_ratio: float = 0.10,
                 anchor_strategy: str = "composite",
                 alpha: float = 0.5,
                 temperature: float = 0.2,
                 contrastive_weight: float = 0.1,
                 dropout: float = 0.1,
                 device: str = "cpu",
                 **kwargs):
        super(EvidGNN, self).__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_evidence_graphs = num_evidence_graphs
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.current_contrastive_loss = torch.tensor(0.0, device=device)
        self.neighbor_sampler = neighbor_sampler
        self._cached_step_embeddings = None
        self._cached_eval_embeddings = None

        # 1. Retrieve empirical training pool
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
            raise ValueError("Must provide either train_data (Data object) or (src_node_ids, dst_node_ids, node_interact_times)!")

        self.num_total_nodes = max(int(self.train_src.max()), int(self.train_dst.max())) + 1
        if node_raw_features is not None:
            self.num_total_nodes = max(self.num_total_nodes, node_raw_features.shape[0])

        # Track seen training nodes for inductive masking
        self.train_node_set = set(self.train_src).union(self.train_dst)
        self.is_train_node = torch.zeros(self.num_total_nodes, dtype=torch.bool, device=self.device)
        train_nodes_arr = np.array(list(self.train_node_set), dtype=np.int64)
        self.is_train_node[torch.from_numpy(train_nodes_arr).long().to(self.device)] = True

        # Check feature presence
        self.has_edge_features = (edge_raw_features is not None) and (not np.all(edge_raw_features == 0))
        self.has_node_features = (node_raw_features is not None) and (not np.all(node_raw_features == 0))

        # Edge feature dimension: raw features dimension if available, else 1 (empirical frequency)
        self.edge_feat_dim = edge_raw_features.shape[1] if self.has_edge_features else 1
        print(f"[EvidGNN] Initialized GINE Base Encoder | Node Feat Dim: {embedding_dim} | Edge Feat Dim: {self.edge_feat_dim} (Raw: {self.has_edge_features})")

        # 2. Identify persistent backbone anchors
        self.anchor_node_ids, self.anchor_stats = find_anchor_nodes(
            src_node_ids=self.train_src,
            dst_node_ids=self.train_dst,
            node_interact_times=self.train_ts,
            num_time_windows=num_time_windows,
            top_k_ratio=top_k_ratio,
            strategy=anchor_strategy,
            alpha=alpha
        )
        self.register_buffer("anchors", torch.from_numpy(self.anchor_node_ids).long())

        # 3. Setup Evidence Graph Builder
        self.builder = EvidenceGraphBuilder(
            src_node_ids=self.train_src,
            dst_node_ids=self.train_dst,
            node_interact_times=self.train_ts,
            anchor_node_ids=self.anchor_node_ids,
            num_total_nodes=self.num_total_nodes,
            edge_ids=self.train_edge_ids,
            edge_raw_features=edge_raw_features,
            peripheral_sample_ratio=peripheral_sample_ratio,
            edge_sample_ratio=edge_sample_ratio,
            device=device
        )

        # 4. Input node representations
        if not self.has_node_features:
            self.node_embedding = nn.Embedding(self.num_total_nodes, embedding_dim)
            nn.init.normal_(self.node_embedding.weight, std=0.01)
        else:
            self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
            self.node_proj = nn.Linear(node_raw_features.shape[1], embedding_dim)

        # 5. Build GINE Encoder Backbone
        self.encoder = GINEEncoder(
            node_dim=embedding_dim,
            edge_dim=self.edge_feat_dim,
            hidden_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # 6. Contrastive Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Pre-build full empirical graph for deterministic inference
        self.full_graph = self.builder.build_full_empirical_graph().to(device)

        # Sample initial evidence graphs
        self.resample_evidence_graphs()

    def set_neighbor_sampler(self, neighbor_sampler):
        """Duck-type compatibility for DyGLib neighbor sampler interface"""
        self.neighbor_sampler = neighbor_sampler

    def resample_evidence_graphs(self):
        """
        Sample M fresh evidence graphs for the upcoming epoch.
        Preserves 100% of anchors and samples fresh peripheral empirical subgraphs.
        """
        self.current_views, self.current_presence = self.builder.sample_evidence_graphs(
            num_views=self.num_evidence_graphs
        )

    def get_initial_node_features(self) -> torch.Tensor:
        if not self.has_node_features:
            return self.node_embedding.weight
        else:
            return self.node_proj(self.node_raw_features)

    def encode_view(self, x: torch.Tensor, view: EvidenceGraph) -> torch.Tensor:
        return self.encoder(x, view.edge_index, view.edge_attr)

    def compute_existence_masked_contrastive_loss(self, view_projections: list, batch_node_ids: torch.Tensor) -> torch.Tensor:
        if len(batch_node_ids) == 0:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        num_pairs = 0
        B = len(batch_node_ids)

        for i in range(len(view_projections)):
            for j in range(i + 1, len(view_projections)):
                valid_i = self.current_presence[i, batch_node_ids]
                valid_j = self.current_presence[j, batch_node_ids]
                valid_pos = valid_i & valid_j

                if valid_pos.sum() < 2:
                    continue

                p_i = view_projections[i][batch_node_ids]
                p_j = view_projections[j][batch_node_ids]

                sim = torch.mm(p_i, p_j.t()) / self.temperature
                sim = sim.masked_fill(~valid_j.unsqueeze(0), -1e9)

                targets = torch.arange(B, device=self.device)
                losses = F.cross_entropy(sim, targets, reduction="none")

                pair_loss = (losses * valid_pos.float()).sum() / (valid_pos.float().sum() + 1e-8)
                total_loss += pair_loss
                num_pairs += 1

        if num_pairs > 0:
            return total_loss / num_pairs
        return torch.tensor(0.0, device=self.device)

    def _extract_node_embeddings(self, node_ids: np.ndarray, interact_times: np.ndarray,
                                  all_node_embeddings: torch.Tensor) -> torch.Tensor:
        node_tensor = torch.from_numpy(node_ids).long().to(self.device)
        node_safe = torch.clamp(node_tensor, 0, self.num_total_nodes - 1)
        embeddings = all_node_embeddings[node_safe].clone()

        is_unseen = (node_tensor >= self.num_total_nodes) | (~self.is_train_node[node_safe])

        if is_unseen.any():
            unseen_mask_np = is_unseen.cpu().numpy()
            unseen_indices = np.where(unseen_mask_np)[0]

            if self.neighbor_sampler is not None and len(unseen_indices) > 0:
                unseen_ids = node_ids[unseen_indices]
                unseen_ts = interact_times[unseen_indices]
                nbr_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
                    unseen_ids, unseen_ts, num_neighbors=10
                )
                nbr_tensor = torch.from_numpy(nbr_ids).long().to(self.device)
                valid_nbr_mask = (nbr_tensor > 0) & (nbr_tensor < self.num_total_nodes)
                nbr_safe = torch.clamp(nbr_tensor, 0, self.num_total_nodes - 1)

                nbr_embeds = all_node_embeddings[nbr_safe] * valid_nbr_mask.unsqueeze(-1).float()
                counts = valid_nbr_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                has_nbrs = valid_nbr_mask.any(dim=1, keepdim=True)
                inductive_embeds = (nbr_embeds.sum(dim=1) / counts) * has_nbrs.float()

                embeddings[unseen_indices] = inductive_embeds
            else:
                embeddings[is_unseen] = 0.0

        return embeddings

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, **kwargs):
        x = self.get_initial_node_features()

        src_tensor = torch.from_numpy(src_node_ids).long().to(self.device)
        dst_tensor = torch.from_numpy(dst_node_ids).long().to(self.device)

        max_valid_id = self.num_total_nodes - 1
        src_safe = torch.clamp(src_tensor, 0, max_valid_id)
        dst_safe = torch.clamp(dst_tensor, 0, max_valid_id)

        if self.training:
            if self._cached_step_embeddings is None:
                view_embeds = []
                view_projs = []

                for view in self.current_views:
                    z_m = self.encode_view(x, view)
                    p_m = F.normalize(self.proj_head(z_m), dim=-1)
                    view_embeds.append(z_m)
                    view_projs.append(p_m)

                batch_unique_nodes = torch.unique(torch.cat([src_safe, dst_safe]))
                self.current_contrastive_loss = self.compute_existence_masked_contrastive_loss(
                    view_projs, batch_unique_nodes
                )

                self._cached_step_embeddings = torch.stack(view_embeds, dim=0).mean(dim=0)
                all_node_embeddings = self._cached_step_embeddings
            else:
                all_node_embeddings = self._cached_step_embeddings
                self._cached_step_embeddings = None
        else:
            if self._cached_eval_embeddings is None:
                with torch.no_grad():
                    self._cached_eval_embeddings = self.encode_view(x, self.full_graph)
            all_node_embeddings = self._cached_eval_embeddings
            self.current_contrastive_loss = torch.tensor(0.0, device=self.device)

        src_embeddings = self._extract_node_embeddings(src_node_ids, node_interact_times, all_node_embeddings)
        dst_embeddings = self._extract_node_embeddings(dst_node_ids, node_interact_times, all_node_embeddings)

        return src_embeddings, dst_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray, **kwargs):
        x = self.get_initial_node_features()
        if self.training:
            view_embeds = [self.encode_view(x, v) for v in self.current_views]
            all_embeddings = torch.stack(view_embeds, dim=0).mean(dim=0)
        else:
            if self._cached_eval_embeddings is None:
                with torch.no_grad():
                    self._cached_eval_embeddings = self.encode_view(x, self.full_graph)
            all_embeddings = self._cached_eval_embeddings

        return self._extract_node_embeddings(node_ids, node_interact_times, all_embeddings)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_eval_embeddings = None
        return self

    def reset_step_cache(self):
        self._cached_step_embeddings = None
        self._cached_eval_embeddings = None

    def get_contrastive_loss(self) -> torch.Tensor:
        return self.current_contrastive_loss
