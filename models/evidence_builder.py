import numpy as np
import torch
from torch_geometric.utils import coalesce
from typing import List, Dict, Tuple


class EvidenceGraph:
    """
    Represents a single empirical evidence graph view.
    """
    def __init__(self, view_idx: int, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 active_nodes: np.ndarray, active_mask: torch.Tensor, num_total_nodes: int,
                 edge_attr: torch.Tensor = None):
        self.view_idx = view_idx
        self.edge_index = edge_index          # shape (2, E_m), LongTensor on device
        self.edge_weight = edge_weight        # shape (E_m, ), FloatTensor on device
        self.edge_attr = edge_attr            # shape (E_m, edge_feat_dim), FloatTensor on device
        self.active_nodes = active_nodes      # ndarray of active node IDs
        self.active_mask = active_mask        # boolean Tensor (num_total_nodes, ) on device
        self.num_total_nodes = num_total_nodes
        self.num_edges = edge_index.size(1)
        self.num_active_nodes = len(active_nodes)

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        self.active_mask = self.active_mask.to(device)
        return self


class EvidenceGraphBuilder:
    """
    Builds M empirical evidence graphs from observed historical interactions.
    Preserves all anchor backbone nodes across 100% of the views, and randomly samples
    peripheral nodes and observed empirical edges.
    Always generates edge_attr:
      - Attributed graphs: Coalesced raw multi-dimensional features (mean).
      - Pure topological graphs: Coalesced empirical interaction frequency (log1p).
    """
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                 anchor_node_ids: np.ndarray, num_total_nodes: int,
                 edge_ids: np.ndarray = None,
                 edge_raw_features: np.ndarray = None,
                 peripheral_sample_ratio: float = 0.60,
                 edge_sample_ratio: float = 1.0,
                 use_edge_weights: bool = True,
                 device: str = "cpu"):
        self.src = src_node_ids
        self.dst = dst_node_ids
        self.ts = node_interact_times
        self.edge_ids = edge_ids
        self.edge_raw_features = edge_raw_features
        self.has_edge_features = (edge_raw_features is not None) and (not np.all(edge_raw_features == 0))
        self.anchors = np.asarray(anchor_node_ids, dtype=np.longlong)
        self.num_total_nodes = num_total_nodes
        self.peripheral_sample_ratio = peripheral_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.use_edge_weights = use_edge_weights
        self.device = device

        all_train_nodes = np.unique(np.concatenate([self.src, self.dst]))
        self.peripheral_nodes = np.setdiff1d(all_train_nodes, self.anchors)

        print(f"[EvidenceGraphBuilder] Initialized:")
        print(f"  - Total train nodes: {len(all_train_nodes)}")
        print(f"  - Anchor backbone: {len(self.anchors)}")
        print(f"  - Peripheral node pool: {len(self.peripheral_nodes)}")
        print(f"  - Raw edge features active: {self.has_edge_features}")

    def sample_evidence_graphs(self, num_views: int = 4) -> Tuple[List[EvidenceGraph], torch.Tensor]:
        views = []
        presence_matrix = torch.zeros((num_views, self.num_total_nodes), dtype=torch.bool)
        num_peripherals_to_sample = int(len(self.peripheral_nodes) * self.peripheral_sample_ratio)

        for m in range(num_views):
            sampled_peripherals = np.random.choice(self.peripheral_nodes, size=num_peripherals_to_sample, replace=False)
            active_nodes = np.concatenate([self.anchors, sampled_peripherals])

            active_mask_np = np.zeros(self.num_total_nodes, dtype=bool)
            active_mask_np[active_nodes] = True
            active_mask = torch.from_numpy(active_mask_np).to(self.device)
            presence_matrix[m] = torch.from_numpy(active_mask_np)

            valid_edge_mask = active_mask_np[self.src] & active_mask_np[self.dst]
            src_m = self.src[valid_edge_mask]
            dst_m = self.dst[valid_edge_mask]
            edge_ids_m = self.edge_ids[valid_edge_mask] if (self.edge_ids is not None and self.has_edge_features) else None

            if self.edge_sample_ratio < 1.0:
                num_sampled_edges = int(len(src_m) * self.edge_sample_ratio)
                edge_perm = np.random.choice(len(src_m), size=num_sampled_edges, replace=False)
                src_m = src_m[edge_perm]
                dst_m = dst_m[edge_perm]
                if edge_ids_m is not None:
                    edge_ids_m = edge_ids_m[edge_perm]

            src_t = torch.from_numpy(src_m).long()
            dst_t = torch.from_numpy(dst_m).long()

            edge_index = torch.cat([torch.stack([src_t, dst_t]), torch.stack([dst_t, src_t])], dim=1)
            raw_weights = torch.ones(edge_index.size(1), dtype=torch.float32)

            if self.has_edge_features and edge_ids_m is not None:
                feat_forward = torch.from_numpy(self.edge_raw_features[edge_ids_m].astype(np.float32))
                edge_attr_raw = torch.cat([feat_forward, feat_forward], dim=0)
                edge_index_c, edge_attr = coalesce(edge_index, edge_attr_raw, num_nodes=self.num_total_nodes, reduce="mean")
                _, edge_weight = coalesce(edge_index, raw_weights, num_nodes=self.num_total_nodes, reduce="add")
                edge_index = edge_index_c
            else:
                edge_index, edge_weight = coalesce(edge_index, raw_weights, num_nodes=self.num_total_nodes, reduce="add")
                edge_attr = torch.log1p(edge_weight).unsqueeze(-1)

            if not self.use_edge_weights:
                edge_weight = torch.ones_like(edge_weight)

            view = EvidenceGraph(
                view_idx=m,
                edge_index=edge_index.to(self.device),
                edge_weight=edge_weight.to(self.device),
                active_nodes=active_nodes,
                active_mask=active_mask,
                num_total_nodes=self.num_total_nodes,
                edge_attr=edge_attr.to(self.device)
            )
            views.append(view)

        presence_matrix = presence_matrix.to(self.device)
        return views, presence_matrix

    def build_full_empirical_graph(self) -> EvidenceGraph:
        active_nodes = np.unique(np.concatenate([self.src, self.dst]))
        active_mask_np = np.zeros(self.num_total_nodes, dtype=bool)
        active_mask_np[active_nodes] = True
        active_mask = torch.from_numpy(active_mask_np).to(self.device)

        src_t = torch.from_numpy(self.src).long()
        dst_t = torch.from_numpy(self.dst).long()
        edge_index = torch.cat([torch.stack([src_t, dst_t]), torch.stack([dst_t, src_t])], dim=1)
        raw_weights = torch.ones(edge_index.size(1), dtype=torch.float32)

        if self.has_edge_features and self.edge_ids is not None:
            feat_forward = torch.from_numpy(self.edge_raw_features[self.edge_ids].astype(np.float32))
            edge_attr_raw = torch.cat([feat_forward, feat_forward], dim=0)
            edge_index_c, edge_attr = coalesce(edge_index, edge_attr_raw, num_nodes=self.num_total_nodes, reduce="mean")
            _, edge_weight = coalesce(edge_index, raw_weights, num_nodes=self.num_total_nodes, reduce="add")
            edge_index = edge_index_c
        else:
            edge_index, edge_weight = coalesce(edge_index, raw_weights, num_nodes=self.num_total_nodes, reduce="add")
            edge_attr = torch.log1p(edge_weight).unsqueeze(-1)

        if not self.use_edge_weights:
            edge_weight = torch.ones_like(edge_weight)

        return EvidenceGraph(
            view_idx=-1,
            edge_index=edge_index.to(self.device),
            edge_weight=edge_weight.to(self.device),
            active_nodes=active_nodes,
            active_mask=active_mask,
            num_total_nodes=self.num_total_nodes,
            edge_attr=edge_attr.to(self.device)
        )
