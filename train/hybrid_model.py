"""HybridDecoder: Pathfinder CNN backbone + sparse defect-GNN message-passing layer.

The CNN does local lattice-aware message passing (Pathfinder's DirectionalConv3d
on the full [T, H, W] grid). After the middle block, a GNN message-passing
layer operates on the KNN graph of the input syndrome's defect positions
(the ~3% of cells where syndrome=1), giving long-range defect-to-defect
information that the local CNN doesn't see. The GNN output is scattered
back to the spatial grid as a residual addition.

This is the architecture-level fusion of Pathfinder's CNN inductive bias
and Lange's GNN inductive bias in a single trainable model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch_cluster import knn_graph
from model import DecoderConfig, DirectionalConv3d, BottleneckBlock


class DefectGNNLayer(nn.Module):
    """One GraphConv-style message-passing layer over the KNN graph of defects.

    Input: feature map x [B, C, T, H, W] and the original input syndrome
    [B, 1, T, H, W]. Output: same shape as x, with GNN-updated features added
    at defect positions (zero elsewhere, so the residual leaves non-defect
    cells unchanged).
    """
    def __init__(self, channels: int, k: int = 10, edge_features: int = 4):
        super().__init__()
        self.channels = channels
        self.k = k
        self.w_self = nn.Linear(channels, channels, bias=False)
        # Message MLP: concatenates (feat_i, feat_j, pos_diff) and produces a message
        self.w_msg = nn.Linear(2 * channels + edge_features, channels, bias=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        [B, C, T, H, W] current CNN feature map
            syndrome: [B, 1, T, H, W] original input syndrome (defects = 1)
        Returns:
            [B, C, T, H, W] — x augmented with GNN-updated defects (residual)
        """
        B, C, T, H, W = x.shape
        assert C == self.channels

        # Find defect positions: nonzero cells in syndrome
        # syn_flat: [B, T*H*W] flattened binary syndrome
        syn_flat = syndrome.reshape(B, -1)
        # Indices of nonzero entries in the flat batch: [num_defects_total, 2] (batch, flat_idx)
        b_idx, flat_idx = torch.nonzero(syn_flat, as_tuple=True)  # both [N_def]
        if b_idx.numel() == 0:
            return x  # No defects in the batch — no GNN op needed

        # Unflatten flat_idx back to (t, h, w)
        t_idx = flat_idx // (H * W)
        hw = flat_idx % (H * W)
        h_idx = hw // W
        w_idx = hw % W

        # Gather features at defect positions: [N_def, C]
        # x[b, :, t, h, w] for each defect
        # x.permute(0,2,3,4,1) is [B, T, H, W, C] — index with [b, t, h, w]
        x_thwc = x.permute(0, 2, 3, 4, 1).contiguous()
        feat = x_thwc[b_idx, t_idx, h_idx, w_idx]  # [N_def, C]

        # Positions for KNN — use (t, h, w) normalized + a batch indicator handled by `batch` arg
        # Use float for KNN distance computation
        pos = torch.stack([t_idx.float(), h_idx.float(), w_idx.float()], dim=1)  # [N_def, 3]

        # Build KNN graph — `batch` argument keeps within-batch-element edges
        # Returns edge_index [2, num_edges]
        k_eff = min(self.k, max(1, feat.shape[0] - 1))
        edge_index = knn_graph(pos, k=k_eff, batch=b_idx, loop=False)
        # edge_index[0] = source (target's neighbor), edge_index[1] = target
        src, tgt = edge_index[0], edge_index[1]

        # Build messages: concat (feat_tgt, feat_src, pos_src - pos_tgt, 1.0)
        pos_diff = pos[src] - pos[tgt]                              # [E, 3]
        edge_feat = torch.cat([pos_diff, torch.ones(src.shape[0], 1, device=pos.device)], dim=1)  # [E, 4]
        msg_input = torch.cat([feat[tgt], feat[src], edge_feat], dim=1)  # [E, 2C + 4]
        msgs = self.w_msg(msg_input)  # [E, C]
        msgs = F.gelu(msgs)

        # Aggregate (sum) messages per target node
        agg = torch.zeros(feat.shape[0], self.channels, dtype=msgs.dtype, device=feat.device)  # [N_def, C]
        agg.index_add_(0, tgt, msgs)

        # Self transformation + aggregated messages
        new_feat = self.w_self(feat) + agg  # [N_def, C]

        # LayerNorm on the new features
        new_feat = self.norm(new_feat)

        # Scatter back to spatial grid as a residual delta
        # Build a delta tensor [B, T, H, W, C] zero everywhere except defect cells
        delta = torch.zeros_like(x_thwc)
        delta[b_idx, t_idx, h_idx, w_idx] = new_feat
        delta = delta.permute(0, 4, 1, 2, 3).contiguous()  # back to [B, C, T, H, W]

        return x + delta


class HybridDecoder(nn.Module):
    """Pathfinder CNN + sparse defect-GNN hybrid.

    Identical to Pathfinder up to the middle block. Then one DefectGNNLayer
    runs on the KNN graph of input-syndrome defects, adding a long-range
    message-passing signal as a residual. Remaining CNN blocks refine.
    """
    def __init__(self, config: DecoderConfig, gnn_k: int = 10):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        self.embed = nn.Conv3d(1, H, kernel_size=1, bias=True)
        L = config.n_blocks
        self.blocks_pre = nn.ModuleList([BottleneckBlock(H) for _ in range(L // 2)])
        self.gnn = DefectGNNLayer(channels=H, k=gnn_k)
        self.blocks_post = nn.ModuleList([BottleneckBlock(H) for _ in range(L - L // 2)])
        self.head = nn.Sequential(
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, config.n_observables),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        x = self.embed(syndrome)
        for block in self.blocks_pre:
            x = block(x)
        x = self.gnn(x, syndrome)         # GNN injection point
        for block in self.blocks_post:
            x = block(x)
        x = x.mean(dim=(2, 3, 4))
        return self.head(x)

    def predict(self, syndrome: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(syndrome) > 0
