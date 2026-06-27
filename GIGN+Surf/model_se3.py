import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GraphNorm,
    GlobalAttention
)

from torch_geometric.utils import scatter

# =========================================================
# EDGE DISTANCE
# =========================================================

def edge_distance(pos, edge_index):

    row, col = edge_index

    d = pos[row] - pos[col]

    dist = torch.norm(
        d,
        dim=1,
        keepdim=True
    )

    return dist


# =========================================================
# EGNN BLOCK
# =========================================================

class EGNNBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        dropout=0.2
    ):
        super().__init__()

        self.edge_mlp = nn.Sequential(

            nn.Linear(
                hidden_dim * 2 + 1,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.node_mlp = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.norm = GraphNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        pos,
        edge_index
    ):

        row, col = edge_index

        dist = edge_distance(
            pos,
            edge_index
        )

        edge_feat = torch.cat([
            x[row],
            x[col],
            dist
        ], dim=1)

        m_ij = self.edge_mlp(edge_feat)
        
        num_nodes = x.size(0)

        row = row.clamp(0, num_nodes - 1)   # safety guard

        agg = scatter(
            m_ij,
            row,
            dim=0,
            dim_size=num_nodes,
            reduce='sum'
        )

        node_input = torch.cat([
            x,
            agg
        ], dim=1)

        dx = self.node_mlp(node_input)

        x = x + dx

        x = self.norm(x)

        x = F.silu(x)

        x = self.dropout(x)

        return x


# =========================================================
# FULL MODEL
# =========================================================

class SurfaceEGNN(nn.Module):

    def __init__(
        self,
        node_dim=49,
        hidden_dim=256,
        num_layers=5,
        dropout=0.2
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            node_dim,
            hidden_dim
        )

        self.intra_layers = nn.ModuleList([
            EGNNBlock(
                hidden_dim,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.inter_layers = nn.ModuleList([
            EGNNBlock(
                hidden_dim,
                dropout
            )
            for _ in range(num_layers)
        ])

        # =====================================================
        # ATTENTION POOL
        # =====================================================

        gate_nn = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.SiLU(),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )

        self.pool = GlobalAttention(
            gate_nn
        )

        # =====================================================
        # FC HEAD
        # =====================================================

        self.fc1 = nn.Linear(
            hidden_dim,
            512
        )

        self.norm1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(
            512,
            256
        )

        self.norm2 = nn.LayerNorm(256)

        self.out = nn.Linear(
            256,
            1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):

        x = data.x

        pos = data.pos

        edge_intra = data.edge_index_intra

        edge_inter = data.edge_index_inter

        batch = data.batch

        # =====================================================
        # INPUT
        # =====================================================

        x = self.input_proj(x)

        # =====================================================
        # GEOMETRIC MESSAGE PASSING
        # =====================================================

        for intra_layer, inter_layer in zip(
            self.intra_layers,
            self.inter_layers
        ):

            x_intra = intra_layer(
                x,
                pos,
                edge_intra
            )

            x_inter = inter_layer(
                x,
                pos,
                edge_inter
            )

            x = x + x_intra + x_inter

        # =====================================================
        # GLOBAL POOL
        # =====================================================

        x = self.pool(
            x,
            batch
        )

        # =====================================================
        # FC
        # =====================================================

        x = self.fc1(x)

        x = self.norm1(x)

        x = F.silu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.norm2(x)

        x = F.silu(x)

        x = self.dropout(x)

        x = self.out(x)

        return x