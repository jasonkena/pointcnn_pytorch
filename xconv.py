# adapted from https://github.com/pyg-team/pytorch_geometric/issues/1470
# and https://github.com/pyg-team/pytorch_geometric/blob/f0e91ade196f88ea343baac2525bfe3817eda370/torch_geometric/nn/conv/x_conv.py

from math import ceil
from typing import Optional

import torch
from torch import Tensor
from torch.nn import ELU
from torch.nn import BatchNorm1d as BN
from torch.nn import Conv1d
from torch.nn import Linear as L
from torch.nn import Sequential as S
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN

from torch_geometric.nn import Reshape
from torch_geometric.nn.inits import reset

try:
    from torch_cluster import knn_graph, knn
except ImportError:
    knn_graph = None
    knn = None


def reset(nn):
    def _reset(item):
        if hasattr(item, "reset_parameters"):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, "children") and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class XConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int,
        hidden_channels: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        num_workers: int = 1,
        with_global: bool = False,
        bn_momentum: float = 0.5,
    ):
        super(XConv, self).__init__()

        if knn_graph is None:
            raise ImportError("`XConv` requires `torch-cluster`.")

        # print(with_global)
        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_workers = num_workers
        self.with_global = with_global

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        self.mlp1 = Seq(
            Lin(dim, C_delta),
            ELU(),
            BN(C_delta, momentum=bn_momentum),
            Lin(C_delta, C_delta),
            ELU(),
            BN(C_delta, momentum=bn_momentum),
            Reshape(-1, K, C_delta),
        )

        self.mlp2 = Seq(
            Lin(D * K, K**2),
            ELU(),
            BN(K**2, momentum=bn_momentum),
            Reshape(-1, K, K),
            Conv1d(K, K**2, K, groups=K),
            ELU(),
            BN(K**2, momentum=bn_momentum),
            Reshape(-1, K, K),
            Conv1d(K, K**2, K, groups=K),
            BN(K**2, momentum=bn_momentum),
            Reshape(-1, K, K),
        )

        if with_global:
            self.global_mlp = Seq(
                Lin(3, out_channels // 4, bias=True),
                ELU(),
                BN(out_channels // 4, momentum=bn_momentum),
                Lin(out_channels // 4, out_channels // 4, bias=True),
                ELU(),
                BN(out_channels // 4, momentum=bn_momentum),
            )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = Seq(
            Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            Lin(C_in * depth_multiplier, C_out, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        torch.nn.init.xavier_normal(self.mlp1[0].weight)
        torch.nn.init.xavier_normal(self.mlp1[3].weight)
        reset(self.mlp2)
        torch.nn.init.xavier_normal(self.mlp2[0].weight)
        torch.nn.init.xavier_normal(self.mlp2[4].weight)
        torch.nn.init.xavier_normal(self.mlp2[8].weight)
        reset(self.conv)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        pos_query: Optional[Tensor] = None,
        batch_query: Optional[Tensor] = None,
    ):
        """"""
        pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        pos_backup = pos.clone()
        (N, D), K = pos.size(), self.kernel_size

        if pos_query is None:
            edge_index = knn_graph(
                pos,
                K * self.dilation,
                batch,
                loop=True,
                flow="target_to_source",
                num_workers=self.num_workers,
            )
            row, col = edge_index[0], edge_index[1]
        else:
            # print('deconv')
            edge_index = knn(
                pos_query, pos, K * self.dilation, batch_x=batch_query, batch_y=batch
            )
            row, col = edge_index[0], edge_index[1]
            # print(row, col)
            # print(row.max(), col.max(), row.shape, col.shape)

        if self.dilation > 1:
            dil = self.dilation
            index = torch.randint(K * dil, (N, K), dtype=torch.long, device=row.device)
            arange = torch.arange(N, dtype=torch.long, device=row.device)
            arange = arange * (K * dil)
            index = (index + arange.view(-1, 1)).view(-1)

            # print(row, index)
            row = row[index]
            col = col[index]

        pos = pos[col] - pos[row]

        x_star = self.mlp1(pos.view(N * K, D))
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[col].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)

        transform_matrix = self.mlp2(pos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)

        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)

        out = self.conv(x_transformed)
        # print(out.shape)

        if self.with_global:
            # print(pos.shape)
            fts_global = self.global_mlp(pos_backup)
            # print(fts_global.shape)
            return torch.cat([fts_global, out], axis=-1)
        else:
            return out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
