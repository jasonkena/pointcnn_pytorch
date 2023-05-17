# adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/point_cnn.py

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from points.datasets import get_dataset
from points.train_eval import run
from torch.nn import Linear as Lin

from torch_geometric.nn import XConv, fps, global_mean_pool
from torch_geometric.profile import rename_profile_file

# using architecture from Fig 1E https://arxiv.org/pdf/1801.07791.pdf
# REMEMBER dropout rate

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--bf16', action='store_true')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        channels = [256, 256, 512, 1024, 512, 256, 256, 256]
        p = [2048, 768, 384, 128, 384, 768, 2048, 2048]
        k = [8, 12, 16, 16, 16, 12, 8, 8]
        d = [1, 2, 2, 6, 6, 6, 6, 4]
        # residual connections
        links = [(0,6), (0,7), (1, 5), (2, 4)]
        self.links = {}
        for source, dest in links:
            if dest in d:
                self.links[dest] += (source,)
            else:
                self.links[dest] = (source,)

        modules = []
        in_channels = 0
        for i in range(len(channels)):
            if i in self.links:
                for j in self.links[i]:
                    # add # input channels from residual connections
                    in_channels += channels[j]
            modules.append(XConv(in_channels, channels[i], dim=3, kernel_size=k[i], dilation=d[i]))
            in_channels = channels[i]
        self.conv = nn.ModuleList(modules)





         XConv(in_channels: int, out_channels: int, dim: int, kernel_size: int, hidden_channels: Optional[int] = None, dilation: int = 1, bias: bool = True, num_workers: int = 1)[source]

        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(48, 96, dim=3, kernel_size=12, hidden_channels=64,
                           dilation=2)
        self.conv3 = XConv(96, 192, dim=3, kernel_size=16, hidden_channels=128,
                           dilation=2)
        self.conv4 = XConv(192, 384, dim=3, kernel_size=16,
                           hidden_channels=256, dilation=2)

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, num_classes)

    def forward(self, pos, batch):
        x = F.relu(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


train_dataset, test_dataset = get_dataset(num_points=1024)
model = Net(train_dataset.num_classes)
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay,
    args.inference, args.profile, args.bf16)

if args.profile:
    rename_profile_file('points', XConv.__name__)
