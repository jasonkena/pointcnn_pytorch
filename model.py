# adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/point_cnn.py
# also see https://github.com/pyg-team/pytorch_geometric/issues/1470

# import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ELU

from torch_geometric.nn import fps
from xconv import XConv

# using architecture from Fig 1E https://arxiv.org/pdf/1801.07791.pdf
# REMEMBER dropout rate

# parser = argparse.ArgumentParser()
# parser.add_argument("--epochs", type=int, default=200)
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--lr_decay_factor", type=float, default=0.5)
# parser.add_argument("--lr_decay_step_size", type=int, default=50)
# parser.add_argument("--weight_decay", type=float, default=0)
# parser.add_argument("--inference", action="store_true")
# parser.add_argument("--profile", action="store_true")
# parser.add_argument("--bf16", action="store_true")
# args = parser.parse_args()
#
#
# class PointCNN(nn.Module):
#     def __init__(self, num_classes, bn_momentum=0.01):
#         super().__init__()
#
#         # channels = [256, 256, 512, 1024, 512, 256, 256, 256]
#         # p = [2049, 768, 384, 128, 384, 768, 2048, 2048]
#         # n_fuse = 4  # num upsampling steps?
#         # k = [8, 12, 16, 16, 16, 12, 8, 8]
#         # d = [1, 2, 2, 6, 6, 6, 6, 4]
#         # # residual connections
#         # links = [(0, 6), (0, 7), (1, 5), (2, 4)]
#
#         self.num_classes = num_classes
#
#         self.conv1 = XConv(
#             0, 256, dim=3, kernel_size=8, hidden_channels=256 // 2, dilation=1
#         )
#         self.conv2 = XConv(256, 256, dim=3, kernel_size=12, dilation=2)
#         self.conv3 = XConv(256, 512, dim=3, kernel_size=16, dilation=2)
#         self.conv4 = XConv(512, 1024, dim=3, kernel_size=16, dilation=6)
#
#         self.deconv1 = XConv(1024, 512, dim=3, kernel_size=16, dilation=6)
#         self.deconv2 = XConv(512, 256, dim=3, kernel_size=12, dilation=6)
#         self.deconv3 = XConv(256, 256, dim=3, kernel_size=8, dilation=6)
#         self.deconv4 = XConv(256, 256, dim=3, kernel_size=8, dilation=4)
#
#         self.fuse1 = nn.Sequential(nn.Linear(1024 + 512//2, 512), nn.ELU(), nn.BatchNorm1d(256, momentum=bn_momentum))
#         self.fuse2 = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.BatchNorm1d(256, momentum=bn_momentum))
#         self.fuse3 = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.BatchNorm1d(256, momentum=bn_momentum))
#         self.fuse4 = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.BatchNorm1d(256, momentum=bn_momentum))
#
#
#         self.lin1 = nn.Sequential(
#             nn.Linear(256, 256, bias=True),
#             nn.ELU(),
#             nn.BatchNorm1d(256, momentum=bn_momentum),
#         )
#         self.lin2 = nn.Sequential(
#             nn.Linear(256, 256, bias=True),
#             nn.ELU(),
#             nn.BatchNorm1d(256, momentum=bn_momentum),
#         )
#         self.lin3 = nn.Linear(256, num_classes, bias=True)
#
#     def forward(self, x, pos, batch):
#         # X of shape [N, |features|]
#         # pos of shape [N, 3]
#         # batch of shape [N]
#         x1 = F.relu(self.conv1(x, pos, batch))
#
#         idx1 = fps(pos, batch, ratio=0.375)
#         x1_sub, pos1_sub, batch1_sub = x1[idx1], pos[idx1], batch[idx1]
#
#         x2 = F.relu(self.conv2(x1_sub, pos1_sub, batch1_sub))
#
#         idx2 = fps(pos1_sub, batch1_sub, ratio=0.5)
#         x2_sub, pos2_sub, batch2_sub = x2[idx2], pos1_sub[idx2], batch1_sub[idx2]
#
#         x3 = F.relu(self.conv3(x2_sub, pos2_sub, batch2_sub))
#
#         idx3 = fps(pos2_sub, batch2_sub, ratio=1 / 3)
#         x3_sub, pos3_sub, batch3_sub = x3[idx3], pos2_sub[idx3], batch2_sub[idx3]
#
#         x = F.relu(self.conv4(x3_sub, pos3_sub, batch3_sub))
#         print(x.shape)
#         x = F.relu(self.deconv1(x, pos3_sub, batch3_sub, pos_query=pos2_sub, batch_query=batch2_sub))
#         print(x.shape)
#         __import__('pdb').set_trace()
#         xd = F.relu(self.deconv1(x, pos3_sub, batch3_sub, pos_query=pos3_sub, batch_query=batch3_sub))
#         x = torch.cat([x, x3], axis=-1)
#
#         x = self.fuse1(x)
#         x = F.relu(self.deconv2(x, pos2_sub, batch2_sub, pos_query=pos3_sub, batch_query=batch3_sub))
#
#         x = torch.cat([x, x2], axis=-1)
#         x = self.fuse2(x)
#
#         x = F.relu(self.deconv3(x, pos1_sub, batch1_sub, pos_query=pos2_sub, batch_query=batch2_sub))
#
#         x = torch.cat([x, x1], axis=-1)
#         x = self.fuse3(x)
#
#         x = F.relu(self.deconv4(x, pos, batch, pos_query=pos1_sub, batch_query=batch1_sub))
#
#         x = torch.cat([x, x1], axis=-1)
#         x = self.fuse4(x)
#
# x = F.relu(self.lin1(x))
# x = F.relu(self.lin2(x))
# x = F.dropout(x, p=0.5, training=self.training)
# x = self.lin3(x)
# return F.log_softmax(x, dim=-1)


class PointCNN(torch.nn.Module):
    def __init__(self, num_classes, bn_momentum=0.01):
        super(PointCNN, self).__init__()

        self.num_classes = num_classes

        self.conv1 = XConv(
            0, 256, dim=3, kernel_size=8, hidden_channels=256 // 2, dilation=1
        )
        self.conv2 = XConv(
            256, 256, dim=3, kernel_size=12, hidden_channels=256 // 4, dilation=2
        )
        self.conv3 = XConv(
            256, 512, dim=3, kernel_size=16, hidden_channels=512 // 4, dilation=2
        )
        self.conv4 = XConv(
            512,
            1024,
            dim=3,
            kernel_size=16,
            hidden_channels=1024 // 4,
            dilation=6,
            with_global=True,
        )

        self.deconv1 = XConv(
            1024 + 1024 // 4,
            1024,
            dim=3,
            kernel_size=16,
            hidden_channels=512 // 4,
            dilation=6,
        )
        self.deconv2 = XConv(
            1024, 512, dim=3, kernel_size=16, hidden_channels=256 // 4, dilation=6
        )
        self.deconv3 = XConv(
            512, 256, dim=3, kernel_size=12, hidden_channels=256 // 4, dilation=6
        )
        self.deconv4 = XConv(
            256, 256, dim=3, kernel_size=8, hidden_channels=256 // 4, dilation=6
        )
        self.deconv5 = XConv(
            256, 256, dim=3, kernel_size=8, hidden_channels=256 // 4, dilation=4
        )

        self.fuse1 = Seq(
            Lin(2048 + 1024 // 4, 1024, bias=True),
            ELU(),
            BN(1024, momentum=bn_momentum),
        )
        self.fuse2 = Seq(
            Lin(1024, 512, bias=True), ELU(), BN(512, momentum=bn_momentum)
        )
        self.fuse3 = Seq(Lin(512, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
        self.fuse4 = Seq(Lin(512, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
        self.fuse5 = Seq(Lin(512, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))

        self.lin1 = Seq(Lin(256, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
        torch.nn.init.xavier_uniform(self.lin1[0].weight)
        self.lin2 = Seq(Lin(256, 256, bias=True), ELU(), BN(256, momentum=bn_momentum))
        torch.nn.init.xavier_uniform(self.lin1[0].weight)
        self.lin3 = Lin(256, num_classes)
        torch.nn.init.xavier_uniform(self.lin1[0].weight)

    def forward(self, x, pos, batch, edge_index=None):
        x1 = F.relu(self.conv1(None, pos, batch))

        idx1 = fps(pos, batch, ratio=0.375)
        x1_sub, pos1_sub, batch1_sub = x1[idx1], pos[idx1], batch[idx1]

        x2 = F.relu(self.conv2(x1_sub, pos1_sub, batch1_sub))

        idx2 = fps(pos1_sub, batch1_sub, ratio=0.5)
        x2_sub, pos2_sub, batch2_sub = x2[idx2], pos1_sub[idx2], batch1_sub[idx2]

        x3 = F.relu(self.conv3(x2_sub, pos2_sub, batch2_sub))

        idx3 = fps(pos2_sub, batch2_sub, ratio=1 / 3)
        x3_sub, pos3_sub, batch3_sub = x3[idx3], pos2_sub[idx3], batch2_sub[idx3]
        x4 = F.relu(self.conv4(x3_sub, pos3_sub, batch3_sub))
        x = F.relu(
            self.deconv1(
                x4, pos3_sub, batch3_sub, pos_query=pos3_sub, batch_query=batch3_sub
            )
        )
        x = torch.cat([x, x4], axis=-1)

        x = self.fuse1(x)
        x = F.relu(
            self.deconv2(
                x, pos2_sub, batch2_sub, pos_query=pos3_sub, batch_query=batch3_sub
            )
        )

        x = torch.cat([x, x3], axis=-1)
        x = self.fuse2(x)

        x = F.relu(
            self.deconv3(
                x, pos1_sub, batch1_sub, pos_query=pos2_sub, batch_query=batch2_sub
            )
        )

        x = torch.cat([x, x2], axis=-1)
        x = self.fuse3(x)

        x = F.relu(
            self.deconv4(x, pos, batch, pos_query=pos1_sub, batch_query=batch1_sub)
        )

        x = torch.cat([x, x1], axis=-1)
        x = self.fuse4(x)

        x = F.relu(self.deconv5(x, pos, batch))

        x = torch.cat([x, x1], axis=-1)
        x = self.fuse5(x)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


# train_dataset, test_dataset = get_dataset(num_points=1024)
# model = Net(train_dataset.num_classes)
# run(
#     train_dataset,
#     test_dataset,
#     model,
#     args.epochs,
#     args.batch_size,
#     args.lr,
#     args.lr_decay_factor,
#     args.lr_decay_step_size,
#     args.weight_decay,
#     args.inference,
#     args.profile,
#     args.bf16,
# )
