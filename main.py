import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import PointCNN

import argparse

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from ribseg_dataset import RibSegDataset


class LightningPointCNN(pl.LightningModule):
    def __init__(self, num_classes, binary):
        super().__init__()
        self.binary = binary
        if self.binary:
            num_classes = 2
        self.model = PointCNN(num_classes)

    def forward(self, x):
        # x: [B, N, 3]
        batch = (
            torch.arange(x.shape[0])
            .unsqueeze(1)
            .repeat(1, x.shape[1])
            .long()
            .to(x.device)
        )
        batch = batch.flatten()
        x = x.view(-1, 3)
        return self.model(None, x, batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.binary:
            y = (y > 0).to(y.dtype)
        out = self(x)
        loss = F.nll_loss(out, y.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        if self.binary:
            y = (y > 0).to(y.dtype)
        out = self(x)
        loss = F.nll_loss(out, y.flatten())
        self.log("val_loss", loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", action="store_true", help="Use binary classification"
    )
    parser.add_argument("--dataset_path", type=str, default="", metavar="N")
    parser.add_argument("--binary_dataset_path", type=str, default="", metavar="N")

    args = parser.parse_args()

    # num_devices = 1
    num_devices = 4
    # data
    npoints = 2048
    # npoints = 8096
    batch_size = 32
    # batch_size = 16

    max_epochs = 200
    # batch_size = 48

    config = {
        "npoints": npoints,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "binary": args.binary,
    }
    train_dataset = RibSegDataset(
        root=args.dataset_path,
        npoints=npoints,
        split="train",
        binary_root=args.binary_dataset_path,
    )
    val_dataset = RibSegDataset(
        root=args.dataset_path,
        npoints=npoints,
        split="val",
        binary_root=args.binary_dataset_path,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=48,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=48,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # pin_memory=True,
    )

    # model
    model = LightningPointCNN(num_classes=25, binary=args.binary)

    # training
    wandb_logger = WandbLogger()
    wandb_logger.log_hyperparams(config)

    trainer = pl.Trainer(
        # devices=num_devices,
        # accelerator="gpu",
        max_epochs=max_epochs,
        precision="bf16-mixed",
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    print(
        f"Trainer initialized: device={trainer.device_ids} strategy={trainer.strategy} "
    )
    print(f"trainer.world_size={trainer.world_size}")
    print(
        f"trainer.strategy.cluster_environment.world_size()={trainer.strategy.cluster_environment.world_size()}"
    )
    print(
        f"trainer.strategy._accelerator.auto_device_count={trainer.strategy._accelerator.auto_device_count()}"
    )
    print(f"trainer.strategy.world_size={trainer.strategy.world_size}")

    # trainer = pl.Trainer(gpus=4, num_nodes=1) # precision=16)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
