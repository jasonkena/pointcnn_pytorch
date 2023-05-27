import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import PointCNN
from ribseg_dataset import RibSegDataset


class LightningPointCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
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
        out = self(x)
        loss = F.nll_loss(out, y.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self(x)
        loss = F.nll_loss(out, y.flatten())
        self.log("val_loss", loss)


def main():
    # num_devices = 1
    num_devices = 4
    # data
    npoints = 2048
    # npoints = 8096
    batch_size = 16

    max_epochs = 200
    # batch_size = 48

    config = {"npoints": npoints, "batch_size": batch_size, "max_epochs": max_epochs}
    train_dataset = RibSegDataset(
        root="../ribseg_benchmark", npoints=npoints, split="train"
    )
    val_dataset = RibSegDataset(
        root="../ribseg_benchmark", npoints=npoints, split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=48,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=48,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    # model
    model = LightningPointCNN(num_classes=25)

    # training
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        devices=num_devices,
        accelerator="gpu",
        max_epochs=max_epochs,
        precision="bf16",
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    # trainer = pl.Trainer(gpus=4, num_nodes=1) # precision=16)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
