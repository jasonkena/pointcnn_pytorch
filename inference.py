import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader.data_list_loader import DataListLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from model import PointCNN

import argparse

import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from ribseg_dataset import RibSegDataset


# this is necessary to ensure that DataParallel splits the batches correctly across the inputs (i.e., forward has to have splittable batch index)
class Wrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = PointCNN(*args, **kwargs)

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


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary", action="store_true", help="Use binary classification"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to model checkpoint to load"
    )
    parser.add_argument("--output_dir", type=str, help="Path to save output")
    parser.add_argument("--dataset_path", type=str, default="", metavar="N")
    parser.add_argument("--binary_dataset_path", type=str, default="", metavar="N")
    parser.add_argument("--batch_size", type=int, default=200, metavar="N")
    parser.add_argument("--dry_run", type=int, default=0, metavar="N")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    model = Wrapper(25 if not args.binary else 2).to(device)
    # model = PointCNN(25 if not args.binary else 2).to(device)
    model = nn.DataParallel(model)

    # hack because of pytorch lightning saving scheme?
    state_dict = torch.load(args.model_path)["state_dict"]
    for k in list(state_dict.keys()):
        assert "model" in k
        state_dict[k.replace("model", "module.model")] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()

    # num_devices=4
    npoints = 2048
    batch_size = args.batch_size

    time_samples = []
    dry_run = args.dry_run
    if dry_run:
        print("Dry run, not saving anything")

    test_dataset = RibSegDataset(
        root=args.dataset_path,
        split="all",
        npoints=npoints,
        eval=True,
        binary_root=args.binary_dataset_path,
    )
    # test_loader = DataListLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    for fn, ct_list, label_list in tqdm(test_loader):
        if dry_run and (len(time_samples) > dry_run):
            break
        time_samples.append(0)
        all_data = []
        all_segs = []
        all_preds = []

        for batch_idx in tqdm(range(0, len(ct_list), batch_size), leave=False):
            data = ct_list[batch_idx : batch_idx + batch_size]
            data = torch.cat(data, dim=0)
            seg = label_list[batch_idx : batch_idx + batch_size]
            seg = torch.cat(seg, dim=0)

            if args.binary:
                seg = (seg > 0).to(seg.dtype)

            data, seg = (
                data.to(device),
                seg.to(device),
            )

            with torch.no_grad():
                # [B*N, 3]
                start = time.time()
                seg_pred = model(data)
                end = time.time()
                time_samples[-1] += end - start
                # seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            data_np = data.cpu().numpy()
            seg_np = seg.cpu().numpy().reshape(-1)
            pred_np = seg_pred.detach().cpu().numpy()
            all_data.append(data_np)
            all_segs.append(seg_np)
            all_preds.append(pred_np)

        if not dry_run:
            name = fn[0].split("/")[-1].split(".")[0]
            np.savez(
                args.output_dir + "/" + name + ".npz",
                data=np.concatenate(all_data, axis=0),
                seg=np.concatenate(all_segs, axis=0),
                pred=np.concatenate(all_preds, axis=0),
            )
    print(
        f"Average inference time: {np.mean(time_samples):.4f} seconds with std: {np.std(time_samples):.4f}"
    )


if __name__ == "__main__":
    inference()
