import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
    from common import visualize_directreg

import header

import select

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
import numpy as np


import DetNet
import py.training.common.a_geometry as geo

from CombinedDataset import CombinedDataset
from py.training.common.a_geometry import *
import wandb

wandb.init(project="detection_nov1_160x160")

device = torch.device("cuda:0")

# torch.autograd.set_detect_anomaly(True)


modelinputW = header.model_input_width
modelinputH = header.model_input_height

def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(states, os.path.join(output_dir, filename))


def firstbatchcpu(d):
    return d[0].detach().cpu().numpy()


def train_batch(val, loss_fn, optimizer, model, ):
    inp = val['image'].to(device)
    print("inp", inp.shape)

    exists_gt = val['exists'].to(device)
    center_x_gt = val['center_x'].to(device)
    center_y_gt = val['center_y'].to(device)
    size_gt = val['size'].to(device)

    pred = model(inp)

    exists_pred = pred[0]
    center_x_pred = pred[1]
    center_y_pred = pred[2]
    size_pred = pred[3]

    loss_exists = loss_fn(exists_gt, exists_pred)

    loss_center_x = loss_fn(center_x_gt*exists_gt, center_x_pred*exists_gt)
    loss_center_y = loss_fn(center_y_gt*exists_gt, center_y_pred*exists_gt)
    loss_size = loss_fn(size_gt*exists_gt, size_pred*exists_gt)

    loss = 0
    loss = loss_exists+loss_center_x+loss_center_y+loss_size

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # wandb.log({f"{name}_loss": loss})
    print(f"loss is {loss}")
    wandb.log({"loss": loss})

def validate_epoch(val_dataloader, loss_fn, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            inp = batch['image'].to(device)
            exists_gt = batch['exists'].to(device)
            center_x_gt = batch['center_x'].to(device)
            center_y_gt = batch['center_y'].to(device)
            size_gt = batch['size'].to(device)

            pred = model(inp)
            exists_pred = pred[0]
            center_x_pred = pred[1]
            center_y_pred = pred[2]
            size_pred = pred[3]

            loss_exists   = loss_fn(exists_gt, exists_pred)
            loss_center_x = loss_fn(center_x_gt * exists_gt, center_x_pred * exists_gt)
            loss_center_y = loss_fn(center_y_gt * exists_gt, center_y_pred * exists_gt)
            loss_size     = loss_fn(size_gt * exists_gt, size_pred * exists_gt)

            total_loss += (loss_exists + loss_center_x + loss_center_y + loss_size).item()

    model.train()
    return total_loss / len(val_dataloader)

def main():
    batch_size = 64
    num_workers = 24
    
    model = DetNet.DetNet()
    model = torch.nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.module.parameters())

    loss_fn = nn.MSELoss(reduction="mean").to(device)

    start_epoch = 0
    last_validation_loss = 10000000000

    checkpoint_file = os.path.join(
        "checkpoints", 'checkpoint.pth'
    )

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    dataset = CombinedDataset()

    # data split 70/10/20
    test_size  = int(0.2 * len(dataset))
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(start_epoch, 200):
        # training
        model.train()
        length = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader):
            print(f"{idx*100/length}% way through epoch {epoch}")
            train_batch(batch, loss_fn, optimizer, model)

        # validation
        val_loss = validate_epoch(val_dataloader, loss_fn, model)
        print(f"Epoch {epoch} validation loss: {val_loss}")
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        
        # save latest checkpoint
        final_output_dir = f"checkpoints"
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, final_output_dir)
        # save best checkpoint when validation loss improves
        if val_loss < last_validation_loss:
            last_validation_loss = val_loss
            print(f"Validation improved — saving best checkpoint")
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, final_output_dir, filename='checkpoint_best.pth')
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, saving extra!")
            os.system(
                f"cp {final_output_dir}/checkpoint.pth {final_output_dir}/checkpoint_{epoch}.pth")

    # test eval after training done
    print("Training complete. Running final test evaluation...")
    test_loss = validate_epoch(test_dataloader, loss_fn, model)
    print(f"Final test loss: {test_loss}")
    wandb.log({"test_loss": test_loss})

if __name__ == "__main__":
    main()
