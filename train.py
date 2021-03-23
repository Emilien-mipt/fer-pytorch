import time

import numpy as np
import torch
from tqdm import tqdm

from config import CFG
from utils.utils import AverageMeter, timeSince


def train_fn(train_loader, model, criterion, optimizer, scaler, epoch, device, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()

    # Iterate over dataloader
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        # zero the gradients
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)

        if CFG.MIXED_PREC:
            # Runs the forward pass with autocasting
            with torch.cuda.amp.autocast():
                y_preds = model(images)
                loss = criterion(y_preds, labels)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
        else:
            y_preds = model(images)
            loss = criterion(y_preds, labels)

            # Compute gradients and do step
            loss.backward()
            optimizer.step()
        # print("Current LR:", optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            # scheduler.step()
            scheduler.step()

        # record loss
        losses.update(loss.item(), batch_size)
        classes = y_preds.argmax(dim=1)
        acc = torch.mean((classes == labels).float())
        accuracy.update(acc, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % CFG.print_freq == 0 or i == (len(train_loader) - 1):
            print(
                "Epoch: [{Epoch:d}][{Iter:d}/{Len:d}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    Epoch=epoch + 1,
                    Iter=i,
                    Len=len(train_loader),
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(i + 1) / len(train_loader)),
                )
            )
    return losses.avg, accuracy.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds = []
    # switch to evaluation mode
    model.eval()
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record accuracy
        preds.append(y_preds.softmax(1).to("cpu").numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{Step:d}/{Len:d}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    Step=step,
                    Len=len(valid_loader),
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions
