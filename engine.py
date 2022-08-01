# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Train and eval functions used in main.py."""
import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import ModelEma, accuracy

from torchmetrics import AUROC, F1Score, Recall


import utils


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        model_ema: Optional[ModelEma] = None,
        mixup_fn: Optional[Mixup] = None,
        disable_amp: bool = False,
):
    """train one epoch function."""
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if disable_amp:
            # Disable AMP and try to solve the NaN issue.
            # Ref: https://github.com/facebookresearch/deit/issues/29
            outputs = model(samples)
            loss = criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if disable_amp:
            loss.backward()
            optimizer.step()
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                    hasattr(optimizer, "is_second_order") and
                    optimizer.is_second_order
            )
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, disable_amp, mc_dropout=False, mc_iter=1, metrics=None):
    """evaluation function."""
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    auroc = AUROC(pos_label=1)
    f1score = F1Score(pos_label=1)
    recall = Recall(pos_label=1)

    # switch to evaluation mode
    model.eval()

    mc_gts = []
    mc_results = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if disable_amp:
            if mc_dropout:
                outputs = []
                for _ in range(mc_iter):
                    output = model(images, drop_on=True)
                    outputs.append(output)
                outputs = torch.stack(outputs, dim=1)
                output = torch.mean(outputs, 1)
                outputs = F.softmax(outputs, dim=2)
                outputs = outputs.detach().cpu().numpy()
                mc_results.append(outputs)
                mc_gts.append(target.detach().cpu().numpy())
            else:
                output = model(images)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                if mc_dropout:
                    outputs = []
                    for _ in range(mc_iter):
                        output = model(images, drop_on=True)
                        outputs.append(output)
                    outputs = torch.stack(outputs, dim=1)
                    output = torch.mean(outputs, 1)
                    outputs = F.softmax(outputs, dim=2)
                    outputs = outputs.detach().cpu().numpy()
                    mc_results.append(outputs)
                    mc_gts.append(target.detach().cpu().numpy())
                else:
                    output = model(images)
                loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

        if metrics:
            auroc_val = auroc(output, target)
            f1_score_val = f1score(output, target)
            recall_val = recall(output, target)
            metric_logger.meters['auroc'].update(auroc_val.item(), n=batch_size)
            metric_logger.meters['f1score'].update(f1_score_val.item(), n=batch_size)
            metric_logger.meters['recall'].update(recall_val.item(), n=batch_size)

    if metrics:
            print('* Acc@1 {top1.global_avg:.3f} auroc {auroc.global_avg:.3f} f1-score {f1score.global_avg:.3f}  '
                  'recall {recall.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, auroc=metric_logger.auroc, f1score=metric_logger.f1score,
                          recall=metric_logger.recall, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f}  aucroc {me}  loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    final_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if mc_dropout:
        mc_results = {
            'probs': np.concatenate(mc_results, axis=0),
            'gts': np.concatenate(mc_gts, axis=0),
            'images': [s[0] for s in data_loader.dataset.samples],
        }
        return final_stats, mc_results
    return final_stats