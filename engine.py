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
#torchmetrics==0.9.3
from torchmetrics import AUROC, ConfusionMatrix

# for torchmetrics==0.11.3
# from torchmetrics.classification import (
#     BinaryConfusionMatrix, 
#     BinaryAUROC, 
#     BinaryF1Score,
#     BinaryPrecision,
#     BinaryRecall)
# from sklearn.metrics import roc_auc_score, average_precision_score


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

    # for torchmetrics==0.11.1
    if metrics:
        auroc = AUROC(num_classes=2).to(device)
        cfmat = ConfusionMatrix(num_classes=2).to(device)


    # switch to evaluation mode
    model.eval()

    mc_gts = []
    mc_results = []
    y_pred_list = []
    y_target_list = []

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

        y_pred_list.append(output) # output's shape : [batch_size, 2]
        # y_pred_list.append(torch.argmax(output, dim=1)) # output's shape : [batch_size,]
        y_target_list.append(target) # target's shape : [batch_size]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    if metrics:
        y_pred_list = torch.cat(y_pred_list)
        y_target_list = torch.cat(y_target_list)
        auc_all = auroc(y_pred_list, y_target_list)
        print(f'AUROC: {auc_all:.4f}')

        cfmat_all = cfmat(y_pred_list, y_target_list)
        TN, FP, FN, TP = cfmat_all[0][0], cfmat_all[0][1], cfmat_all[1][0], cfmat_all[1][1]
        print(f'TN:{cfmat_all[0][0]}, FP:{cfmat_all[0][1]}, FN:{cfmat_all[1][0]}, TP:{cfmat_all[1][1]}')
        # Accuracy -> (TP + TN) / (TP + TN + FP + FN)
        Acc_from_cmt = (TP + TN) / (TP + TN + FP + FN)
        print(f'Acc. from confusion matrix: {Acc_from_cmt:.4f}')
        # Sensitivity -> TPR = TP / (TP + FN)
        TPR = TP/(TP + FN)
        print(f'Sensitivity: {TPR:.4f}')
        # Specificity -> TNR = TN / (TN + FP)
        TNR = TN/(TN + FP)
        print(f'Specificity: {TNR:.4f}')
        # Positive predictive value -> TP / (TP + FP) = Precision (PPV)
        PPV = TP/(TP + FP)
        print(f'Positive predictive value: {PPV:.4f}')
        # negative predictive value -> TN / (TN + FN)
        NPV = TN/(TN + FN)
        print(f'Negative predictive value: {NPV:.4f}')
        # F1-Score -> 2 x (PPV x TPR) / (PPV + TPR)
        f1_score = 2 * (PPV * TPR) / (PPV + TPR)
        print(f'F1-score: {f1_score:.4f}')
        
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    final_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if metrics:
        final_stats["sensitivity"] = TPR.item()
        final_stats["specificity"] = TNR.item()
        final_stats["positive predictive value"] = PPV.item()
        final_stats["negative predictive value"] = NPV.item()
        final_stats["F1-score"] = f1_score.item()
        final_stats["auroc"] = auc_all.item()
        final_stats["acc_from_cmt"] = Acc_from_cmt.item()
        

    if mc_dropout:
        mc_results = {
            'probs': np.concatenate(mc_results, axis=0),
            'gts': np.concatenate(mc_gts, axis=0),
            'images': [s[0] for s in data_loader.dataset.samples],
        }
        return final_stats, mc_results
    return final_stats