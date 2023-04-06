# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Train and eval functions used in main.py."""
import math
import sys
from typing import Iterable, Optional
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
def evaluate(data_loader, model, device, disable_amp, mc_dropout=False, mc_iter=1, metrics=None, cam=False):
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

    if cam:
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(0),
                tensor.size(1), 1, 1)

            # Bring the channels to the first dimension,
            # like in CNNs.
            # result = result.transpose(2, 3).transpose(1, 2)
            return result
        # cam_model = partial(model, drop_on=True)
        target_layers = [model.mhca_stages[-1].aggregate.bn]
        gcam = GradCAM(model=model, target_layers=target_layers)#, reshape_transform=reshape_transform)

    mc_gts = []
    mc_results = []
    y_pred_list = []
    y_target_list = []
    cam_all = []
    rgb_images = [s[0] for s in data_loader.dataset.samples]
    start = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # print(target)
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
            
            if cam:
                with torch.set_grad_enabled(True):
                    cam_images = [[] for i in range(len(images))]
                    for mc_i in range(1):
                        grayscale_cam = gcam(input_tensor=images,
                                targets=[ClassifierOutputTarget(t) for t in target])

                        model.zero_grad()
                        # denorm = images.cpu() * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]
                        # print(denorm)
                        # grayscale_cam[grayscale_cam < 0.3] = 0
                        for img_i, (rgbi, g) in enumerate(zip(rgb_images[start: start+len(images)], grayscale_cam)):
                            rgbi = Image.open(rgbi).convert('RGB')
                            rgbi = np.array(rgbi) / 255
                            # g = cv2.resize(g, (224, 224))
                            # pad g to 256, 256
                            # g = np.pad(g, ((16, 16), (16, 16)), 'constant', constant_values=np.minimum(g.min(), 0))
                            # resize as rgbi
                            g = cv2.resize(g, (rgbi.shape[1], rgbi.shape[0]))
                            # g[g < 0.5] = 0
                            cap = utils.show_cam_on_image(rgbi, g, image_weight=0.5, colormap=cv2.COLORMAP_HOT, thr=0.)
                            # mask = np.expand_dims(g, axis=-1)
                            # mask = np.repeat(mask, 3, axis=-1)
                            # cap[mask < 0.3] = (rgbi[mask < 0.3] * 255).astype(np.uint8)
                            cam_images[img_i].append(cap)
                    # cam_images
                        # cam_images.append(
                        #     np.stack([show_cam_on_image(rgbi, g, image_weight=0.5) for rgbi, g in zip(denorm.cpu().permute(0, 2, 3, 1).numpy(), grayscale_cam)], axis=0))
                    # cam_images = cam_images, axis=0)
                    cam_all += [np.stack(c, axis=-1)[:, :, ::-1] for c in cam_images]
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
                
                if cam:
                    with torch.set_grad_enabled(True):
                        cam_images = []
                        for _ in range(10):
                            grayscale_cam = gcam(input_tensor=images, targets=[ClassifierOutputTarget(t) for t in target])
                            denorm = images * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]

                            cam_images.append(
                                np.stack([show_cam_on_image(rgbi.cpu().detach().numpy(), g, image_weight=0.8) for rgbi, g in zip(denorm, grayscale_cam)], axis=0))
                        cam_images = np.stack(cam_images, axis=-1)
                        cam_all.append(cam_images)

                loss = criterion(output, target)
        start += len(images)
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
            'images': rgb_images,
            'cam': cam_all if cam else None
        }
        return final_stats, mc_results
    return final_stats