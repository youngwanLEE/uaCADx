import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import random
import cv2
import copy
import glob
from IPython.display import Image

import PIL 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader,Dataset
from torchinfo import summary 
from torchmetrics import AUROC, ConfusionMatrix

import tqdm
import timm

import sklearn
from sklearn.model_selection import train_test_split



def get_args_parser():
    """
    get argugment parser.
    """
    parser = argparse.ArgumentParser("MPViT training and evaluation script", add_help=False)

    # Basic training parameters.
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--save_freq", default=10, type=int)

    # Additional args.
    parser.add_argument(
        "--model_kwargs", type=str, default="{}", help="additional parameters for model"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="resnet34",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--drop-path", type=float, default=0.0, metavar="PCT", help="Drop path rate (default: 0.)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, metavar="LR", help="learning rate (default: 5e-4)"
    )

    # Dataset parameters
    parser.add_argument("--data-path", default="data", type=str, help="dataset path")
    
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)  # Note: Original 10 is very high.

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')

    parser.add_argument('--pretrained_mpvit', default="", help="pretrained weight of mpvit")

    parser.add_argument("--ext_val", default="",
                        help="ext_val set. e.g., `Ext_val/ys` or `Ext_val/eh` or `Ext_val/as` ")
    
    parser.add_argument("--mc", action="store_true", default=False, help="Perform mc dropout evaluation")
    parser.add_argument("--mc_iter", type=int, default=100, help="mc dropout iteration")
    parser.add_argument("--metrics", action="store_true", default=False,
                        help="shows otehr metrics: AUC, F1-score, and Recall")
    return parser



def build_transforms(is_train=True):
    
    if is_train:

        data_T = T.Compose([
                T.Resize(size = (256, 256)),
                T.CenterCrop(size=224),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    else:
        data_T = T.Compose([
                T.Resize(size = (256, 256)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_T



def build_dataset(is_train, args):
    """buld_dataset."""
    transform = build_transform(is_train)
    val_type = args.ext_val if args.ext_val else 'Int_val'
    path = os.path.join(args.data_path, 'Train' if is_train else val_type)
    print(path)
    dataset = ImageFolder(path, transform=transform)
    print(len(dataset.samples))
    return dataset, 2


def train_model(args, model, train_loader, test_loader, device,
                lr=0.0001, epochs=30, batch_size=32, weight_decay=0.05, gamma=0.5,
                patience=7):
    
    model_name = args.model
    # history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    # set up loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # pass in the parameters to be updated and learning rate
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    # Training Loop
    max_acc = 0.
    best_epoch = 0
    print("Training Start:")
    for epoch in range(epochs):
        model.train()  # start to train the model, activate training behavior

        train_loss = 0.
        val_loss = 0.
        val_acc = 0.

        for data in tqdm.tqdm(train_loader):
            
            inputs, labels = data[0].to(device), data[1].to(device)            
            optimizer.zero_grad()
            outputs = model(inputs)  # forward
            cur_train_loss = criterion(outputs, labels)  # loss

            # backward
            cur_train_loss.backward()   # run back propagation
            optimizer.step()            # optimizer update all model parameters
            
            # loss
            train_loss += cur_train_loss.item() 
            
        # validation
        model.eval()  # start to train the model, activate training behavior
        with torch.no_grad():  # tell pytorch not to update parameters
            print("validation starts!")
            for data in tqdm.tqdm(test_loader):

                inputs, labels = data[0].to(device), data[1].to(device)            
                outputs = model(inputs)
                # loss
                cur_valid_loss = criterion(outputs, labels)
                val_loss += cur_valid_loss.item()

                _, preds = torch.max(outputs.data, dim=1) 
                cur_val = (preds == labels).sum().item() / labels.size(0)
                val_acc += cur_val
        
        scheduler.step() # learning schedule step
        
        # print training feedback
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(test_loader)
        val_acc = val_acc / len(test_loader)
        
        if val_acc > max_acc:
            # print save checkpoint
            torch.save(model.state_dict(), f'{args.output_dir}/{model_name}_best_model.pt')
            max_acc = val_acc
            best_epoch = epoch

        print(f"Epoch:{epoch + 1} / {epochs}, lr: {optimizer.param_groups[0]['lr']:.6f} train loss:{train_loss:.5f}, val loss:{val_loss:.5f}, val acc:{val_acc:.5f}, max acc:{max_acc:.5f}")
    
        # update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]["lr"])
    
    print(f'best_epoch:{best_epoch}, best accuracy:{max_acc}')

    return history


def main(args):
    
    device = torch.device(args.device)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    print(f"Creating model: {args.model}")
    if args.model == 'inception_v3':
        model = create_model('inception_v3', pretrained=args.pretrained, num_classes=args.nb_classes)
    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            **eval(args.model_kwargs),
        )

    if args.pretrained_mpvit:
        checkpoint = torch.load(args.pretrained_mpvit, map_location="cpu")
        checkpoint_state_dict = checkpoint["model"]
        model_state_dict = model.state_dict()
        for k in ['cls_head.cls.weight', 'cls_head.cls.bias']:
            if k in checkpoint_state_dict and checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter:{k},"
                        f"required shape: {model_state_dict[k].shape},"
                        f"loaded shape: {checkpoint_state_dict[k].shape}")
                del checkpoint_state_dict[k]
        model.load_state_dict(checkpoint_state_dict, strict=False)

    model.to(device)
    
    # training
    history = train_model(args, model, train_loader, val_loader, device=device, lr=args.lr, epochs=args.epochs)
    