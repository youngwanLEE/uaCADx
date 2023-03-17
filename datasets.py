# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
datasets class
"""
import json
import os

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torchvision.transforms as T


class INatDataset(ImageFolder):
    """Dataset class."""

    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in
        # ['kingdom','phylum','class',
        # 'order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    """buld_dataset."""
    if args.simple_aug:
        print("simple augmentation is applied")
        transform = build_simple_transform(is_train)
    else:
        transform = build_timm_transform(is_train, args)
    val_type = args.ext_val if args.ext_val else 'Int_val'
    path = os.path.join(args.data_path, 'Train' if is_train else val_type)
    print(path)
    dataset = ImageFolder(path, transform=transform)
    print(len(dataset.samples))
    return dataset, 2


def build_timm_transform(is_train, args):
    """build transform."""
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] =\
                transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []  # Test-time transformations.
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_simple_transform(is_train=True):
    
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
