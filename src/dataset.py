"""数据集与数据加载器。

不依赖 ImageFolder 的自动排序（那会把标签顺序交给文件系统，容易出错），
而是显式按 classes.txt + unknown 的顺序分配标签，与推理端共用同一映射。
"""
from __future__ import annotations

import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from common import (IMG_EXTS, INPUT_SIZE, NORM_MEAN, NORM_STD, RESIZE_SIZE,
                    UNKNOWN, label_to_index)

# 少数图片可能在采集时被截断，允许加载残缺图片而不是直接崩溃
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarDataset(Dataset):
    """(图片路径, 标签索引) 列表式数据集；目录名不在映射内会直接报错。"""

    def __init__(self, root: str, transform=None, limit_per_class: int | None = None,
                 seed: int = 0):
        self.root = root
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.class_counts: dict[str, int] = {}

        lab2idx = label_to_index()
        allowed = set(lab2idx)

        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            if name not in allowed:
                raise ValueError(f"{root} 下出现未登记类别目录：{name!r}，请检查 classes.txt 是否与数据一致")
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS))
            if limit_per_class is not None and len(files) > limit_per_class:
                import random
                files = random.Random(seed).sample(files, limit_per_class)
                files.sort()
            self.class_counts[name] = len(files)
            self.samples.extend((os.path.join(path, f), lab2idx[name]) for f in files)

        if not self.samples:
            raise ValueError(f"{root} 未找到任何图片")
        missing = allowed - set(self.class_counts)
        if missing:
            raise ValueError(f"{root} 缺少类别目录：{sorted(missing)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # 单张坏图不应中断训练
            raise RuntimeError(f"图片读取失败：{path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_train_transform(cfg: dict) -> transforms.Compose:
    """训练期预处理。cfg["augmentation"] 为 "none" 时退化为 Resize + CenterCrop。"""
    size = cfg.get("input_size", INPUT_SIZE)
    if cfg.get("augmentation", "basic") == "none":
        return transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

    ops = [
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if cfg.get("augmentation") == "strong":
        ops += [
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.6),
            transforms.RandomRotation(10),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
            transforms.RandomGrayscale(p=0.05),
        ]
    ops += [transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12))]
    return transforms.Compose(ops)


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """验证／推理期预处理，固定且确定，与训练共用同一归一化参数。
    resize_size 缺省按 256/224 比例跟随 input_size，支持 320 等高分辨率输入。"""
    size = cfg.get("input_size", INPUT_SIZE)
    resize = cfg.get("resize_size") or int(size * RESIZE_SIZE / INPUT_SIZE)
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def build_loaders(cfg: dict):
    train_dir = cfg["train_dir"]
    val_dir = cfg["val_dir"]
    limit = cfg.get("limit_per_class")
    seed = cfg.get("seed", 0)
    # limit_per_class 仅用于冒烟测试，正式训练必须为 None 以使用全部样本
    train_set = CarDataset(train_dir, build_train_transform(cfg),
                           limit_per_class=limit, seed=seed)
    val_set = CarDataset(val_dir, build_eval_transform(cfg),
                         limit_per_class=limit, seed=seed + 1)

    common = dict(num_workers=cfg.get("num_workers", 4),
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=cfg.get("num_workers", 4) > 0)
    if cfg.get("known_oversample"):
        # 类别不均衡过采样：已知类(~110张/类)远少于 unknown(7750张)。
        # 按类频率倒数给每个样本设采样权重，已知类再乘 oversample 系数，
        # 让模型每轮见到更多已知车的随机增广视图 —— 直击「已知车被误拒成 unknown」。
        targets = torch.tensor([lab for _, lab in train_set.samples])
        counts = torch.bincount(targets, minlength=51).clamp(min=1)
        w = (1.0 / counts[targets]).double()
        w[targets < len(counts) - 1] *= float(cfg["known_oversample"])   # 末位索引=unknown，不放大
        sampler = WeightedRandomSampler(w, num_samples=len(targets), replacement=True)
        train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 64),
                                  sampler=sampler, drop_last=False, **common)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 64),
                                  shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=cfg.get("eval_batch_size", 128),
                            shuffle=False, **common)
    return train_loader, val_loader, train_set, val_set


__all__ = ["CarDataset", "build_train_transform", "build_eval_transform",
           "build_loaders", "UNKNOWN"]
