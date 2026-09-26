"""模型构建与模型文件的保存／加载。

模型文件里除了权重，还固化「模型结构、类别映射、预处理参数、训练配置」四项信息，
这样任何端（训练脚本、单张推理、批量推理、Web 服务）只需读一个文件即可正确复现。
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import models

from common import (INPUT_SIZE, NORM_MEAN, NORM_STD, NUM_CLASSES, RESIZE_SIZE,
                    build_mapping)

SUPPORTED_ARCHS = ("resnet18", "resnet34", "resnet50", "efficientnet_b0")


def build_model(arch: str = "resnet18", num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """按 ImageNet 预训练权重初始化，并替换成 51 类输出头。"""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"不支持的模型结构：{arch}，可选 {SUPPORTED_ARCHS}")

    if arch.startswith("resnet"):
        weights = None
        if pretrained:
            weights = getattr(models, f"ResNet{arch[6:]}_Weights").IMAGENET1K_V1
        net = getattr(models, arch)(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    return net


def build_criterion(name: str = "cross_entropy", label_smoothing: float = 0.0,
                    class_weights=None) -> nn.Module:
    """class_weights: 每类的 loss 权重张量（需与模型同设备），用于类别不均衡。"""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    raise ValueError(f"不支持的损失函数：{name}")


def freeze_backbone(net: nn.Module) -> list[nn.Parameter]:
    """冻结特征提取部分，只训练分类头，返回需要优化的参数。"""
    head_names = ("fc", "classifier")
    for name, param in net.named_parameters():
        param.requires_grad = any(name.startswith(h) for h in head_names)
    return [p for p in net.parameters() if p.requires_grad]


def unfreeze_all(net: nn.Module) -> list[nn.Parameter]:
    for param in net.parameters():
        param.requires_grad = True
    return list(net.parameters())


def save_checkpoint(path: str, net: nn.Module, cfg: dict, *, epoch: int,
                    best_macro_f1: float, optimizer: torch.optim.Optimizer | None = None,
                    tag: str = "best") -> None:
    """保存完整模型文件：权重 + 结构 + 类别映射 + 预处理参数 + 训练配置。"""
    class_ids, class_names = build_mapping()
    payload = {
        "format_version": 2,
        "tag": tag,
        "arch": cfg["arch"],
        "num_classes": NUM_CLASSES,
        "model_state": net.state_dict(),
        "class_ids": class_ids,
        "class_names": class_names,
        "label_to_index": {lab: i for i, lab in enumerate(class_ids)},
        "index_to_label": {str(i): lab for i, lab in enumerate(class_ids)},
        "preprocess": {
            "input_size": cfg.get("input_size", INPUT_SIZE),
            "resize_size": cfg.get("resize_size", RESIZE_SIZE),
            "mean": NORM_MEAN,
            "std": NORM_STD,
        },
        "train_config": {k: v for k, v in cfg.items()},
        "epoch": epoch,
        "best_val_macro_f1": best_macro_f1,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu"):
    """加载模型文件，返回 (模型, checkpoint)。模型已切到 eval 模式。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到模型文件：{path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    expected = [ckpt["index_to_label"][str(i)] for i in range(ckpt["num_classes"])]
    if len(expected) != NUM_CLASSES:
        raise ValueError(f"模型文件的类别数为 {len(expected)}，与当前工程的 {NUM_CLASSES} 不一致")

    net = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, ckpt
