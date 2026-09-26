"""训练脚本：建立 51 类（50 车型 + unknown）分类基线。

用法：
    python src/train.py --config configs/baseline.json
    python src/train.py --config configs/baseline.json --epochs 2 --limit-per-class 20   # 冒烟测试

产出（全部落在 models/ 与 logs/）：
    models/<name>_last.pt     最后一轮
    models/<name>_best.pt    验证集 Macro-F1 最高的一轮（用于选模型）
    models/<name>_mid_epN.pt 中期 checkpoint
    logs/<name>_history.csv  逐轮指标
    logs/<name>_config.json  本次使用的完整配置（可复现）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, TRAIN_DIR, VAL_DIR,
                    build_mapping, describe_device, ensure_dirs, get_device, set_seed)
from dataset import build_loaders
from metrics import accuracy, macro_f1
from model import (build_criterion, build_model, freeze_backbone, save_checkpoint,
                   unfreeze_all)

DEFAULT_CFG = {
    "name": "baseline",
    "arch": "resnet18",
    "pretrained": True,
    "epochs": 8,
    "batch_size": 64,
    "eval_batch_size": 128,
    "optimizer": "adamw",
    "lr": 1e-3,
    "head_lr": None,
    "weight_decay": 1e-4,
    "scheduler": "none",
    "label_smoothing": 0.0,
    "augmentation": "none",
    "input_size": 224,
    "num_workers": 4,
    "seed": 20260910,
    "freeze_backbone_epochs": 0,
    "amp": True,
    "mid_checkpoint_every": 2,
    "limit_per_class": None,
}


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = dict(DEFAULT_CFG)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["train_dir"] = TRAIN_DIR
    cfg["val_dir"] = VAL_DIR
    return cfg


def build_optimizer(net, cfg, params):
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"], nesterov=True)
    raise ValueError(f"不支持的优化器：{cfg['optimizer']}")


@torch.no_grad()
def evaluate(net, loader, device, criterion, num_classes=51):
    net.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = net(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(labels.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "loss": total_loss / max(n, 1),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "accuracy": accuracy(y_true, y_pred),
    }, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="51 类车型分类训练")
    parser.add_argument("--config", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--augmentation", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, {
        "name": args.name, "arch": args.arch, "epochs": args.epochs, "lr": args.lr,
        "augmentation": args.augmentation, "scheduler": args.scheduler,
        "batch_size": args.batch_size, "limit_per_class": args.limit_per_class,
        "pretrained": False if args.no_pretrained else None,
    })
    ensure_dirs()
    set_seed(cfg["seed"])
    device = get_device()

    class_ids, class_names = build_mapping()
    print(f"[配置] {cfg['name']} | 结构={cfg['arch']} | 预训练={cfg['pretrained']} | "
          f"轮数={cfg['epochs']} | 增强={cfg['augmentation']} | 设备={describe_device()}")

    train_loader, val_loader, train_set, val_set = build_loaders(cfg)
    print(f"[数据] train {len(train_set)} 张 / val {len(val_set)} 张，共 {len(class_ids)} 类")

    net = build_model(cfg["arch"], len(class_ids), cfg["pretrained"]).to(device)
    # 可选：从已有 checkpoint 初始化（微调场景，保留已学到的 unknown 知识）
    if cfg.get("init_checkpoint"):
        ck = torch.load(cfg["init_checkpoint"], map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_state"])
        print(f"[初始化] 从 {cfg['init_checkpoint']} 加载权重（epoch={ck['epoch']}, "
              f"其最佳F1={round(float(ck['best_val_macro_f1']), 4)}），进入微调模式")
    # 可选：类别权重——unknown 训练样本远多于已知类时，给已知类更大 loss 权重，
    # 让模型不至于把已知车都推给 unknown（压误拒），同时不减少 unknown 样本量（保召回）。
    criterion_weights = None
    if cfg.get("known_class_weight"):
        w = torch.ones(len(class_ids))
        w[:len(class_ids) - 1] = float(cfg["known_class_weight"])   # 末位是 unknown，权重保持 1
        criterion_weights = w.to(device)
        print(f"[损失] 已知类 loss 权重 = {cfg['known_class_weight']}，unknown = 1.0")
    criterion = build_criterion("cross_entropy", cfg["label_smoothing"], criterion_weights)

    optimizer = build_optimizer(net, cfg, net.parameters())
    scheduler = None
    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif cfg["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg["epochs"] // 3, 1), gamma=0.3)

    scaler = torch.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    history, best_f1, best_epoch = [], -1.0, 0
    started = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # 迁移学习策略：前若干轮冻结主干只训分类头，之后解冻全网络微调
        if epoch == cfg["freeze_backbone_epochs"] + 1 and cfg["freeze_backbone_epochs"] > 0:
            params = unfreeze_all(net)
            optimizer = build_optimizer(net, cfg, params)
            if cfg["scheduler"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["epochs"] - epoch + 1)
            print("[策略] 主干已解冻，进入全网络微调")
        elif epoch == 1 and cfg["freeze_backbone_epochs"] > 0:
            params = freeze_backbone(net)
            optimizer = build_optimizer(net, cfg, params)
            print("[策略] 主干已冻结，仅训练分类头")

        net.train()
        running, seen, correct = 0.0, 0, 0
        t0 = time.time()
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg["amp"] and device.type == "cuda"):
                logits = net(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            if step % 20 == 0:
                print(f"  ep{epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f} acc={correct / seen:.4f}", flush=True)

        train_loss = running / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_metrics, _, _ = evaluate(net, val_loader, device, criterion, len(class_ids))
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"[第 {epoch} 轮] 训练 loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"验证 Macro-F1={val_metrics['macro_f1']:.4f} acc={val_metrics['accuracy']:.4f} | "
              f"{row['seconds']}s", flush=True)

        # 实时进度快照：供监控程序（monitor.py）读取，用户可实时看到训练进展
        elapsed = time.time() - started
        eta = elapsed / epoch * (cfg["epochs"] - epoch) if epoch else 0
        with open(os.path.join(LOGS_DIR, f"{cfg['name']}_progress.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "name": cfg["name"],
                "epoch": epoch,
                "total_epochs": cfg["epochs"],
                "train_loss": row["train_loss"],
                "train_acc": row["train_acc"],
                "val_macro_f1": row["val_macro_f1"],
                "best_val_macro_f1": round(best_f1, 4),
                "best_epoch": best_epoch,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1),
                "finished": False,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }, f, ensure_ascii=False)

        # 中期 checkpoint：便于回溯训练过程和做错误分析
        if cfg["mid_checkpoint_every"] and epoch % cfg["mid_checkpoint_every"] == 0 \
                and epoch != cfg["epochs"]:
            save_checkpoint(os.path.join(MODELS_DIR, f"{cfg['name']}_mid_ep{epoch}.pt"),
                            net, cfg, epoch=epoch, best_macro_f1=best_f1,
                            optimizer=None, tag=f"mid_ep{epoch}")

        # 用验证集 Macro-F1 选模型，与教师评分口径一致
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            save_checkpoint(os.path.join(MODELS_DIR, f"{cfg['name']}_best.pt"),
                            net, cfg, epoch=epoch, best_macro_f1=best_f1,
                            optimizer=None, tag="best")

    save_checkpoint(os.path.join(MODELS_DIR, f"{cfg['name']}_last.pt"),
                    net, cfg, epoch=cfg["epochs"], best_macro_f1=best_f1,
                    optimizer=optimizer, tag="last")

    # 标记训练结束（监控程序据此停止刷新）
    _pp = os.path.join(LOGS_DIR, f"{cfg['name']}_progress.json")
    if os.path.exists(_pp):
        with open(_pp, encoding="utf-8") as f:
            _d = json.load(f)
        _d["finished"] = True
        _d["best_val_macro_f1"] = round(best_f1, 4)
        _d["best_epoch"] = best_epoch
        _d["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(_pp, "w", encoding="utf-8") as f:
            json.dump(_d, f, ensure_ascii=False)

    # 发布一个稳定的模型入口：智能体与 Web 端只认 models/best.pt
    import shutil
    shutil.copyfile(os.path.join(MODELS_DIR, f"{cfg['name']}_best.pt"),
                    os.path.join(MODELS_DIR, "best.pt"))

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, f"{cfg['name']}_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(LOGS_DIR, f"{cfg['name']}_history.csv"), "w", encoding="utf-8") as f:
        keys = list(history[0].keys())
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    summary = {
        "name": cfg["name"], "arch": cfg["arch"], "epochs": cfg["epochs"],
        "augmentation": cfg["augmentation"], "lr": cfg["lr"], "scheduler": cfg["scheduler"],
        "label_smoothing": cfg["label_smoothing"],
        "freeze_backbone_epochs": cfg["freeze_backbone_epochs"],
        "best_val_macro_f1": round(best_f1, 4), "best_epoch": best_epoch,
        "final_val_macro_f1": round(history[-1]["val_macro_f1"], 4),
        "total_minutes": round((time.time() - started) / 60, 2),
        "device": describe_device(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, f"train_summary_{cfg['name']}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 最佳验证 Macro-F1 = {best_f1:.4f}（第 {best_epoch} 轮），"
          f"总耗时 {summary['total_minutes']} 分钟")


if __name__ == "__main__":
    main()
