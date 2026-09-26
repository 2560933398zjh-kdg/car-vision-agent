"""在验证集上评估模型：51 类 Macro-F1、每类指标、混淆矩阵、unknown 专项指标、典型错误样本。

用法：
    python src/evaluate.py --checkpoint models/resnet18_baseline_best.pt
    python src/evaluate.py --checkpoint models/xxx.pt --split val --tag baseline

产出（outputs/）：
    <tag>_metrics.json            总体指标（含 51 类 Macro-F1）
    <tag>_per_class.csv           每类 precision / recall / F1 / support
    <tag>_confusion_matrix.csv    51×51 混淆矩阵
    <tag>_confusion_matrix.png    混淆矩阵热图（可直接放报告）
    <tag>_top_confusions.csv      最易混淆的类别对
    <tag>_errors.csv              典型错误样本清单
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (IMG_EXTS, OUTPUTS_DIR, UNKNOWN, ensure_dirs, get_device)
from dataset import CarDataset, build_eval_transform
from metrics import (accuracy, confusion_matrix, macro_f1, per_class_report,
                     top_confusions, unknown_metrics)
from model import load_checkpoint

UNKNOWN_INDEX = 50


def plot_confusion(conf: np.ndarray, labels: list[str], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors
        import matplotlib.pyplot as plt
        # 图中有中文标签，必须指定中文字体，否则渲染成方块
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        print("[提示] 未安装 matplotlib，跳过混淆矩阵热图（CSV 已生成）")
        return

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(conf, cmap="Blues",
                   norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(int(conf.max()), 2)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6, color="#2c2c2a")
    ax.set_yticklabels(labels, fontsize=6, color="#2c2c2a")
    ax.set_xlabel("预测类别", fontsize=10, color="#2c2c2a")
    ax.set_ylabel("真实类别", fontsize=10, color="#2c2c2a")
    ax.set_title("51 类混淆矩阵（对数色阶，对角线为正确预测）", fontsize=12, color="#2c2c2a")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.add_patch(plt.Rectangle((UNKNOWN_INDEX - 0.5, UNKNOWN_INDEX - 0.5), 1, 1,
                               fill=False, edgecolor="#D85A30", lw=1.5))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def collect_predictions(net, loader, device):
    y_true, y_pred, confs, paths = [], [], [], []
    offset = 0
    samples = loader.dataset.samples
    for images, labels in loader:
        logits = net(images.to(device))
        prob = torch.softmax(logits.float(), dim=1)
        conf, pred = prob.max(dim=1)
        y_true.append(labels.numpy())
        y_pred.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
        paths.extend(samples[offset + i][0] for i in range(len(labels)))
        offset += len(labels)
    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(confs), paths)


def evaluate(checkpoint: str, split: str, tag: str | None, batch_size: int, workers: int) -> dict:
    ensure_dirs()
    device = get_device()
    net, ckpt = load_checkpoint(checkpoint, map_location="cpu")
    net.to(device)

    cfg = {"input_size": ckpt["preprocess"]["input_size"],
           "resize_size": ckpt["preprocess"].get("resize_size")}
    labels = [ckpt["index_to_label"][str(i)] for i in range(ckpt["num_classes"])]
    names = ckpt["class_names"]

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", split)
    ds = CarDataset(root, build_eval_transform(cfg))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    y_true, y_pred, confs, paths = collect_predictions(net, loader, device)
    conf = confusion_matrix(y_true, y_pred, ckpt["num_classes"])
    n = ckpt["num_classes"]
    tag = tag or os.path.splitext(os.path.basename(checkpoint))[0]

    report = per_class_report(conf, labels, names)
    result = {
        "checkpoint": os.path.basename(checkpoint),
        "arch": ckpt["arch"],
        "split": split,
        "num_classes": n,
        "images": int(len(y_true)),
        "macro_f1": round(macro_f1(y_true, y_pred, n), 4),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "unknown": unknown_metrics(conf, UNKNOWN_INDEX),
        "worst_classes": sorted(report, key=lambda r: r["f1"])[:10],
    }

    with open(os.path.join(OUTPUTS_DIR, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUTS_DIR, f"{tag}_per_class.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "label", "name", "support", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(report)

    np.savetxt(os.path.join(OUTPUTS_DIR, f"{tag}_confusion_matrix.csv"), conf,
               fmt="%d", delimiter=",", header=",".join(labels), comments="")
    plot_confusion(conf, labels, os.path.join(OUTPUTS_DIR, f"{tag}_confusion_matrix.png"))

    with open(os.path.join(OUTPUTS_DIR, f"{tag}_top_confusions.csv"), "w", newline="", encoding="utf-8-sig") as f:
        rows = top_confusions(conf, labels, names, k=20)
        w = csv.DictWriter(f, fieldnames=["true", "true_name", "pred", "pred_name", "count", "share_of_true"])
        w.writeheader()
        w.writerows(rows)

    # 典型错误样本：只记被误判的图，并标注真实/预测类别与置信度
    with open(os.path.join(OUTPUTS_DIR, f"{tag}_errors.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "true_name", "pred_label", "pred_name", "confidence"])
        for i in np.where(y_true != y_pred)[0]:
            w.writerow([os.path.basename(paths[i]), labels[y_true[i]], names[y_true[i]],
                        labels[y_pred[i]], names[y_pred[i]], round(float(confs[i]), 4)])

    print(f"[评估] {result['checkpoint']} @ {split}")
    print(f"  图片数        {result['images']}")
    print(f"  Macro-F1(51)  {result['macro_f1']:.4f}   <- 本实训唯一评分指标")
    print(f"  Accuracy      {result['accuracy']:.4f}")
    print(f"  unknown 检出率 {result['unknown']['unknown_detection_rate']:.4f} "
          f"(误收 {result['unknown']['unknown_false_accept_rate']:.4f})")
    print(f"  已知类误拒率   {result['unknown']['known_rejection_rate']:.4f}")
    worst = "，".join(f"{r['label']}({r['f1']:.2f})" for r in result["worst_classes"][:5])
    print(f"  最差 5 类     {worst}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="验证集评估（51 类 Macro-F1）")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--tag", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    evaluate(a.checkpoint, a.split, a.tag, a.batch_size, a.workers)
