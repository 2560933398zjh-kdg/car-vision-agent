"""在验证集上确定置信度拒识阈值。

任务书要求：低置信度处理可设置额外的拒识阈值，且**阈值须通过验证集确定**。
本脚本扫描不同阈值下的 51 类 Macro-F1，输出曲线供报告引用，并给出推荐阈值。

规则：若最大类别概率 < 阈值，则把该图判为 unknown（索引 50）。

用法：
    python src/tune_threshold.py --checkpoint models/best.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUTS_DIR, PROJECT_ROOT, get_device
from dataset import CarDataset, build_eval_transform
from metrics import (confusion_matrix, f1_per_class, macro_f1, unknown_metrics)
from model import load_checkpoint

UNKNOWN_INDEX = 50


@torch.no_grad()
def collect(net, loader, device):
    probs, y_true = [], []
    for images, labels in loader:
        p = torch.softmax(net(images.to(device)).float(), dim=1)
        probs.append(p.cpu().numpy())
        y_true.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(y_true)


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    pred = probs.argmax(axis=1)
    if threshold <= 0:
        return pred
    low = probs.max(axis=1) < threshold
    pred = pred.copy()
    pred[low] = UNKNOWN_INDEX
    return pred


def main() -> None:
    p = argparse.ArgumentParser(description="验证集置信度阈值搜索")
    p.add_argument("--checkpoint", default=os.path.join("models", "best.pt"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grid", type=float, nargs="*", default=None)
    a = p.parse_args()

    device = get_device()
    net, ckpt = load_checkpoint(a.checkpoint, map_location="cpu")
    net.to(device)

    root = os.path.join(PROJECT_ROOT, "data", "val")
    cfg = {"input_size": ckpt["preprocess"]["input_size"]}
    ds = CarDataset(root, build_eval_transform(cfg))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    probs, y_true = collect(net, loader, device)

    n = ckpt["num_classes"]
    base_f1 = macro_f1(y_true, probs.argmax(axis=1), n)
    print(f"[阈值搜索] 验证集 {len(y_true)} 张，{n} 类")
    print(f"  不设阈值（threshold=0）Macro-F1 = {base_f1:.4f}\n")

    grid = a.grid if a.grid is not None else [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]
    rows = []
    for t in grid:
        pred = apply_threshold(probs, t)
        f1 = macro_f1(y_true, pred, n)
        conf = confusion_matrix(y_true, pred, n)
        u = unknown_metrics(conf, UNKNOWN_INDEX)
        f1s = f1_per_class(conf)
        rows.append({
            "threshold": round(float(t), 3),
            "macro_f1": round(float(f1), 4),
            "unknown_f1": round(float(f1s[UNKNOWN_INDEX]), 4),
            "unknown_detection_rate": u["unknown_detection_rate"],
            "unknown_false_accept_rate": u["unknown_false_accept_rate"],
            "known_rejection_rate": u["known_rejection_rate"],
            "delta_vs_none": round(float(f1 - base_f1), 4),
        })

    best = max(rows, key=lambda r: r["macro_f1"])
    out = os.path.join(OUTPUTS_DIR, "threshold_sweep.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"  {'阈值':>6} {'Macro-F1':>10} {'unknown F1':>11} {'检出率':>8} {'已知类误拒':>10} {'相对无阈值':>10}")
    for r in rows:
        flag = "  <- 最佳" if r is best else ""
        print(f"  {r['threshold']:>6.2f} {r['macro_f1']:>10.4f} {r['unknown_f1']:>11.4f} "
              f"{r['unknown_detection_rate']:>8.4f} {r['known_rejection_rate']:>10.4f} "
              f"{r['delta_vs_none']:>+10.4f}{flag}")

    print(f"\n推荐阈值 = {best['threshold']}，验证集 Macro-F1 = {best['macro_f1']:.4f}"
          f"（相对不设阈值 {best['delta_vs_none']:+.4f}）")
    print(f"曲线已写入 {out}")
    print("\n注意：阈值只依据验证集确定，未使用测试图片；报告中须说明该阈值的来源与代价"
          "（已知类误拒率会随之上升）。")


if __name__ == "__main__":
    main()
