"""评估指标：51 类 Macro-F1、每类指标、混淆矩阵、未知类专项指标。

Macro-F1 是本实训唯一的模型效果指标，口径必须与教师评分一致：
先算每类 F1，再对 51 类等权平均；某类分母为 0 时该类 F1 记 0；unknown 与其他类同权。
"""
from __future__ import annotations

import numpy as np


def f1_per_class(conf: np.ndarray) -> np.ndarray:
    """由混淆矩阵（行为真实类、列为预测类）逐类计算 F1，分母为 0 时返回 0。"""
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 51) -> float:
    conf = confusion_matrix(y_true, y_pred, num_classes)
    return float(f1_per_class(conf).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 51) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (y_true.astype(int), y_pred.astype(int)), 1)
    return conf


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def per_class_report(conf: np.ndarray, labels: list[str], names: list[str]) -> list[dict]:
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    support = conf.sum(axis=1)
    f1 = f1_per_class(conf)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return [{
        "index": i,
        "label": labels[i],
        "name": names[i] if i < len(names) else labels[i],
        "support": int(support[i]),
        "precision": round(float(precision[i]), 4),
        "recall": round(float(recall[i]), 4),
        "f1": round(float(f1[i]), 4),
    } for i in range(len(labels))]


def unknown_metrics(conf: np.ndarray, unknown_index: int = 50) -> dict:
    """unknown 检出率（unknown 被认成 unknown 的比例）与已知类误拒率（50 类被认成 unknown 的比例）。"""
    tp = conf[unknown_index, unknown_index]
    support = conf[unknown_index].sum()
    leak = conf[unknown_index, :].sum() - tp           # unknown 被误判成某个具体车型
    known_to_unknown = conf[:unknown_index, unknown_index].sum()
    known_total = conf[:unknown_index, :].sum()
    return {
        "unknown_detection_rate": round(float(tp / support), 4) if support else 0.0,
        "unknown_false_accept_rate": round(float(leak / support), 4) if support else 0.0,
        "known_rejection_rate": round(float(known_to_unknown / known_total), 4) if known_total else 0.0,
        "unknown_support": int(support),
    }


def top_confusions(conf: np.ndarray, labels: list[str], names: list[str], k: int = 12) -> list[dict]:
    """排除对角线后，找出最容易混淆的类别对。"""
    pairs = []
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            if i != j and conf[i, j] > 0:
                pairs.append({
                    "true": labels[i], "true_name": names[i],
                    "pred": labels[j], "pred_name": names[j],
                    "count": int(conf[i, j]),
                    "share_of_true": round(float(conf[i, j] / max(conf[i].sum(), 1)), 4),
                })
    pairs.sort(key=lambda r: -r["count"])
    return pairs[:k]
