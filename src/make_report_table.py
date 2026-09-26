"""从实验汇总与评估结果生成实训报告可直接引用的对比表。

用法：
    python src/make_report_table.py

读取 outputs/experiments_summary.csv 以及各实验的 *_metrics.json，
输出：
    outputs/report_experiment_table.md   对比实验表（Markdown，直接粘进报告）
    outputs/report_baseline_vs_best.md   基线与最佳改进的逐类差异
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUTS_DIR

SUMMARY = os.path.join(OUTPUTS_DIR, "experiments_summary.csv")

CJK_PAD = "  "


def read_rows() -> list[dict]:
    if not os.path.isfile(SUMMARY):
        raise SystemExit(f"还没有实验汇总文件：{SUMMARY}\n请先运行 python src/run_experiments.py")
    with open(SUMMARY, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metric_of(name: str) -> dict | None:
    path = os.path.join(OUTPUTS_DIR, f"{name}_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def experiment_table(rows: list[dict]) -> str:
    lines = [
        "# 对比实验汇总（验证集口径，51 类等权 Macro-F1）",
        "",
        "| 实验 | 模型结构 | 数据增强 | 学习率 | 调度 | 标签平滑 | 冻结轮数 | 最佳轮次 | **验证 Macro-F1** | 相对基线 | 耗时(分) |",
        "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base = next((float(r["best_val_macro_f1"]) for r in rows if r["name"] == "e0_baseline"), None)
    best = max(float(r["best_val_macro_f1"]) for r in rows)
    for r in rows:
        f1 = float(r["best_val_macro_f1"])
        delta = f"{f1 - base:+.4f}" if base is not None else "—"
        mark = " ★" if abs(f1 - best) < 1e-9 else ""
        lines.append(
            f"| `{r['name']}`{mark} | {r['arch']} | {r['augmentation']} | {r['lr']} | "
            f"{r['scheduler']} | {r['label_smoothing']} | {r['freeze_backbone_epochs']} | "
            f"{r['best_epoch']} | **{f1:.4f}** | {delta} | {r['total_minutes']} |"
        )
    lines += [
        "",
        "> 说明：所有实验使用同一份训练/验证数据划分、同一随机种子，仅改动单一变量以便归因；",
        "> 模型选择一律依据验证集 51 类 Macro-F1，未使用测试图片调参。★ 标记为该轮最佳。",
        "",
        "## 各实验 unknown 表现对比",
        "",
        "| 实验 | unknown 检出率 | unknown 误收率 | 已知类误拒率 | Accuracy |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        m = metric_of(r["name"])
        if not m:
            lines.append(f"| `{r['name']}` | — | — | — | — |")
            continue
        u = m["unknown"]
        lines.append(f"| `{r['name']}` | {u['unknown_detection_rate']:.4f} | "
                     f"{u['unknown_false_accept_rate']:.4f} | {u['known_rejection_rate']:.4f} | "
                     f"{m['accuracy']:.4f} |")
    return "\n".join(lines) + "\n"


def per_class_diff(base: dict, best: dict) -> str:
    def load(name):
        path = os.path.join(OUTPUTS_DIR, f"{name}_per_class.csv")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8-sig") as f:
            return {row["label"]: row for row in csv.DictReader(f)}

    b, t = load(base["name"]), load(best["name"])
    if not b or not t:
        return "> 缺少逐类指标文件，无法生成差异表（请先对两个实验各跑一次 evaluate.py）。\n"

    rows = []
    for lab, row in t.items():
        f1_new, f1_old = float(row["f1"]), float(b[lab]["f1"])
        rows.append((lab, row["name"], b[lab]["f1"], row["f1"], f1_new - f1_old))
    rows.sort(key=lambda r: r[4])
    lines = [
        f"# {base['name']} → {best['name']} 逐类 F1 变化",
        "",
        f"- 基线验证 Macro-F1：**{base['best_val_macro_f1']}**",
        f"- 最佳验证 Macro-F1：**{best['best_val_macro_f1']}**",
        f"- 提升：**{float(best['best_val_macro_f1']) - float(base['best_val_macro_f1']):+.4f}**",
        "",
        "## 退步最明显的 10 类（改进后反而变差，需在报告中解释）",
        "",
        "| 类别 ID | 车型 | 基线 F1 | 改进 F1 | 变化 |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for lab, name, old, new, d in rows[:10]:
        lines.append(f"| {lab} | {name} | {old} | {new} | {d:+.4f} |")
    lines += ["", "## 提升最明显的 10 类", "",
              "| 类别 ID | 车型 | 基线 F1 | 改进 F1 | 变化 |",
              "| :--- | :--- | ---: | ---: | ---: |"]
    for lab, name, old, new, d in rows[::-1][:10]:
        lines.append(f"| {lab} | {name} | {old} | {new} | {d:+.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = read_rows()
    table = experiment_table(rows)
    with open(os.path.join(OUTPUTS_DIR, "report_experiment_table.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print("已生成 outputs/report_experiment_table.md\n")
    print(table)

    base = next((r for r in rows if r["name"] == "e0_baseline"), None)
    best = max(rows, key=lambda r: float(r["best_val_macro_f1"]))
    if base and base["name"] != best["name"]:
        diff = per_class_diff(base, best)
        with open(os.path.join(OUTPUTS_DIR, "report_baseline_vs_best.md"), "w", encoding="utf-8") as f:
            f.write(diff)
        print("\n已生成 outputs/report_baseline_vs_best.md")


if __name__ == "__main__":
    main()
