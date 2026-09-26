"""批量跑对比实验并汇总结果。

用法：
    python src/run_experiments.py                       # 跑 configs/ 下全部实验
    python src/run_experiments.py --only e0_baseline e1_aug
    python src/run_experiments.py --smoke               # 用少量样本快速验证流程

每跑完一个实验，就把它记录进 outputs/experiments_summary.csv，便于直接写进报告的对比表。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import CONFIGS_DIR, LOGS_DIR, OUTPUTS_DIR, PROJECT_ROOT

SUMMARY = os.path.join(OUTPUTS_DIR, "experiments_summary.csv")
FIELDS = ["name", "arch", "epochs", "augmentation", "lr", "scheduler", "label_smoothing",
          "freeze_backbone_epochs", "best_epoch", "best_val_macro_f1", "final_val_macro_f1",
          "total_minutes", "device"]


def run_one(config_path: str, extra: list[str]) -> dict | None:
    name = json.load(open(config_path, encoding="utf-8"))["name"]
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "train.py"),
           "--config", config_path, *extra]
    print(f"\n{'=' * 68}\n[实验] {name}\n{'=' * 68}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        print(f"[实验] {name} 训练失败（返回码 {proc.returncode}），跳过汇总")
        return None
    path = os.path.join(OUTPUTS_DIR, f"train_summary_{name}.json")
    if not os.path.isfile(path):
        return None
    summary = json.load(open(path, encoding="utf-8"))
    summary["wall_minutes"] = round((time.time() - started) / 60, 2)
    return summary


def append_summary(rows: list[dict]) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(SUMMARY, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})


def existing_summaries() -> dict[str, dict]:
    """扫描 outputs/ 下已完成的实验汇总，使批量实验可以断点续跑。"""
    found = {}
    for path in glob.glob(os.path.join(OUTPUTS_DIR, "train_summary_*.json")):
        try:
            s = json.load(open(path, encoding="utf-8"))
            found[s["name"]] = s
        except Exception:
            continue
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="批量对比实验")
    p.add_argument("--only", nargs="*", default=None, help="只跑指定实验名")
    p.add_argument("--smoke", action="store_true", help="冒烟模式：2 轮 + 每类 20 张")
    p.add_argument("--force", action="store_true", help="已完成的实验也重跑")
    a = p.parse_args()

    configs = sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.json")))
    if a.only:
        configs = [c for c in configs if json.load(open(c, encoding="utf-8"))["name"] in a.only]
    if not configs:
        raise SystemExit("没有匹配的实验配置")

    extra = ["--epochs", "2", "--limit-per-class", "20"] if a.smoke else []

    # 先把已有结果读进来，避免中断后全部重跑
    done = {} if a.force else existing_summaries()
    rows = list(done.values())
    todo = [c for c in configs if json.load(open(c, encoding="utf-8"))["name"] not in done]

    print(f"共 {len(configs)} 个实验：已完成 {len(configs) - len(todo)} 个，本次待跑 {len(todo)} 个")
    if done:
        print(f"  已跳过：{sorted(done)}")
    if todo:
        print(f"  待执行：{[os.path.basename(c) for c in todo]}")
    append_summary(rows)

    for path in todo:
        summary = run_one(path, extra)
        if summary:
            rows.append(summary)
            append_summary(rows)
            print(f"[汇总] {summary['name']} 最佳 Macro-F1 = {summary['best_val_macro_f1']}")

    if rows:
        best = max(rows, key=lambda r: r["best_val_macro_f1"])
        print(f"\n{'=' * 68}\n最佳实验：{best['name']}  Macro-F1 = {best['best_val_macro_f1']}"
              f"（{best['arch']}，增强={best['augmentation']}）")
        print(f"对比表已写入 {SUMMARY}")
        print("注意：对比表只反映验证集表现，最终结论仍需结合混淆矩阵与错误样本分析。")


if __name__ == "__main__":
    main()
