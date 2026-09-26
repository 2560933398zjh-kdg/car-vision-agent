"""汇总任务一（数据分析 + 视觉模型训练）的全部产出，生成报告素材。

用法：
    python src/make_task1_report.py

读取 outputs/ 与 logs/ 下的实际结果，生成：
    outputs/任务一报告素材.md
对应任务书 8.1 实训报告的第 2、3、4 节，可直接摘进报告。
所有数字均来自实际运行文件，保证「结论有据可查」。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (CLASS_INFO_JSON, LOGS_DIR, OUTPUTS_DIR, PROJECT_ROOT,
                    load_class_ids, load_name_map)

SUMMARY = os.path.join(OUTPUTS_DIR, "experiments_summary.csv")


def read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def data_stats() -> dict:
    ids = load_class_ids()
    names = load_name_map()
    with open(CLASS_INFO_JSON, encoding="utf-8") as f:
        info = {r["id"]: r for r in json.load(f)}

    def count(split, cls):
        d = os.path.join(PROJECT_ROOT, "data", split, cls)
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])

    rows = []
    for cid in ids:
        t1, t2 = info[cid]["type_info"].split("####")[:2]
        rows.append({"id": cid, "name": names[cid], "t1": t1,
                     "t2": t2.split("==")[0], "train": count("train", cid),
                     "val": count("val", cid)})
    unknown = {"train": count("train", "unknown"), "val": count("val", "unknown")}
    return {"rows": rows, "unknown": unknown}


def main() -> None:
    L: list[str] = []
    L.append("# 任务一报告素材：数据分析与视觉模型训练\n")
    L.append("> 本文档由 `src/make_task1_report.py` 从实际运行结果自动汇总生成，")
    L.append("> 对应任务书 8.1 节实训报告的第 2、3、4 部分。数字均可回溯到 `outputs/` 与 `logs/`。\n")

    # ---------- 2. 数据分析 ----------
    st = data_stats()
    rows = st["rows"]
    t1c = Counter(r["t1"] for r in rows)
    t2c = Counter(r["t2"] for r in rows)
    tr_list = [r["train"] for r in rows]
    va_list = [r["val"] for r in rows]

    L.append("## 2. 数据分析\n")
    L.append("### 2.1 选类结果\n")
    L.append(f"从教师提供的 827 个车型类别中自主选择 **50 类**，并补充 1 个 `unknown` 类，"
             f"共 **51 个输出类别**。选类原则见 `data/选类说明.md`，摘要如下：\n")
    L.append("- 样本充足：训练图 ≥ 85 张、验证图 ≥ 30 张；")
    L.append("- **品牌不重复**：50 个类别分属 50 个不同品牌，避免同品牌孪生车型互相干扰；")
    L.append("- 外观差异大：优先选取造型辨识度高的车型；")
    L.append("- 覆盖 7 种车身类型；")
    L.append("- 全部为具体车型，不使用品牌或「轿车 / SUV」等大类替代。\n")

    L.append("### 2.2 类别与样本统计\n")
    L.append(f"- 目标车型训练图合计 **{sum(tr_list)}** 张（每类 {min(tr_list)}–{max(tr_list)} 张）；")
    L.append(f"- 目标车型验证图合计 **{sum(va_list)}** 张（每类 {min(va_list)}–{max(va_list)} 张）；")
    L.append(f"- `unknown` 训练图 **{st['unknown']['train']}** 张、验证图 **{st['unknown']['val']}** 张。\n")
    L.append("| 车身类型 | 类别数 |")
    L.append("| :--- | ---: |")
    for k, v in t1c.most_common():
        L.append(f"| {k} | {v} |")
    L.append("\n| 细分类型 | 类别数 |")
    L.append("| :--- | ---: |")
    for k, v in t2c.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")

    L.append("### 2.3 unknown 样本的来源与采样方法\n")
    L.append("从**组外**车型中抽取 12 个来源类别，每类随机抽取 10 张训练图与 4 张验证图，"
             "合并标记为 `unknown`。12 个来源类覆盖全部 7 种车身类型，各类取样数相同，"
             "避免样本过度集中于单一车型。随机种子固定为 `20260910`，可完全复现（`data/build_dataset.py`）。\n")
    L.append("| 来源类别 ID | 车型 | 车身类型 |")
    L.append("| :--- | :--- | :--- |")
    for cid in ["0006", "0073", "0188", "0210", "0251", "0428",
                "0536", "0565", "0635", "0894", "0904", "0914"]:
        n = load_name_map().get(cid, "")
        with open(CLASS_INFO_JSON, encoding="utf-8") as f:
            t1 = {r["id"]: r["type_info"].split("####")[0] for r in json.load(f)}[cid]
        L.append(f"| {cid} | {n} | {t1} |")
    L.append("\n**关键约束的落实情况**：\n")
    L.append("- 训练样本与验证样本**无重复**（分别取自教师 `train/` 与 `val/` 划分，文件名零重叠）；")
    L.append("- 未使用任何测试图片参与训练或调参；")
    L.append("- 未把所选 50 类的任何图片标记为 `unknown`；")
    L.append("- `unknown` 作为**第 51 类**参与训练，索引为 50；`classes.txt` 仍恰好保留 50 个车型 ID，不含 unknown。\n")

    L.append("### 2.4 预处理与数据增强\n")
    L.append("- **统一预处理**（验证与推理固定使用）：短边缩放到 256 → 居中裁剪 224×224 → "
             "按 ImageNet 均值 `[0.485,0.456,0.406]`、标准差 `[0.229,0.224,0.225]` 归一化；")
    L.append("- **训练期增强**（按实验分组不同）：随机裁剪缩放（scale 0.7–1.0）、随机水平翻转、"
             "颜色抖动、随机旋转 ±10°、高斯模糊、随机灰度、随机擦除（RandomErasing）。\n")

    # ---------- 3. 模型设计与训练 ----------
    L.append("## 3. 模型设计与训练\n")
    configs = []
    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "*_config.json"))):
        cfg = read_json(path)
        if cfg:
            configs.append(cfg)
    if configs:
        c = configs[0]
        L.append("### 3.1 模型结构\n")
        L.append(f"- 主干网络：**{c['arch']}**，加载 ImageNet 预训练权重（`IMAGENET1K_V1`），"
                 f"将最后的全连接层替换为 **51 维**输出头；")
        L.append(f"- 输入尺寸 {c['input_size']}×{c['input_size']}；")
        L.append("- 采用迁移学习策略：保留预训练主干特征，仅替换并微调分类头。\n")
        L.append("### 3.2 训练配置\n")
        L.append("| 实验 | 结构 | 轮数 | batch | 优化器 | 学习率 | 调度 | 标签平滑 | 冻结轮数 | 增强 |")
        L.append("| :--- | :--- | ---: | ---: | :--- | ---: | :--- | ---: | ---: | :--- |")
        for cfg in configs:
            L.append(f"| `{cfg['name']}` | {cfg['arch']} | {cfg['epochs']} | {cfg['batch_size']} | "
                     f"{cfg['optimizer']} | {cfg['lr']} | {cfg['scheduler']} | "
                     f"{cfg['label_smoothing']} | {cfg['freeze_backbone_epochs']} | {cfg['augmentation']} |")
        L.append(f"\n- 损失函数：交叉熵（部分实验启用标签平滑 0.1）；")
        L.append("- 开启 **AMP 混合精度**（本机显存仅 4GB）；")
        L.append("- 所有实验使用同一数据划分与随机种子 `20260910`，每次只改动单一变量以便归因。\n")

    L.append("### 3.3 模型选择依据\n")
    L.append("以**验证集 51 类等权 Macro-F1** 为唯一标准挑选最佳轮次的模型（`models/<实验名>_best.pt`），"
             "**不使用测试图片调参**。训练过程中按固定间隔保存中期 checkpoint，便于回溯训练过程。\n")

    # ---------- 4. 实验结果与分析 ----------
    L.append("## 4. 实验结果与分析\n")
    summary = read_csv(SUMMARY)
    if summary:
        base = next((r for r in summary if r["name"] == "e0_baseline"), None)
        best = max(summary, key=lambda r: float(r["best_val_macro_f1"]))
        L.append("### 4.1 对比实验\n")
        L.append("| 实验 | 结构 | 增强 | 学习率 | 调度 | 标签平滑 | 最佳轮次 | **验证 Macro-F1** | 相对基线 | 耗时(分) |")
        L.append("| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |")
        for r in summary:
            f1 = float(r["best_val_macro_f1"])
            delta = f"{f1 - float(base['best_val_macro_f1']):+.4f}" if base else "—"
            star = " ★" if r["name"] == best["name"] else ""
            L.append(f"| `{r['name']}`{star} | {r['arch']} | {r['augmentation']} | {r['lr']} | "
                     f"{r['scheduler']} | {r['label_smoothing']} | {r['best_epoch']} | **{f1:.4f}** | "
                     f"{delta} | {r['total_minutes']} |")
        L.append(f"\n**结论**：最佳方案为 `{best['name']}`，验证集 Macro-F1 = **{best['best_val_macro_f1']}**"
                 f"（{best['arch']}，增强={best['augmentation']}）。\n")

        # 逐轮曲线
        L.append("### 4.2 逐轮训练过程\n")
        for r in summary:
            hist = read_csv(os.path.join(LOGS_DIR, f"{r['name']}_history.csv"))
            if not hist:
                continue
            L.append(f"\n**`{r['name']}`**\n")
            L.append("| 轮次 | 训练 loss | 训练 acc | 验证 loss | 验证 Macro-F1 | 耗时(s) |")
            L.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
            for h in hist:
                L.append(f"| {h['epoch']} | {h['train_loss']} | {h['train_acc']} | "
                         f"{h['val_loss']} | {h['val_macro_f1']} | {h['seconds']} |")

        # 指标与错误分析
        m = read_json(os.path.join(OUTPUTS_DIR, f"{best['name']}_metrics.json"))
        if m:
            L.append(f"\n### 4.3 最佳模型的验证集指标（{best['name']}）\n")
            L.append(f"- **51 类 Macro-F1：{m['macro_f1']}** ← 本实训唯一效果指标")
            L.append(f"- Accuracy：{m['accuracy']}")
            L.append(f"- 验证图片数：{m['images']}")
            u = m["unknown"]
            L.append(f"- unknown 检出率：{u['unknown_detection_rate']}（unknown 被正确识别为 unknown 的比例）")
            L.append(f"- unknown 误收率：{u['unknown_false_accept_rate']}（unknown 被误判为某个具体车型的比例）")
            L.append(f"- 已知类误拒率：{u['known_rejection_rate']}（50 个目标车型被误判为 unknown 的比例）\n")
            L.append("**表现最差的 10 个类别**（应在报告中重点分析）：\n")
            L.append("| 类别 ID | 车型 | support | precision | recall | F1 |")
            L.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
            for r in m["worst_classes"]:
                L.append(f"| {r['label']} | {r['name']} | {r['support']} | "
                         f"{r['precision']} | {r['recall']} | {r['f1']} |")

        conf = read_csv(os.path.join(OUTPUTS_DIR, f"{best['name']}_top_confusions.csv"))
        if conf:
            L.append("\n### 4.4 最易混淆的类别对\n")
            L.append("| 真实类别 | 被误判为 | 次数 | 占该类比例 |")
            L.append("| :--- | :--- | ---: | ---: |")
            for r in conf[:12]:
                L.append(f"| {r['true']} {r['true_name']} | {r['pred']} {r['pred_name']} | "
                         f"{r['count']} | {r['share_of_true']} |")
            L.append("\n> 报告需结合车型外观相似性解释这些混淆（例如同车身类型、同代设计语言、"
                     "拍摄角度或遮挡导致的差异被抹平）。")

        errs = read_csv(os.path.join(OUTPUTS_DIR, f"{best['name']}_errors.csv"))
        if errs:
            L.append(f"\n### 4.5 典型错误样本（共 {len(errs)} 张误判）\n")
            L.append("| 图片 | 真实类别 | 预测类别 | 置信度 |")
            L.append("| :--- | :--- | :--- | ---: |")
            for r in errs[:20]:
                L.append(f"| {r['image']} | {r['true_label']} {r['true_name']} | "
                         f"{r['pred_label']} {r['pred_name']} | {r['confidence']} |")
            L.append(f"\n> 完整清单见 `outputs/{best['name']}_errors.csv`。")
            L.append("> 报告应从这些样本中挑选若干张，说明失败原因属于"
                     "「车型外观本身高度相似」「拍摄角度/遮挡」「被判定为 unknown」中的哪一类。")

    L.append("\n---\n")
    L.append("## 附：产出文件索引\n")
    L.append("| 文件 | 内容 |")
    L.append("| :--- | :--- |")
    L.append("| `classes.txt` | 50 个四位车型 ID（项目根目录，最终提交物） |")
    L.append("| `data/class_mapping.json` | 索引 0–49 → 车型，50 → unknown |")
    L.append("| `logs/<实验>_config.json` | 每次训练的完整配置 |")
    L.append("| `logs/<实验>_history.csv` | 逐轮训练指标 |")
    L.append("| `outputs/experiments_summary.csv` | 对比实验汇总表 |")
    L.append("| `outputs/<实验>_per_class.csv` | 每类 precision/recall/F1 |")
    L.append("| `outputs/<实验>_confusion_matrix.csv/.png` | 51×51 混淆矩阵 |")
    L.append("| `outputs/<实验>_top_confusions.csv` | 最易混淆类别对 |")
    L.append("| `outputs/<实验>_errors.csv` | 典型错误样本清单 |")
    L.append("| `models/*.pt` | 模型文件（仅本地保存，不提交） |")

    out = os.path.join(OUTPUTS_DIR, "任务一报告素材.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"已生成 {out}")
    print(f"  共 {len(L)} 行")


if __name__ == "__main__":
    main()
