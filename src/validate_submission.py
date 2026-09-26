"""提交物自检：按任务书规定逐条校验 classes.txt 与 predictions.csv。

用法：
    python src/validate_submission.py --predictions outputs/predictions.csv
    python src/validate_submission.py --predictions outputs/predictions.csv --truth data/val

校验项对应任务书第九节「提交前自查清单」与 4.1、8 节的格式规定。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import IMG_EXTS, PROJECT_ROOT, load_class_ids

OK, BAD = "[通过]", "[失败]"
issues: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {OK if cond else BAD} {msg}")
    if not cond:
        issues.append(msg)


def check_classes_txt() -> list[str]:
    print("\n=== classes.txt ===")
    path = os.path.join(PROJECT_ROOT, "classes.txt")
    check(os.path.isfile(path), f"文件存在：{path}")
    raw = open(path, "rb").read()
    check(not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 无 BOM")
    check(b"\r" not in raw, "无 Windows 换行符（统一 \\n）")

    lines = [l.strip() for l in raw.decode("utf-8").split("\n") if l.strip()]
    check(len(lines) == 50, f"恰好 50 行（实际 {len(lines)} 行）")
    check(len(set(lines)) == 50, "50 个 ID 不重复")
    check(all(len(x) == 4 and x.isdigit() for x in lines), "每个 ID 恰好 4 位数字（保留前导零）")
    check(lines == sorted(lines), "按类别 ID 升序排列")
    check("unknown" not in "".join(lines), "不包含 unknown")

    with open(os.path.join(PROJECT_ROOT, "cls_info", "class_info.json"), encoding="utf-8") as f:
        import json
        valid = {r["id"] for r in json.load(f)}
    outside = [x for x in lines if x not in valid]
    check(not outside, f"所有 ID 均属于教师提供的 827 类（越界：{outside or '无'}）")
    return lines


def check_predictions(path: str, class_ids: list[str], truth_dir: str | None) -> None:
    print(f"\n=== predictions.csv：{path} ===")
    check(os.path.isfile(path), "文件存在")
    raw = open(path, "rb").read()
    check(raw.startswith(b"image_id,predicted_label,confidence"), "表头固定为 image_id,predicted_label,confidence")
    check(b"\xef\xbb\xbf" not in raw[:3], "UTF-8 无 BOM")

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) > 0, f"记录数大于 0（实际 {len(rows)} 条）")

    ids = [r["image_id"] for r in rows]
    check(len(ids) == len(set(ids)), "image_id 不重复")

    bad_label = [r["image_id"] for r in rows
                 if not (r["predicted_label"] == "unknown" or
                         (len(r["predicted_label"]) == 4 and r["predicted_label"].isdigit()
                          and r["predicted_label"] in class_ids))]
    check(not bad_label, f"predicted_label 为保留前导零的四位 ID 或小写 unknown（异常：{bad_label[:5] or '无'}）")

    bad_conf = []
    for r in rows:
        try:
            v = float(r["confidence"])
            if not (0.0 <= v <= 1.0):
                bad_conf.append(r["image_id"])
        except ValueError:
            bad_conf.append(r["image_id"])
    check(not bad_conf, f"confidence 为 0~1 的有效数值（异常：{bad_conf[:5] or '无'}）")

    check(all(r["image_id"].lower().endswith(IMG_EXTS) for r in rows), "image_id 含扩展名")

    if truth_dir and os.path.isdir(truth_dir):
        print("\n=== 与真实标签一致性核对 ===")
        truth = {}
        for cls in os.listdir(truth_dir):
            d = os.path.join(truth_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(IMG_EXTS):
                    truth[f] = cls
        missing = [i for i in ids if i not in truth]
        extra = [k for k in truth if k not in set(ids)]
        check(not missing, f"每张图片都有预测记录（缺失 {len(missing)} 张）")
        check(not extra, f"没有遗漏图片（漏 {len(extra)} 张）")
        hit = sum(1 for r in rows if truth.get(r["image_id"]) == r["predicted_label"])
        acc = hit / len(rows) if rows else 0
        print(f"  准确率（对 {len(rows)} 张）：{acc:.4f}")
        print("  说明：该准确率应与 evaluate.py 的 Accuracy 一致，用于交叉验证两者的标签映射相同。")


def main() -> None:
    p = argparse.ArgumentParser(description="提交物自检")
    p.add_argument("--predictions", default=os.path.join(PROJECT_ROOT, "outputs", "predictions.csv"))
    p.add_argument("--truth", default=None, help="带真实标签的目录（按类别子目录组织），用于一致性核对")
    a = p.parse_args()

    print("任务书提交物自查")
    print("=" * 60)
    class_ids = []
    try:
        class_ids = check_classes_txt()
    except Exception as exc:
        print(f"  {BAD} classes.txt 读取异常：{exc}")
    try:
        check_predictions(a.predictions, class_ids, a.truth)
    except Exception as exc:
        print(f"  {BAD} predictions.csv 校验异常：{exc}")

    print("\n" + "=" * 60)
    if issues:
        print(f"共 {len(issues)} 项未通过：")
        for i, m in enumerate(issues, 1):
            print(f"  {i}. {m}")
    else:
        print("全部校验项通过。")
    print("\n提醒：predictions.csv 最终必须针对【教师指定图片】重新生成，"
          "且 image_id 与教师给的图片一一对应，不得多、漏、重复。")


if __name__ == "__main__":
    main()
