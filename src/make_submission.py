"""生成 EduBench 提交物：result.csv 与 result.npz。

任务书里的 predictions.csv（含 confidence）是报告配套格式；
EduBench 实际上传的是三件套：classes.txt（项目根目录，已存在）、result.csv、result.npz。
本脚本只生成后两个，规则严格对齐 student_submit/README.md：

    result.csv   两列 image_id,predicted_label，image_id 为文件名去掉扩展名
    result.npz   键 image_id(Unicode 字符串) + logits(float32, N×51)
                 logits 为 softmax 之前的原始分类分数，列序 = classes.txt 行序，第 51 列(索引50)=unknown

用法：
    python src/make_submission.py                          # 默认 test_images -> 项目根目录
    python src/make_submission.py --test-dir 某目录 --out-dir 输出目录
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import IMG_EXTS, PROJECT_ROOT, load_class_ids
from dataset import build_eval_transform
from model import load_checkpoint


class FlatImageDataset(Dataset):
    """测试图片是扁平目录（000001.jpg …），与训练用的按类别子目录不同，单独建一个轻量 Dataset。"""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def main() -> None:
    p = argparse.ArgumentParser(description="生成 result.csv 与 result.npz")
    p.add_argument("--test-dir", default=os.path.join(PROJECT_ROOT, "test_images"))
    p.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "models", "best.pt"))
    p.add_argument("--ensemble", default=None,
                   help="逗号分隔的多个 checkpoint 路径，对 logits 取平均（模型集成，通常优于单模型）")
    p.add_argument("--tta", action="store_true",
                   help="水平翻转 TTA：原图与翻转图 logits 取平均，几乎零成本的稳定提升")
    p.add_argument("--out-dir", default=PROJECT_ROOT)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_paths = ([x.strip() for x in a.ensemble.split(",") if x.strip()]
                  if a.ensemble else [a.checkpoint])

    # 1) 收集测试图片，按编号排序（000001 … 000827）
    files = sorted(
        [os.path.join(a.test_dir, f) for f in os.listdir(a.test_dir)
         if f.lower().endswith(IMG_EXTS)],
        key=lambda path: int(os.path.splitext(os.path.basename(path))[0]),
    )
    if not files:
        raise SystemExit(f"{a.test_dir} 下没有找到图片")
    image_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

    # 2) 逐模型前向取原始 logits（softmax 前），多模型时累加后平均
    #    每个模型用各自 checkpoint 里记录的预处理参数（支持不同分辨率混合集成）
    total = None
    for cp in ckpt_paths:
        net, ckpt = load_checkpoint(cp, map_location="cpu")
        net = net.eval().to(device)
        pre = ckpt["preprocess"]
        transform = build_eval_transform({"input_size": pre["input_size"],
                                          "resize_size": pre.get("resize_size")})
        loader = DataLoader(FlatImageDataset(files, transform),
                            batch_size=a.batch_size, shuffle=False, num_workers=0)
        chunks = []
        with torch.no_grad():
            for images in loader:
                x = images.to(device)
                if a.tta:
                    # 水平翻转 TTA：车头朝向不影响车型类别，两个视图的 logits 平均
                    chunks.append((net(x).float() + net(torch.flip(x, dims=[3])).float())
                                  .cpu().numpy() / 2)
                else:
                    chunks.append(net(x).float().cpu().numpy())
        one = np.concatenate(chunks, axis=0).astype("float32")
        total = one if total is None else total + one
        # 及时释放显存，避免多模型集成时显存累积导致 OOM
        del net
        if device == "cuda":
            torch.cuda.empty_cache()
    logits = (total / len(ckpt_paths)).astype("float32")
    if len(ckpt_paths) > 1:
        print(f"[集成] {len(ckpt_paths)} 个模型 logits 平均："
              f"{', '.join(os.path.basename(c) for c in ckpt_paths)}")

    # 3) 一致性校验
    assert logits.shape == (len(files), ckpt["num_classes"]), \
        f"logits 形状 {logits.shape} 与预期 ({len(files)}, {ckpt['num_classes']}) 不符"
    assert np.isfinite(logits).all(), "logits 出现 NaN 或 inf，禁止提交"

    # 4) 预测标签：argmax 后按模型内部映射还原成四位 ID 或 unknown
    mapping = ckpt["index_to_label"]                     # '0'..'49' -> classes.txt 行序, '50' -> unknown
    labels = [mapping[str(i)] for i in logits.argmax(axis=1)]

    # 5) 写 result.csv（两列，无 BOM、无 confidence 列）
    csv_path = os.path.join(a.out_dir, "result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_label"])
        for image_id, label in zip(image_ids, labels):
            writer.writerow([image_id, label])

    # 6) 写 result.npz（只有 image_id 和 logits 两个键）
    npz_path = os.path.join(a.out_dir, "result.npz")
    np.savez(npz_path,
             image_id=np.array(image_ids, dtype="<U6"),
             logits=logits)

    # 7) 打印汇总
    n_unknown = sum(1 for lab in labels if lab == "unknown")
    print(f"[提交物生成] 测试图片 {len(files)} 张")
    print(f"  预测为 unknown 的图片：{n_unknown} 张")
    print(f"  result.csv  -> {os.path.relpath(csv_path, PROJECT_ROOT)}（{len(files)}+1 行）")
    print(f"  result.npz  -> {os.path.relpath(npz_path, PROJECT_ROOT)}"
          f"（logits {logits.shape}, {logits.dtype}）")
    print(f"  classes.txt -> {os.path.relpath(os.path.join(PROJECT_ROOT, 'classes.txt'), PROJECT_ROOT)}"
          f"（{len(load_class_ids())} 类，已存在无需改动）")
    print("\n请自检三件事：")
    print("  1) result.csv 数据行数 = 测试图片数，且表头为 image_id,predicted_label")
    print("  2) result.npz 只有 image_id、logits 两个键，logits 为 (N,51) float32 且无 NaN")
    print("  3) classes.txt 仍是 50 行四位 ID、升序、无 unknown")


if __name__ == "__main__":
    main()
