"""推理脚本：单张图片推理 + 批量推理生成 predictions.csv。

单张推理返回的字典后续会直接作为智能体的视觉工具返回值（含类别、名称、置信度、Top-K）。
批量推理输出格式严格按任务书要求：UTF-8，表头 image_id,predicted_label,confidence。

用法：
    python src/predict.py --image path/to/car.jpg                       # 单张
    python src/predict.py --dir path/to/test_dir --out outputs/predictions.csv   # 批量
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch
from PIL import Image, ImageFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import IMG_EXTS, MODELS_DIR, OUTPUTS_DIR, get_device
from dataset import build_eval_transform
from model import load_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True
UNKNOWN_INDEX = 50
# 训练结束后最佳模型会被统一复制为 models/best.pt，作为对外唯一模型入口
DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "best.pt")


class CarClassifier:
    """把「模型 + 类别映射 + 预处理参数」封装成一个可复用的推理器。

    智能体的 classify_car 工具直接调用 classify()，无需重复加载模型。
    """

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None,
                 confidence_threshold: float = 0.0):
        self.device = torch.device(device) if device else get_device()
        self.net, self.ckpt = load_checkpoint(checkpoint, map_location="cpu")
        self.net.to(self.device).eval()
        self.transform = build_eval_transform({"input_size": self.ckpt["preprocess"]["input_size"]})
        self.labels = [self.ckpt["index_to_label"][str(i)] for i in range(self.ckpt["num_classes"])]
        self.names = self.ckpt["class_names"]
        self.checkpoint_path = checkpoint
        # 可选拒识阈值：最大概率低于它时判为 unknown。
        # 默认关闭 —— 验证集搜索显示阈值带来的 Macro-F1 提升在噪声范围内（见 outputs/threshold_sweep.csv），
        # 而任务书也强调低置信度处理不能替代 unknown 类别的训练。
        self.confidence_threshold = confidence_threshold

    def _reject(self, label: str, confidence: float) -> bool:
        return self.confidence_threshold > 0 and confidence < self.confidence_threshold

    @torch.no_grad()
    def classify(self, image_path: str, topk: int = 3) -> dict:
        """识别单张图片，返回统一结构的结果字典。"""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return {"ok": False, "error": f"图片读取失败：{os.path.basename(str(image_path))}（{exc.__class__.__name__}）"}

        x = self.transform(image).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.net(x).float(), dim=1)[0]
        conf, idx = prob.topk(min(topk, len(self.labels)))

        top = [{"label": self.labels[i], "name": self.names[i], "confidence": round(float(c), 4)}
               for c, i in zip(conf.cpu(), idx.cpu())]
        best = top[0]
        rejected = self._reject(best["label"], best["confidence"])
        return {
            "ok": True,
            "image": os.path.basename(image_path),
            "label": "unknown" if rejected else best["label"],
            "name": "unknown" if rejected else best["name"],
            "confidence": best["confidence"],
            "is_unknown": rejected or best["label"] == "unknown",
            "rejected_by_threshold": rejected,
            "topk": top,
            "model": {
                "arch": self.ckpt["arch"],
                "checkpoint": os.path.basename(self.checkpoint_path),
                "best_val_macro_f1": self.ckpt.get("best_val_macro_f1"),
                "confidence_threshold": self.confidence_threshold,
            },
        }

    @torch.no_grad()
    def classify_batch(self, image_paths: list[str], batch_size: int = 64, topk: int = 1) -> list[dict]:
        """批量识别，返回与 classify() 同结构的列表（见 predictions.csv 生成）。"""
        results = []
        buffers, names = [], []

        def flush():
            if not buffers:
                return
            x = torch.stack(buffers).to(self.device)
            prob = torch.softmax(self.net(x).float(), dim=1)
            conf, idx = prob.topk(min(topk, len(self.labels)), dim=1)
            for row in range(x.size(0)):
                top = [{"label": self.labels[i], "name": self.names[i], "confidence": round(float(c), 4)}
                       for c, i in zip(conf[row].cpu(), idx[row].cpu())]
                rejected = self._reject(top[0]["label"], top[0]["confidence"])
                results.append({"ok": True, "image": names[row],
                                "label": "unknown" if rejected else top[0]["label"],
                                "name": "unknown" if rejected else top[0]["name"],
                                "confidence": top[0]["confidence"],
                                "rejected_by_threshold": rejected,
                                "topk": top})
            buffers.clear()
            names.clear()

        for path in image_paths:
            try:
                buffers.append(self.transform(Image.open(path).convert("RGB")))
                names.append(os.path.basename(path))
            except Exception:
                results.append({"ok": False, "image": os.path.basename(path),
                                "error": "图片读取失败"})
                continue
            if len(buffers) >= batch_size:
                flush()
        flush()
        return results


def list_images(directory: str, recursive: bool = True) -> list[str]:
    """列出待推理图片。

    recursive=True 时会向下遍历子目录，兼容「教师给的测试图按类别文件夹组织」的情况；
    image_id 只取文件名（含扩展名），与任务书要求一致。
    """
    files = []
    if recursive:
        for root, _dirs, names in os.walk(directory):
            for f in names:
                if f.lower().endswith(IMG_EXTS):
                    files.append(os.path.join(root, f))
    else:
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(directory, f))]
    files.sort(key=lambda p: os.path.basename(p))
    if not files:
        raise SystemExit(f"{directory} 下没有找到图片文件")

    names = [os.path.basename(p) for p in files]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise SystemExit(f"存在同名图片，image_id 会重复，请先处理：{sorted(duplicates)[:5]}")
    return files


def write_predictions_csv(results: list[dict], out_path: str) -> dict:
    """按任务书格式写出 predictions.csv：每张图片恰好一条记录。"""
    written, failed = 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "predicted_label", "confidence"])
        for r in results:
            if not r.get("ok"):
                failed += 1
                continue
            w.writerow([r["image"], r["label"], f"{r['confidence']:.4f}"])
            written += 1
    return {"rows": written, "skipped": failed, "path": out_path}


def main() -> None:
    p = argparse.ArgumentParser(description="车型分类推理")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--image", default=None, help="单张图片路径")
    p.add_argument("--dir", default=None, help="批量推理的图片目录（默认递归子目录）")
    p.add_argument("--no-recursive", action="store_true", help="只扫描目录第一层")
    p.add_argument("--out", default=os.path.join(OUTPUTS_DIR, "predictions.csv"))
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--confidence-threshold", type=float, default=0.0,
                   help="低于该置信度判为 unknown；默认 0 表示不启用（阈值应由验证集确定）")
    a = p.parse_args()

    if not a.image and not a.dir:
        raise SystemExit("请至少指定 --image 或 --dir")

    clf = CarClassifier(a.checkpoint, a.device, a.confidence_threshold)

    if a.image:
        print(json.dumps(clf.classify(a.image, a.topk), ensure_ascii=False, indent=2))

    if a.dir:
        paths = list_images(a.dir, recursive=not a.no_recursive)
        results = clf.classify_batch(paths, a.batch_size, topk=1)
        stats = write_predictions_csv(results, a.out)
        unknown_n = sum(1 for r in results if r.get("ok") and r["label"] == "unknown")
        print(f"[批量推理] 目录 {a.dir}")
        print(f"  图片总数        {len(paths)}")
        print(f"  已写出记录      {stats['rows']}")
        print(f"  读取失败跳过    {stats['skipped']}")
        print(f"  判为 unknown    {unknown_n}")
        print(f"  输出文件        {stats['path']}")


if __name__ == "__main__":
    main()
