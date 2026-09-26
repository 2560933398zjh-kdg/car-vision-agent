"""公共设施：路径、类别映射、预处理参数、随机种子、设备。

类别映射是全局唯一的：索引 0~49 对应 classes.txt 的行顺序，索引 50 固定为 unknown。
训练、验证、单张推理、批量推理、Web 端全部读同一份映射，避免出现标签错位。
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CLASSES_TXT = os.path.join(PROJECT_ROOT, "classes.txt")
CLASS_INFO_JSON = os.path.join(PROJECT_ROOT, "cls_info", "class_info.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
UNKNOWN = "unknown"
NUM_CLASSES = 51

# 预处理参数集中在此处，并随模型一起保存，保证推理与训练完全一致
INPUT_SIZE = 224
RESIZE_SIZE = 256
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def load_class_ids() -> list[str]:
    """读取 classes.txt：必须是 50 个四位 ID、升序、无重复。"""
    with open(CLASSES_TXT, encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != 50:
        raise ValueError(f"classes.txt 必须恰好 50 行，当前 {len(ids)} 行")
    if len(set(ids)) != 50:
        raise ValueError("classes.txt 存在重复 ID")
    for cid in ids:
        if len(cid) != 4 or not cid.isdigit():
            raise ValueError(f"非法类别 ID：{cid!r}，必须是四位数字")
    if ids != sorted(ids):
        raise ValueError("classes.txt 必须按类别 ID 升序排列")
    return ids


def load_name_map() -> dict[str, str]:
    """类别 ID -> 中文车型名称（取自 class_info.json 的 name_from_new）。"""
    with open(CLASS_INFO_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["id"]: r["name_from_new"] for r in rows}


def build_mapping() -> tuple[list[str], list[str]]:
    """返回 (标签列表, 展示名列表)，长度均为 51，最后一位是 unknown。"""
    ids = load_class_ids() + [UNKNOWN]
    names = load_name_map()
    display = [names.get(cid, cid) for cid in ids[:-1]] + [UNKNOWN]
    return ids, display


def label_to_index() -> dict[str, int]:
    ids, _ = build_mapping()
    return {lab: i for i, lab in enumerate(ids)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)


def describe_device() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
    return "cpu"


# ---------------------------------------------------------------- 智能体配置

AGENT_CONFIG_JSON = os.path.join(CONFIGS_DIR, "agent.json")

AGENT_DEFAULTS = {
    "model": "deepseek-v4-flash",
    "api_url": "https://api.deepseek.com/chat/completions",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0.2,
    "max_tokens": 2048,
    "max_rounds": 8,
    "max_tool_calls": 12,
    "request_timeout": [10, 90],
    "vision": {"checkpoint": "models/best.pt", "top_k": 3, "confidence_threshold": 0.0},
    "storage": {"db_path": "data/app.db"},
}


def load_agent_config(path: str | None = None) -> dict:
    """读取智能体配置。配置里的相对路径统一按项目根目录解析。"""
    cfg = json.loads(json.dumps(AGENT_DEFAULTS))  # 深拷贝，避免调用方改到默认值
    if os.path.isfile(path or AGENT_CONFIG_JSON):
        with open(path or AGENT_CONFIG_JSON, encoding="utf-8") as f:
            user_cfg = json.load(f)
        for key, value in user_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    for key in ("vision", "storage"):
        for field, val in cfg[key].items():
            if isinstance(val, str) and ("/" in val or "\\" in val) and not os.path.isabs(val):
                cfg[key][field] = os.path.join(PROJECT_ROOT, val)
    return cfg

