"""按任务书要求构建本组数据集：50 个车型 + 1 个 unknown。

用法：python build_dataset.py
输出：data/train/<51 类>、data/val/<51 类>、classes.txt、class_mapping.json
同盘符优先用硬链接（不额外占空间），失败自动退化为复制。脚本可重复运行，已存在的文件跳过。
"""
import json
import os
import random
import shutil

ROOT = r"D:\实训课\智能系统综合实训"
SRC_TRAIN = os.path.join(ROOT, "train")
SRC_VAL = os.path.join(ROOT, "val")
DATA = os.path.join(ROOT, "data")
DST_TRAIN = os.path.join(DATA, "train")
DST_VAL = os.path.join(DATA, "val")
INFO = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "cls_info", "class_info.json"), encoding="utf-8"))}

IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 50 个目标车型，按 ID 升序（classes.txt 的行顺序 == 模型索引 0~49）
SELECTED = [
    "0009", "0047", "0052", "0098", "0118", "0119", "0123", "0141", "0157", "0173",
    "0174", "0183", "0192", "0193", "0202", "0222", "0225", "0228", "0259", "0275",
    "0287", "0311", "0323", "0377", "0418", "0432", "0489", "0496", "0497", "0541",
    "0558", "0583", "0608", "0632", "0638", "0651", "0659", "0681", "0698", "0742",
    "0745", "0748", "0783", "0848", "0854", "0873", "0879", "0950", "0955", "0984",
]

# unknown 来源：除 50 个目标类以外的**全部车型**都作为 unknown 候选，每类限量抽样。
#
# 最初只用了 12 个来源类（120 张），结果测试集的组外车型来自 777 个从没见过的类别，
# 未知召回率只有 9.4%——模型只是记住了「这 12 种车不是我的菜」，完全没有泛化。
# 扩充到覆盖全部组外类别后，模型才能学到「不属于我 50 类的车都算 unknown」。
UNKNOWN_TRAIN_PER_CLASS = 30    # ~775 类 × 30 ≈ 23250 张，扩充组外样本多样性以降低组外车被误判成已知类
UNKNOWN_VAL_PER_CLASS = 2      # ~775 类 × 2  ≈ 1550 张

SEED = 20260910


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def images_of(split_dir, cid):
    d = os.path.join(split_dir, cid)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG))


def unknown_candidate_classes():
    """除 50 个目标类以外、train 和 val 都有数据的全部车型。"""
    usable = []
    for cid in sorted(os.listdir(SRC_TRAIN)):
        if cid in SELECTED or not os.path.isdir(os.path.join(SRC_TRAIN, cid)):
            continue
        if images_of(SRC_TRAIN, cid) and images_of(SRC_VAL, cid):
            usable.append(cid)
    return usable


def preflight():
    """选类防呆：检查所选类别在 train/ 和 val/ 中是否都有数据。

    教师给的 val/ 比 train/ 少 0001、0110 两个类，若选中它们就没有验证图，
    该类 F1 无法计算，会直接拉低 Macro-F1。
    """
    problems = []
    for cid in SELECTED:
        n_tr, n_va = len(images_of(SRC_TRAIN, cid)), len(images_of(SRC_VAL, cid))
        if n_tr == 0 or n_va == 0:
            problems.append(f"{cid} {INFO.get(cid, {}).get('name_from_new', '?')}：train={n_tr}, val={n_va}")
    if problems:
        raise SystemExit("选类检查未通过，以下类别缺少训练或验证数据：\n  " + "\n  ".join(problems))
    usable = unknown_candidate_classes()
    if len(usable) < 700:
        raise SystemExit(f"unknown 候选类只有 {len(usable)} 个（预期约 777），数据可能不完整")
    print(f"选类检查通过：50 个目标车型均有数据；unknown 候选类 {len(usable)} 个。")


def build():
    preflight()
    rng = random.Random(SEED)
    stat = {"link": 0, "copy": 0, "skip": 0}

    # 1) 50 个目标车型：整目录搬运
    for cid in SELECTED:
        for split, src_root, dst_root in (("train", SRC_TRAIN, DST_TRAIN), ("val", SRC_VAL, DST_VAL)):
            files = images_of(src_root, cid)
            out = os.path.join(dst_root, cid)
            os.makedirs(out, exist_ok=True)
            for f in files:
                stat[link_or_copy(os.path.join(src_root, cid, f), os.path.join(out, f))] += 1

    # 2) unknown：从所有组外车型抽样合并（覆盖全部类别，不是固定的 12 类）
    #    先清空旧的 unknown 目录，避免残留旧样本（旧方案只含 12 类）
    for dst_root in (DST_TRAIN, DST_VAL):
        unknown_dir = os.path.join(dst_root, "unknown")
        if os.path.isdir(unknown_dir):
            shutil.rmtree(unknown_dir)
    for cid in unknown_candidate_classes():
        for split, src_root, dst_root, cap in (
                ("train", SRC_TRAIN, DST_TRAIN, UNKNOWN_TRAIN_PER_CLASS),
                ("val", SRC_VAL, DST_VAL, UNKNOWN_VAL_PER_CLASS)):
            files = images_of(src_root, cid)
            picked = rng.sample(files, min(cap, len(files)))
            out = os.path.join(dst_root, "unknown")
            os.makedirs(out, exist_ok=True)
            for f in picked:
                # 文件名带类别 ID 前缀（如 0006_0004.jpg），跨类天然不冲突；
                # 万一撞名则加前缀兜底，绝不覆盖丢失
                dst_name = f if not os.path.exists(os.path.join(out, f)) else f"{cid}_{f}"
                stat[link_or_copy(os.path.join(src_root, cid, f), os.path.join(out, dst_name))] += 1

    # 3) classes.txt：恰好 50 行、四位 ID、升序、UTF-8 无 BOM
    assert len(SELECTED) == 50 and len(set(SELECTED)) == 50
    assert all(len(c) == 4 and c.isdigit() for c in SELECTED)
    assert SELECTED == sorted(SELECTED)
    with open(os.path.join(DATA, "classes.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(SELECTED) + "\n")

    # 4) 类别映射：索引 0~49 对应 classes.txt，索引 50 = unknown
    names = [INFO[c]["name_from_new"] for c in SELECTED] + ["unknown"]
    mapping = {"class_names": names,
               "class_ids": SELECTED + ["unknown"],
               "index_to_id": {str(i): (SELECTED + ["unknown"])[i] for i in range(51)}}
    with open(os.path.join(DATA, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print("文件操作：", stat)
    return names


def verify(names):
    print("\n%-6s %-30s %-8s %6s %6s" % ("ID", "车型名称", "索引", "train", "val"))
    t_tr = t_va = 0
    for i, cid in enumerate(SELECTED + ["unknown"]):
        n_tr = len([f for f in os.listdir(os.path.join(DST_TRAIN, cid)) if f.lower().endswith(IMG)])
        n_va = len([f for f in os.listdir(os.path.join(DST_VAL, cid)) if f.lower().endswith(IMG)])
        t_tr += n_tr
        t_va += n_va
        print("%-6s %-30s %-8s %6d %6d" % (cid, names[i], i, n_tr, n_va))
    print("\n合计 train %d 张 / val %d 张；类别数 %d" % (t_tr, t_va, len(names)))

    # 校验：unknown 样本不得与 50 个目标类的图片重名，train/val 之间也不得重叠
    def names_in(split_dir, cid):
        return set(os.listdir(os.path.join(split_dir, cid)))

    for cid in SELECTED:
        for split_dir in (DST_TRAIN, DST_VAL):
            assert not (names_in(split_dir, "unknown") & names_in(split_dir, cid)), cid
    assert not (names_in(DST_TRAIN, "unknown") & names_in(DST_VAL, "unknown"))
    print("校验通过：unknown 与 50 个目标类无重名，unknown 的 train/val 无重叠。")


if __name__ == "__main__":
    names = build()
    verify(names)
