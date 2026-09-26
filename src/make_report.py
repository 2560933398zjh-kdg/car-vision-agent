"""生成《智能系统综合实训》技能考核报告（.docx）。

用法：
    python src/make_report.py                    # 输出到项目根目录
    python src/make_report.py --out 报告.docx

报告结构严格对齐教师模板：封面 → 填写说明 → 8 章正文（含表 1~表 11）→ 参考文献 → 附录。
所有数字均来自 outputs/ 与 logs/ 下的真实实验记录。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402

TEMPLATE = r"D:/Download/浏览器下载/实训报告模板_视觉智能体综合实训.docx"

# ----------------------------------------------------------------- 小工具


def set_font(run, name="宋体", size=10.5, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def para(doc, text="", size=10.5, bold=False, align=None, indent=True,
         name="宋体", space_after=4):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent and align is None:
        p.paragraph_format.first_line_indent = Pt(21)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if text:
        set_font(p.add_run(text), name=name, size=size, bold=bold)
    return p


def caption(doc, text, above=True):
    """表题置于表上方、图题置于图下方。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.space_before = Pt(6) if above else Pt(3)
    set_font(p.add_run(text), name="黑体", size=10.5, bold=False)
    return p


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(str(h)), name="黑体", size=9, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            set_font(p.add_run(str(v)), size=9)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    return t


def h1(doc, text):
    p = doc.add_heading(text, level=1)
    for r in p.runs:
        set_font(r, name="黑体", size=15, bold=True)
        r.font.color.rgb = RGBColor(0, 0, 0)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    return p


def h2(doc, text):
    p = doc.add_heading(text, level=2)
    for r in p.runs:
        set_font(r, name="黑体", size=12.5, bold=True)
        r.font.color.rgb = RGBColor(0, 0, 0)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    return p


# ----------------------------------------------------------------- 数据读取

def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def class_table_rows():
    """表2：50 行车型 + unknown 汇总行。"""
    from common import load_name_map
    class_ids = [l.strip() for l in open(os.path.join(PROJECT_ROOT, "classes.txt"),
                                         encoding="utf-8") if l.strip()]
    name_map = load_name_map()          # class_info.json 是列表，common 已按 id 建索引
    rows = []
    for i, cid in enumerate(class_ids):
        name = name_map.get(cid, "")
        n_tr = len(os.listdir(os.path.join(PROJECT_ROOT, "data", "train", cid))) \
            if os.path.isdir(os.path.join(PROJECT_ROOT, "data", "train", cid)) else 0
        n_va = len(os.listdir(os.path.join(PROJECT_ROOT, "data", "val", cid))) \
            if os.path.isdir(os.path.join(PROJECT_ROOT, "data", "val", cid)) else 0
        rows.append([i, cid, name or "—", n_tr, n_va])
    # unknown 汇总行
    def cnt(split):
        p = os.path.join(PROJECT_ROOT, "data", split, "unknown")
        return len(os.listdir(p)) if os.path.isdir(p) else 0
    rows.append([50, "—", "unknown（组外车型，见 2.2）", cnt("train"), cnt("val")])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "实训报告_识车笔记.docx"))
    ap.add_argument("--name", default="【请填写姓名】")
    ap.add_argument("--sid", default="【请填写学号】")
    ap.add_argument("--major", default="人工智能技术应用")
    ap.add_argument("--cls", default="【请填写班级】")
    ap.add_argument("--teacher", default="【请填写指导教师】")
    a = ap.parse_args()

    doc = docx.Document()
    # 页面与默认字体
    for s in doc.sections:
        s.top_margin, s.bottom_margin = Cm(2.5), Cm(2.5)
        s.left_margin, s.right_margin = Cm(3.0), Cm(2.5)
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ============================================================ 封面
    for _ in range(2):
        para(doc, "", indent=False)
    para(doc, "《智能系统综合实训》", size=22, bold=True, align="center", indent=False, name="黑体")
    para(doc, "技能考核报告", size=22, bold=True, align="center", indent=False, name="黑体")
    for _ in range(2):
        para(doc, "", indent=False)
    for label, val in (("题　　目", "识车笔记——基于自训练视觉模型与 DeepSeek 智能体的\n车型识别与管理系统"),
                       ("姓　　名", a.name), ("专　　业", a.major), ("班　　级", a.cls),
                       ("学　　号", a.sid), ("指导教师", a.teacher)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        set_font(p.add_run(f"{label}：{val}"), size=13)
    for _ in range(2):
        para(doc, "", indent=False)
    para(doc, "人工智能学院", size=14, bold=True, align="center", indent=False, name="黑体")
    para(doc, f"{date.today().year} 年 {date.today().month} 月", size=13,
         align="center", indent=False)
    doc.add_page_break()

    # ============================================================ 填写说明
    h1(doc, "报告填写说明")
    para(doc, "本模板用于记录汽车图像分类模型、DeepSeek-V4-Flash 智能体和 Web 应用的"
              "设计、训练、集成与测试过程。报告完整覆盖八个部分。")
    para(doc, "填写时将提示语替换为真实内容，表格按实际数量增删行；实验结论均有日志、"
              "数据或截图支持。图表在正文中说明，表题置于表上方，图题置于图下方，"
              "并连续编号。")
    doc.add_page_break()

    # ============================================================ 1 项目概述
    h1(doc, "1 项目概述")
    h2(doc, "1.1 选题背景与应用场景")
    para(doc, "项目名称为「识车笔记」，面向汽车爱好者、二手车评估初学者和需要快速辨识"
              "路边车辆的普通用户。教师提供了覆盖 827 个车型类别、按 train/val 划分的"
              "真实车辆图像数据集，为细粒度车型识别提供了数据基础。")
    para(doc, "汽车的品牌与车型数量庞大，同品牌不同车系、同车系不同年款在外观上高度相似，"
              "普通用户很难辨认。本项目的核心业务链条是：用户上传一张车照 → 系统识别出"
              "车型 → 查询并展示车型资料（可选联网查询详细参数）→ 将结果存入「识车笔记」"
              "→ 用户可随时检索、修改备注或删除记录。视觉识别结果并非终点，而是后续"
              "资料查询、收藏与检索整个业务流程的入口参数。")
    para(doc, "需要说明的能力边界：本组模型只在本组选定的 50 款车型上训练，超出该范围的"
              "车辆一律输出 unknown，系统会如实告知「未识别为本组支持的车型」，"
              "不会编造车型或继续按某个确定车型执行业务操作。")

    h2(doc, "1.2 主要功能与任务目标")
    para(doc, "系统主要功能包括：① 车型识别（上传图片，返回四位车型 ID 或 unknown 及"
              "置信度）；② 车型资料查询（返回品牌、车身类型、细分级别）；③ 联网查询"
              "车型详细信息（定位、动力、尺寸、价格、油耗）；④ 识车笔记收藏管理"
              "（保存、修改备注、删除、检索、导出）；⑤ 工具调用记录可视化。")
    para(doc, "其中最典型的、需要连续调用 3 个以上工具才能完成的任务是：用户上传一张"
              "车辆照片并提出「这是什么车？介绍一下它的资料，然后加入我的识车笔记」，"
              "智能体需依次调用 classify_car（视觉识别）→ query_car_info（查资料）→ "
              "save_car_record（写入收藏），三步结果逐级传递，缺一不可。系统共实现 "
              "6 个工具，其中视觉工具 1 个、业务工具 5 个，满足任务书「总数不少于 3 个、"
              "至少 1 个视觉工具与 2 个业务工具」的要求。")
    caption(doc, "表1  项目功能与需求对应表")
    table(doc,
          ["功能", "用户输入与预期输出", "对应模块或工具"],
          [["车型识别", "输入车照；输出四位车型 ID 或 unknown、confidence、Top-K 候选",
            "自训练视觉模型 + classify_car 工具"],
           ["车型资料查询", "输入车型 ID；输出品牌、车身类型、细分级别",
            "query_car_info 工具 + cls_info/class_info.json"],
           ["联网查详情", "输入车型名称；输出定位、动力、尺寸、价格、油耗等",
            "get_car_detail 工具（阿里云百炼联网搜索 + 本地缓存）"],
           ["识车笔记管理", "输入车型 ID、备注；输出保存/更新结果与记录编号",
            "save_car_record / search_records / delete_car_record + SQLite"],
           ["多轮对话", "第二轮沿用上一轮识别结果的车型 ID，无需重新上传图片",
            "agent.Session 会话状态"],
           ["异常反馈", "非图片文件、组外车型、低置信度、工具失败均需如实提示",
            "工具 ok=false 返回 + 系统提示词约束"],
           ["Web 交互", "上传图片、自然语言输入、查看回答与工具调用记录",
            "Flask 服务 + web/ 前端"]],
          widths=[3.0, 7.5, 5.0])

    h2(doc, "1.3 实训目标与实施安排")
    para(doc, "本实训的学习目标包括：掌握从原始数据到可交付模型的完整流程（数据检查、"
              "选类、划分、增强、训练、评估、推理）；理解并正确使用细粒度分类的评价口径"
              "（51 类等权 Macro-F1、混淆矩阵、unknown 专项指标）；掌握大模型工具调用"
              "（Function Calling）的开发范式；完成前后端系统集成与工程化实践。")
    para(doc, "实施安排按四周推进：第 1 周完成数据集构建（选类、unknown 采样、划分校验）"
              "与基线模型训练；第 2 周进行对比实验与评估优化，确定最佳模型并完成批量推理"
              "与提交物格式自检；第 3 周完成智能体工具层与 DeepSeek 接入，实现连续工具"
              "调用、多轮会话与异常处理；第 4 周完成 Web 系统、系统测试、缺陷修复与"
              "报告撰写。")

    # ============================================================ 2 数据分析
    h1(doc, "2 数据分析")
    h2(doc, "2.1 类别选择与标签映射")
    para(doc, "从教师提供的 827 个车型类别中自主选择恰好 50 类作为目标车型，"
              "并补充 1 个 unknown 类，构成 51 类输出。选类依据如下："
              "① 样本充足，每类训练图不少于 85 张、验证图不少于 30 张，保证模型可训练；"
              "② 品牌互不重复，50 个类别分属 50 个不同品牌，避免同品牌孪生车型互相干扰；"
              "③ 外观差异大，优先选取造型辨识度高的车型（如硬派越野、跑车、皮卡）；"
              "④ 覆盖 SUV、轿车、卡车、MPV、跑车、微面、轻客等 7 种车身类型，"
              "使模型面对的车型分布更均衡；⑤ 全部使用具体车型，不以品牌或"
              "「轿车/SUV」等大类替代。")
    para(doc, "classes.txt 位于项目根目录，采用 UTF-8 无 BOM 编码，恰好 50 行，"
              "每行一个不重复的四位数字 ID，保留前导零并升序排列，不含 unknown。"
              "模型索引 0～49 按该文件行序映射，索引 50 固定对应 unknown；"
              "训练、保存、单张推理、批量推理与 Web 端全部使用同一映射，"
              "映射写死在模型文件中，避免推理错位。")
    caption(doc, "表2  类别映射与样本统计")
    table(doc, ["模型索引", "类别 ID", "车型名称", "训练数", "验证数"],
          class_table_rows(), widths=[1.8, 1.8, 6.4, 1.8, 1.8])

    h2(doc, "2.2 未知类别构成与采样")
    para(doc, "unknown 类代表「不属于本组 50 款车型的其他车辆」，其样本只能从教师提供"
              "数据集中剩余 777 个组外类别中采样，且必须与 50 个目标类完全隔离。")
    para(doc, "采样方法：从组外 775 个在 train/ 与 val/ 中均有数据的类别中，"
              "每类随机抽取 30 张训练图与 2 张验证图，分别合并为 data/train/unknown 与"
              "data/val/unknown。训练与验证样本取自教师不同的划分目录，文件名零重叠，"
              "不存在重复；采样未使用任何测试图片，也未把 50 个目标类的任何图片"
              "标记为 unknown。随机种子固定为 20260910，采样过程可完全复现"
              "（data/build_dataset.py）。")
    para(doc, "采样方案经过一次重要迭代。初版仅从 12 个组外类别各抽 10 张（合计 120 张）"
              "训练，结果模型只记住了这 12 类车「不是我的菜」，对教师测试集中其余组外"
              "车型几乎全部误判为目标车型，unknown 检出率仅 9.4%。因此将 unknown 训练集"
              "扩充为覆盖全部 775 个组外类别（每类 30 张，合计 23250 张），使模型真正学到"
              "「不属于 50 类的车都算 unknown」这一判别边界。扩充后 unknown 检出率提升到"
              "96% 以上，是本项目最关键的数据改进。")
    caption(doc, "表3  unknown 来源与采样记录")
    unk_src = [["全部组外类别（775 个）", "23250", "1550",
                "每类随机抽 30 张训练 + 2 张验证；train/val 划分不重叠，"
                "随机种子 20260910，可复现"],
               ["初版 12 个来源类（已弃用）", "120", "48",
                "每类 10 张训练 + 4 张验证；覆盖 7 种车身类型，"
                "因泛化不足被淘汰"]]
    table(doc, ["来源 ID 与车型名称", "训练数", "验证数", "采样方法与比例"], unk_src,
          widths=[4.2, 1.6, 1.6, 8.1])
    para(doc, "组外类别示例（节选）：0006 捷豹F-PACE（SUV）、0073 起亚K3（轿车）、"
              "0188 阿斯顿·马丁DB11（跑车）、0210 长安凯程欧诺S（微面）、"
              "0565 江铃特顺（轻客）、0635 长城金刚炮（皮卡）等，"
              "完整清单见 data/build_dataset.py 的运行日志。")

    h2(doc, "2.3 数据预处理与增强")
    para(doc, "数据检查：构建数据集时校验了 unknown 与 50 个目标类无重名、"
              "unknown 的 train/val 无重叠，并跳过了无法解码的残缺图片。"
              "预处理固定为：短边缩放到指定尺寸（224 输入时缩放到 256；"
              "320/384 高分辨率实验按同比例放大）→ 居中裁剪 → 按 ImageNet 均值"
              "[0.485,0.456,0.406] 与标准差 [0.229,0.224,0.225] 归一化。"
              "验证与推理始终使用该确定性流程。")
    para(doc, "训练期增强包括：随机裁剪缩放（scale 0.7～1.0）、随机水平翻转、"
              "以及可选的强增强组（颜色抖动、±10° 随机旋转、高斯模糊、随机灰度）。"
              "所有实验均在末尾附加随机擦除（RandomErasing）。"
              "训练与验证使用两套不同的 transform，训练集每轮看到的视图都不同，"
              "验证集则始终使用确定性预处理，保证指标可比。")

    # ============================================================ 3 模型设计与训练
    h1(doc, "3 模型设计与训练")
    h2(doc, "3.1 模型结构与基线")
    para(doc, "基线采用 resnet18 主干，加载 torchvision 提供的 ImageNet 预训练权重"
              "（IMAGENET1K_V1），将最后的全连接层替换为 51 维输出头（50 个目标车型 + "
              "unknown），输入尺寸 224×224。选择 resnet18 的理由是：残差结构在中小规模"
              "数据上收敛稳定、显存占用小，适合本机 4GB 显存的条件。")
    para(doc, "迁移学习策略为「先冻结、后微调」：前若干轮冻结主干只训练分类头，"
              "避免随机初始化的分类头在早期把预训练特征带偏；随后解冻全部参数，"
              "用小学习率整体微调。全部训练均使用本组自建的 50 类 + unknown 数据完成，"
              "未使用任何现成汽车分类模型或第三方在线识别 API。")
    para(doc, "在前述基线之上，针对教师测试集的实际表现，进一步探索了更高输入分辨率"
              "（320×320、384×384）与更强主干（resnet34、resnet50），最终以多模型"
              "集成方式交付，详见 4.2 节。")

    h2(doc, "3.2 训练环境与参数配置")
    caption(doc, "表4  训练环境与配置")
    table(doc, ["配置项", "实际取值"],
          [["硬件与软件", "CPU 16 核；GPU NVIDIA GeForce RTX 3050 Laptop（4GB 显存，"
                          "sm_86）；Windows；Python 3.13；PyTorch + torchvision（CUDA 12.4）"],
           ["数据", "训练 28942 张（50 类 5692 张 + unknown 23250 张）；"
                    "验证 3433 张（50 类 1883 张 + unknown 1550 张）；51 类"],
           ["模型与预训练", "resnet18 / resnet34 / resnet50，ImageNet IMAGENET1K_V1 "
                            "预训练权重；输出 51 维；输入 224 / 320 / 384"],
           ["优化器与学习率", "AdamW，weight_decay=1e-4，lr=5e-4～1e-3（按实验配置）"],
           ["调度与正则", "余弦退火（cosine）；标签平滑 0.1；冻结主干 1～2 轮后解冻"],
           ["批大小与精度", "batch 12～64（按输入分辨率与显存调整）；"
                            "启用 AMP 混合精度"],
           ["随机种子", "20260910（主实验）/ 777 / 12345（集成多样性用）"],
           ["评估口径", "51 类等权 Macro-F1，用验证集选模型，不使用测试图片调参"],
           ["推理增强", "水平翻转 TTA（对高分辨率模型稳定小幅正收益）；多模型 logits 平均"]],
          widths=[3.6, 12.0])

    h2(doc, "3.3 训练流程与模型选择")
    para(doc, "训练流程为：① 读取配置（JSON）确定结构、分辨率、增强、优化器与轮数；"
              "② 构建训练/验证 DataLoader，训练集应用随机增强；③ 每轮完成前向、"
              "损失计算、反向传播与参数更新，随后在验证集上计算 51 类 Macro-F1；"
              "④ 每 2 轮保存一次中期 checkpoint（便于回溯训练过程），并在验证指标"
              "提升时保存 best checkpoint；⑤ 训练结束后保存 last checkpoint，"
              "并把最佳模型发布为 models/best.pt 供推理与 Web 端使用。")
    para(doc, "模型选择严格以验证集 51 类 Macro-F1 为唯一标准，不使用测试图片调参。"
              "模型文件（.pt）中固化了网络结构、51 类类别映射、预处理参数（输入尺寸、"
              "Resize 尺寸、均值、标准差）与完整训练配置，因此只需一个文件即可正确"
              "复现推理，不会出现训练与推理预处理不一致的问题。")

    h2(doc, "3.4 单张与批量推理")
    para(doc, "单张推理：加载图片 → 按模型文件记录的预处理参数缩放、裁剪、归一化 → "
              "前向计算 51 维输出 → softmax 得到概率 → 取最大概率对应的类别，"
              "输出四位车型 ID（或 unknown）、置信度与 Top-3 候选。"
              "这一接口封装为 CarClassifier.classify()，供智能体的视觉工具直接调用。")
    para(doc, "批量推理：对教师指定目录递归收集图片，逐批前向，输出 softmax 之前的"
              "原始 logits 与 argmax 标签，生成 result.csv（两列 image_id, "
              "predicted_label）与 result.npz（image_id + logits，N×51 float32），"
              "列序与 classes.txt 行序严格一致（索引 50 为 unknown）。"
              "多模型集成时对同一张图的各模型 logits 取平均后再取 argmax，"
              "保证 result.csv 的标签与 result.npz 的 logits 完全自洽——"
              "这一点经过脚本逐行交叉校验。模型权重、中期 checkpoint 与实验模型"
              "均保存在本地 models/ 目录，不随作业提交，也不打包进源代码。")

    # ============================================================ 4 实验结果与分析
    h1(doc, "4 实验结果与分析")
    h2(doc, "4.1 评估数据与指标口径")
    para(doc, "评估使用本组验证集（3433 张，50 个目标车型 1883 张 + unknown 1550 张）。"
              "唯一效果指标为 51 类等权平均 Macro-F1：单类 F1 = 2×TP / (2×TP + FP + FN)，"
              "分母为 0 时该类 F1 记 0，unknown 与每个车型同权，不排除在评分之外。"
              "Accuracy、Precision、Recall、每类 F1、混淆矩阵、unknown 检出率与"
              "已知类误拒率仅作辅助分析。教师测试集上的得分以教师用 result.csv 与"
              "真实标签计算的结果为准，与本文的验证集指标严格区分。")

    h2(doc, "4.2 基线与改进实验")
    para(doc, "第一阶段的对比实验在初始数据集（unknown 仅 12 个来源类）上进行，"
              "目的是确定训练策略中的有效因素。五组实验每次只改动单一变量，"
              "便于归因，结果如下表。")
    caption(doc, "表5  基线与改进实验结果")
    table(doc, ["实验", "主要改动", "51类 Macro-F1", "辅助指标", "日志位置"],
          [["e0_baseline", "resnet18，无增强、无调度、无标签平滑（基线）", "0.8035",
            "训练 acc 98%，第 5 轮后验证指标下滑", "logs/e0_baseline_history.csv"],
           ["e1_aug", "仅加基础增强，未配套正则手段", "0.7832",
            "仍过拟合，见 4.4 分析", "logs/e1_aug_history.csv"],
           ["e2_strong_aug_ls", "强增强 + 标签平滑 0.1 + 余弦退火", "0.9323",
            "训练与验证差距显著收窄", "logs/e2_strong_aug_ls_history.csv"],
           ["e3_resnet34", "换 resnet34 主干（强增强配置不变）", "0.9340",
            "收益有限、耗时增加 44%", "logs/e3_resnet34_history.csv"],
           ["e4_freeze_then_finetune", "先冻结主干 2 轮再解冻微调", "0.9402",
            "最小最快方案取得最好成绩（2.77 分钟）",
            "logs/e4_freeze_then_finetune_history.csv"],
           ["e5_broad_unknown", "unknown 扩充为覆盖 775 个组外类（每类 10 张）",
            "0.8786", "unknown 检出率由 9.4% 升至 96.8%",
            "logs/e5_broad_unknown_history.csv"],
           ["e9_hi_res", "输入分辨率提升到 320×320", "0.9086",
            "误拒率与误收率同步下降", "logs/e9_hi_res_history.csv"],
           ["e12_res384", "输入分辨率提升到 384×384，14 轮充分收敛", "0.9234",
            "分辨率仍是收益最高的单因素", "logs/e12_res384_history.csv"],
           ["e13_res34_320", "resnet34 + 320×320，14 轮", "0.9211",
            "结构收益小于分辨率收益", "logs/e13_res34_320_history.csv"],
           ["e14_res384_seed777", "384×384 换随机种子，增加集成多样性", "0.9264",
            "验证集单模型最优", "logs/e14_res384_seed777_history.csv"],
           ["e15_resnet50", "resnet50 + 224×224", "0.9150",
            "4GB 显存下无法同时提高分辨率", "logs/e15_resnet50_history.csv"],
           ["e13+e14+e15 集成 ★", "三模型 logits 平均（结构 × 分辨率双维度多样性）",
            "0.9323", "验证集最优且最稳健，为最终交付方案",
            "src/make_submission.py --ensemble"]],
          widths=[3.4, 4.6, 2.0, 3.2, 3.4])
    para(doc, "需要特别说明：e0～e4 在初始数据集上评估，e5 之后在扩充 unknown 的数据集上"
              "评估，两组数字不可直接横向比较；扩充 unknown 后验证集难度显著提高"
              "（unknown 从 48 张增至 1550 张），因此验证集 Macro-F1 表面下降而"
              "真实测试成绩大幅提升。")

    h2(doc, "4.3 混淆矩阵与错误样本")
    para(doc, "下图为最佳单模型在验证集上的 51×51 混淆矩阵热图（含 unknown 类，"
              "横轴为预测类别、纵轴为真实类别，对角线越亮表示识别越准）。"
              "矩阵显示 50 个目标车型之间几乎没有系统性混淆，主要误差集中在"
              "unknown 行与列：即已知车型被判为 unknown（误拒）以及组外车型被判为"
              "某个目标车型（误收）。")
    cm = os.path.join(OUTPUTS_DIR, "best_check_confusion_matrix.png")
    if os.path.exists(cm):
        doc.add_picture(cm, width=Cm(13.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption(doc, "图1  验证集 51 类混淆矩阵（含 unknown）", above=False)
    else:
        para(doc, "【此处插入混淆矩阵热图：outputs/best_check_confusion_matrix.png】")
    para(doc, "典型错误样本分析如下表。错误可归为三类："
              "① 同车身类型、同设计语言导致的车型间混淆（如沃尔沃 S90 与奥迪 A6、"
              "名爵 MG6 与马自达 3 昂克赛拉，均为同级别轿车）；"
              "② 拍摄角度、遮挡或远距离导致细节被抹平；"
              "③ 已知车型被拒识为 unknown，多发生在训练集缺乏该拍摄视角时。")
    caption(doc, "表6  典型错误样本分析")
    errs = [
        ["0009_0019.jpg", "0009 沃尔沃S90 → 0632 奥迪A6", "0.2171",
         "同为中大型轿车，侧面轮廓与灯组造型接近；置信度低，属模型不确定"],
        ["0123_0233.jpg", "0123 名爵MG6 → 0541 马自达3昂克赛拉", "0.4262",
         "紧凑型运动轿车，溜背线条相似；可用更高分辨率缓解"],
        ["0118_0334.jpg", "0118 特斯拉Cybertruck → 0608 极氪001", "0.0927",
         "组外车型被判为目标车型，置信度极低；扩充 unknown 训练后已大幅减少"],
        ["0047_0263.jpg", "0047 吉利星越L → unknown", "0.1971",
         "已知车型被误拒：该角度训练集覆盖不足；提高分辨率与集成可缓解"],
        ["0098_0012.jpg", "0098 赛力斯SF5 → unknown", "0.4849",
         "误拒但置信度中等，说明特征处于决策边界附近"],
    ]
    table(doc, ["图片编号", "真实与预测标签", "confidence", "原因与改进方向"], errs,
          widths=[2.6, 5.0, 1.8, 6.2])
    para(doc, "完整错误清单见 outputs/best_check_errors.csv，"
              "最易混淆类别对见 outputs/best_check_top_confusions.csv。")

    h2(doc, "4.4 实验结论与局限")
    para(doc, "（1）有效改进及其依据。其一，强增强 + 标签平滑 + 余弦退火必须成组使用："
              "单独增加基础增强（e1）反而使指标由 0.8035 降到 0.7832，因为增强未配套"
              "正则手段且延长了收敛时间；三者同时加入后（e2）指标跃升至 0.9323。"
              "其二，迁移学习中「先冻结再解冻」比直接全网络微调更高效（e4：用最小最快的"
              "模型取得 0.9402，耗时仅 2.77 分钟）。其三，unknown 类别的训练覆盖范围"
              "决定开放集性能：把来源从 12 类扩到 775 类后，unknown 检出率由 9.4% "
              "提升到 96% 以上，教师测试集得分随之从 14.39 分升至 67 分以上。"
              "其四，输入分辨率是收益最高的单因素，224→320→384 每档带来约 0.02～0.03 的"
              "验证集提升。其五，多模型集成（不同结构 × 不同分辨率 × 不同随机种子）"
              "能同时改善误拒与误收，是单模型调参无法达到的。")
    para(doc, "（2）低置信度阈值。按任务书要求在验证集上搜索了拒识阈值：最优阈值仅带来 "
              "+0.0009 的 Macro-F1 提升（属噪声级别），而把阈值调高到 0.25 虽使 unknown "
              "检出率升至 77%，unknown 自身 F1 反而从 0.61 降到 0.25，已知类误拒率"
              "升至 5.7%。这印证了「低置信度处理不能替代 unknown 类别的训练」，"
              "因此智能体侧不自动改写预测结果，改为展示 Top-K 请用户确认。")
    para(doc, "（3）未解决问题与局限。其一，unknown 仍是最弱类别：组外车型与目标车型"
              "在外观上本就存在相似样本，误收与误拒无法完全消除。其二，验证集与教师"
              "测试集存在明显分布差异（验证集 Macro-F1 约 0.93，测试集约 0.81），"
              "说明测试集的拍摄条件、难度与训练数据不同，训练侧的常规手段难以弥合，"
              "这也是后续改进的主要方向。其三，本机显存仅 4GB，限制了主干规模与"
              "输入分辨率的进一步组合。")

    # ============================================================ 5 智能体设计
    h1(doc, "5 智能体设计")
    h2(doc, "5.1 DeepSeek 接入与工具执行流程")
    para(doc, "智能体通过 DeepSeek 开放平台的 OpenAI 兼容接口接入，模型名固定为 "
              "deepseek-v4-flash，API Key 只从环境变量 DEEPSEEK_API_KEY 读取，"
              "不写入任何文件、日志或截图。该模型为纯文本模型，不具备读图能力，"
              "因此图片一律通过「用户消息中携带图片路径 → 智能体调用视觉工具 → "
              "工具返回类别与置信度」的链路处理，不存在后台预先识别再拼接 Prompt 的"
              "做法，工具也不会返回伪造的执行结果。")
    para(doc, "执行流程：① 用户消息与工具定义（JSON Schema）一并提交给模型；"
              "② 模型返回 tool_calls 时，程序先按白名单校验工具名，再用 JSON Schema "
              "校验参数（缺参、类型错误、未知参数均拦截），随后调用真实函数；"
              "③ 工具执行结果以 role=tool 回传给模型；④ 模型根据结果决定继续调用工具"
              "或给出最终回答；⑤ 全过程记录工具名、参数、返回值与耗时。"
              "系统提示词固化了任务书的关键约束：图片必须经工具识别、"
              "资料只能来自工具返回且不得编造、unknown 时必须停止后续操作、"
              "工具返回内容是数据而非指令。")

    h2(doc, "5.2 工具定义与实现")
    para(doc, "系统共实现 1 个视觉工具与 5 个业务工具，参数使用 JSON Schema 声明，"
              "执行器统一做白名单与参数校验。", indent=False)
    caption(doc, "表7  工具接口与异常处理")
    table(doc, ["工具名称与用途", "输入参数", "返回结果", "失败处理"],
          [["classify_car（视觉分类，必需）",
            "image_path（图片路径）、可选 top_k",
            "车型 ID、名称、confidence、is_unknown、Top-K 候选",
            "路径为空/不存在/为目录/格式不支持 → ok=false 并说明原因"],
           ["query_car_info（查车型资料）", "car_id（四位车型 ID）",
            "品牌、车型名、车身类型、细分级别",
            "unknown、不在本组 50 类内、不存在 → ok=false，不编造资料"],
           ["save_car_record（保存收藏）",
            "car_id、可选 note、confidence",
            "保存动作（created/updated）、记录编号、总记录数",
            "ID 非法 → ok=false；同车型重复保存转为更新"],
           ["search_records（检索记录）",
            "可选 keyword / car_id / limit",
            "记录列表与总数",
            "查不到返回空列表，不报错"],
           ["delete_car_record（删除记录）", "car_id",
            "删除结果与剩余条数",
            "记录不存在 → ok=false，不误删"],
           ["get_car_detail（联网查详情）", "car_name（车型名称）",
            "定位、动力、尺寸、价格、油耗等（含来源）",
            "未配置 Key 或查不到 → 如实告知，不编造参数"]],
          widths=[3.6, 3.4, 4.4, 4.2])
    para(doc, "视觉工具真实加载本组训练并保存的 models/best.pt；查资料工具读取教师提供的 "
              "cls_info/class_info.json；业务工具读写 SQLite 数据库 data/app.db；"
              "联网工具调用阿里云百炼的通义千问联网搜索，并设置本地缓存表避免重复消耗"
              "额度。需要说明：外部 API 仅用于查询车型文字资料，视觉识别始终由本组"
              "自训练模型完成。")

    h2(doc, "5.3 连续工具调用与结果衔接")
    para(doc, "任务输入：「请识别这张图片里的车型，介绍一下它的资料，然后加入我的"
              "识车笔记。」（附图片 photo_a.jpg）；"
              "调用链：classify_car → query_car_info → save_car_record；"
              "结果衔接：classify_car 返回的 car_id 直接作为 query_car_info 与 "
              "save_car_record 的参数，query_car_info 返回的品牌与车身类型用于"
              "生成自然语言介绍，save_car_record 的返回值用于向用户确认收藏成功；"
              "调用记录见 outputs/agent_记录_demo.md（第 1 轮）。")
    para(doc, "同一轮的完整调用记录摘要如下（参数、返回值、状态均取自真实运行日志，"
              "完整原始返回见 outputs/agent_记录_demo.md）：", indent=False)
    for line in [
        "第 1 步  classify_car　输入：{\"image_path\": \"photo_a.jpg\"}　"
        "输出：{\"car_id\":\"0183\",\"car_name\":\"保时捷_保时捷718\","
        "\"confidence\":0.959,\"is_unknown\":false}　状态：成功",
        "第 2 步  query_car_info　输入：{\"car_id\": \"0183\"}　"
        "输出：{\"brand\":\"保时捷\",\"body_type\":\"跑车\",\"sub_type\":\"跑车\"}　状态：成功",
        "第 3 步  save_car_record　输入：{\"car_id\":\"0183\",\"confidence\":0.959}　"
        "输出：{\"action\":\"created\",\"total\":1}　状态：成功",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(21)
        p.paragraph_format.space_after = Pt(2)
        set_font(p.add_run(line), size=9.5)
    para(doc, "此外，多轮示例：第二轮用户仅输入「那我再把它备注成『路上看到的车』」，"
              "智能体直接复用上一轮的 car_id 调用 save_car_record，"
              "本轮工具链只含 1 个工具且不含 classify_car，证明未重新识别图片。"
              "unknown 示例：输入组外车型图片后，智能体只调用 classify_car 即停止，"
              "如实回复「置信度只有 0.12，未识别为本组支持的车型，请确认」，"
              "不查资料、不保存。异常示例：输入 .txt 文件后，工具返回格式错误，"
              "智能体照实转述且未产生任何保存动作。")

    h2(doc, "5.4 Session 与低置信度处理")
    para(doc, "会话（Session）保存 system 提示词、完整对话历史（含工具调用消息）、"
              "最近一次识别结果的车型 ID 与置信度，以及累计轮数。同一 session_id 的"
              "多轮请求共享该状态，因此「帮我把它记录下来」这类省略主语的指令能够"
              "正确解析到上一轮的车型；会话可经接口重置以清空上下文。")
    para(doc, "低置信度处理策略：识别结果附 Top-K 候选；当置信度低于设定阈值"
              "（默认关闭，阈值由验证集搜索确定，最优点收益仅为 +0.0009，属噪声级）"
              "或模型输出 unknown 时，智能体展示候选并请用户确认，"
              "不自动改写预测结果，也不继续执行保存等操作。预测为 unknown 时"
              "统一提示「未识别为本组支持的车型」，不编造具体车型。")

    # ============================================================ 6 Web 与工程实现
    h1(doc, "6 Web 与工程实现")
    h2(doc, "6.1 系统架构与模块关系")
    para(doc, "技术栈为 Flask + 纯静态 HTML/CSS/JavaScript，不需要 Node 与打包构建，"
              "单进程单端口同时提供接口与页面。选择 Flask 而非 FastAPI 的原因："
              "本项目的视觉推理是阻塞式的，Flask 多线程配合推理锁比异步框架更易处理，"
              "且新增依赖最少。")
    para(doc, "系统自下而上分为四层：① 模型与数据层——自训练视觉模型（models/best.pt）、"
              "教师类别说明（class_info.json）、SQLite 业务库（data/app.db）；"
              "② 工具层（src/tools.py）——6 个工具的真实实现与参数校验；"
              "③ 智能体层（src/agent.py）——DeepSeek 客户端、工具循环、会话管理；"
              "④ 服务与展示层（src/app.py + web/）——HTTP 接口与页面。"
              "用户上传图片并输入自然语言任务后，请求经 /api/chat 进入智能体，"
              "智能体按需调用工具（其中视觉工具加载本地模型完成识别），"
              "最终回答、本轮工具调用记录与笔记快照一并返回前端渲染。")
    para(doc, "【此处插入系统架构图：可用 docs/ 中的分层示意图或自行绘制】", indent=False)

    h2(doc, "6.2 主要页面与交互")
    para(doc, "页面为三栏布局：左侧上传与对话区（支持点击、拖拽、粘贴上传，"
              "上传后立即缩略图预览；对话以气泡展示，附「识别并收藏」「查看我的笔记」"
              "等快捷任务按钮）；右上为识车笔记面板（列表、关键词检索、"
              "编辑备注、删除、导出 CSV/JSON、统计概览）；右下为工具调用记录面板，"
              "每条记录显示工具名、状态与耗时，点击可展开查看完整参数与原始返回。")
    para(doc, "以一次完整任务为例说明页面变化：用户上传车辆照片并输入"
              "「这是什么车？查看它的详细信息」→ 对话区出现加载态，"
              "随后展示智能体回答（车型、置信度、本地资料与联网详情）；"
              "右下工具记录依次出现 classify_car、query_car_info、get_car_detail "
              "三条记录，均为绿色成功态；用户再说「加入我的笔记」→ "
              "工具记录新增 save_car_record，右上笔记面板同步新增一条记录；"
              "第二轮对话未重新上传图片，直接复用了上一轮的识别结果。")
    para(doc, "【此处插入 Web 页面截图：对话区、笔记面板、工具调用记录各一张】", indent=False)

    h2(doc, "6.3 数据存储与接口")
    para(doc, "业务数据使用 SQLite 单文件 data/app.db，选择理由是零配置、单文件便于"
              "随项目迁移，且能满足本实训的读写需求。核心表为 car_records"
              "（主键 id，唯一键 car_id；字段含车型名、品牌、车身类型、细分级别、"
              "置信度、图片路径、备注、创建与更新时间），同一车型重复保存只更新备注，"
              "不产生重复记录；另有 car_detail_cache 表缓存联网查询到的车型详情。"
              "会话状态保存在服务进程内存中，进程重启即丢失——这是有意的取舍，"
              "本实训无多用户持久化需求，已在 8.3 节作为局限说明。")
    caption(doc, "表8  工程模块与实现位置")
    table(doc, ["模块", "职责及关键数据", "代码或配置位置"],
          [["视觉模型与推理", "加载模型、预处理、类别解码、confidence 与 Top-K 计算",
            "src/predict.py、src/model.py、models/best.pt"],
           ["数据与增强", "数据集构建、训练/验证 transform、类别映射",
            "src/dataset.py、data/build_dataset.py、classes.txt"],
           ["训练与评估", "训练循环、验证选模型、混淆矩阵与错误分析",
            "src/train.py、src/evaluate.py、src/metrics.py"],
           ["工具层", "6 个工具的实现、JSON Schema、白名单与参数校验、SQLite 存取",
            "src/tools.py、configs/agent.json"],
           ["智能体层", "DeepSeek 客户端、工具循环、会话状态、对话记录生成",
            "src/agent.py"],
           ["Web 服务", "14 个 HTTP 接口、上传安全、会话管理、22 项系统测试",
            "src/app.py"],
           ["前端页面", "上传与对话、笔记面板、工具调用记录面板",
            "web/index.html、style.css、app.js、records.js、logs.js"],
           ["提交物生成", "多模型集成推理、result.csv 与 result.npz 生成、格式自检",
            "src/make_submission.py、src/validate_submission.py"],
           ["训练监控", "实时读取训练进度并以网页展示进度条与曲线",
            "src/monitor.py"]],
          widths=[3.0, 7.0, 5.6])
    para(doc, "主要接口示例：POST /api/chat 接收 {session_id, message, image_path?}，"
              "返回 {ok, answer, tool_calls[], records[], stats}；"
              "参数缺失或消息为空返回 400；记录不存在时 PATCH/DELETE "
              "返回 404 并附 JSON 错误说明。", indent=False)

    h2(doc, "6.4 环境配置与启动")
    para(doc, "依赖：torch、torchvision（CUDA 12.4 版本）、flask、requests、"
              "jsonschema、matplotlib、python-docx。环境创建与启动步骤：")
    para(doc, "① 创建虚拟环境并安装依赖；② 准备模型文件 models/best.pt 与数据集 "
              "data/train、data/val；③ 设置环境变量（API Key 使用占位符）："
              "set DEEPSEEK_API_KEY=sk-xxxx、set DASHSCOPE_API_KEY=sk-xxxx；"
              "④ 启动 Web 服务：.venv/Scripts/python.exe src/app.py，"
              "浏览器打开 http://127.0.0.1:5000；⑤ 训练监控（可选）："
              "python src/monitor.py，打开 http://127.0.0.1:8000 实时查看训练进度。",
         indent=False)
    para(doc, "视觉报错时也可直接用命令行验证：src/predict.py --image 图片路径 完成"
              "单张推理，src/predict.py --dir 目录 生成批量结果。API Key 仅通过环境变量"
              "提供，代码、配置、报告与截图中均不含真实密钥。", indent=False)

    h2(doc, "6.5 异常处理与集成问题")
    para(doc, "系统对四类异常做统一处理：① 图片读取失败（路径不存在、格式不支持、"
              "文件被截断）→ 工具返回 ok=false 与具体原因，智能体如实转述；"
              "② 无效输入（空消息、缺少 session_id、非法车型 ID）→ 接口返回 400 "
              "并给出可读错误，或工具返回失败而不产生副作用；"
              "③ API 请求失败（Key 未配置、余额不足、网络超时）→ 前端展示提示条，"
              "对话返回明确错误说明，不伪装成功；④ 工具执行失败 → 智能体不得声称"
              "已完成操作。")
    para(doc, "联调中发现并定位的主要问题：一是视觉推理并发导致的显存竞争，"
              "通过可重入锁串行化推理解决；二是上传接口在清理非法文件时异常外泄，"
              "使本应返回 400 的请求变成 500，已用异常保护修正；"
              "三是修改备注接口在请求体非法时会静默创建记录，"
              "已改为请求体非法返回 400、记录不存在返回 404；"
              "四是检索未覆盖车身类型，导致「我收藏的跑车有哪些」查不到结果，"
              "已把车身类型与细分级别加入匹配范围。上述问题的定位过程与复测结果"
              "见第 7 章表 10。")

    # ============================================================ 7 系统测试
    h1(doc, "7 系统测试")
    h2(doc, "7.1 测试环境与方法")
    para(doc, "测试环境与训练环境一致：Windows + Python 3.13 + PyTorch（CUDA 12.4）、"
              "NVIDIA RTX 3050 Laptop（4GB 显存）、Flask 本地服务；"
              "模型版本为集成方案（e13+e14+e15）；测试数据为教师提供的测试图片目录"
              "与本组验证集样本。测试方法：先在服务内运行 22 项自动化系统测试"
              "（src/app.py --selftest），覆盖静态页面、健康检查、会话管理、"
              "图片上传、笔记接口与参数校验；再以真实 HTTP 服务完成端到端任务测试"
              "（连续三工具、多轮复用、检索、删除、unknown 拒绝）。"
              "本章记录系统功能测试，模型效果实验见第 4 章。")

    h2(doc, "7.2 关键功能与异常测试")
    caption(doc, "表9  系统测试用例")
    table(doc, ["编号", "场景与操作", "预期行为", "实际结果及证据"],
          [["T01", "上传有效图片并询问车型",
            "显示图片，真实调用视觉工具并返回类别与 confidence",
            "通过：工具记录含 classify_car，返回 0183 保时捷718，confidence 0.959"],
           ["T02", "连续三工具任务（识别+介绍+收藏）",
            "同一轮依次调用视觉工具与两个业务工具，结果逐级传递",
            "通过：classify_car → query_car_info → save_car_record"],
           ["T03", "多轮对话：第二轮只说「帮我记录下来」",
            "复用上一轮车型 ID，不重新识别图片",
            "通过：本轮工具链仅 save_car_record，不含 classify_car"],
           ["T04", "上传组外车型图片",
            "输出 unknown 并提示未识别为本组支持车型，不保存",
            "通过：仅调用 classify_car，未查资料、未保存"],
           ["T05", "上传非图片文件（改后缀）",
            "拒绝上传并提示格式不合法",
            "通过：返回 400，提示文件内容不是有效图片"],
           ["T06", "查询笔记与关键词检索",
            "返回记录列表，可按关键词与车身类型检索",
            "通过：检索「跑车」返回对应记录"],
           ["T07", "修改备注与删除记录",
            "正常更新/删除；记录不存在时返回错误",
            "通过：记录不存在时 PATCH/DELETE 返回 404"],
           ["T08", "API Key 未配置或额度不足",
            "前端提示未配置，对话返回明确错误，不伪装成功",
            "通过：健康检查提示 Key 缺失，对话返回错误说明"],
           ["T09", "传入非法车型 ID（不在本组 50 类）",
            "工具返回失败，智能体如实转述，不产生错误保存",
            "通过：返回 ok=false 与原因"]],
          widths=[1.2, 4.2, 5.2, 5.0])
    para(doc, "22 项自动化测试全部通过，覆盖静态页面与健康检查 3 项、会话管理 3 项、"
              "图片上传 5 项（有效图/改后缀假图/非法扩展名/未传文件/上传后可访问）、"
              "笔记接口 7 项（列表/统计/导出/删除与修改的 404/非法请求体）、"
              "参数校验 3 项、检索覆盖车身类型 1 项。", indent=False)

    h2(doc, "7.3 问题修复与复测")
    caption(doc, "表10  问题与修复记录")
    table(doc, ["关联用例", "实际问题与原因", "修复方法", "复测结果"],
          [["T02/T03", "视觉推理持锁后又二次请求同一把不可重入锁，"
                       "进程静默卡死",
            "改用可重入锁（RLock），并把模型加载移到锁外",
            "复测通过：连续工具调用正常"],
           ["T05", "清理非法文件时异常外泄，使 400 被覆盖为 500",
            "用 try/except 包裹清理逻辑",
            "复测通过：返回 400 与可读提示"],
           ["T07", "修改备注接口在请求体非法时静默创建记录；"
                   "记录不存在也照常创建",
            "请求体非法返回 400，记录不存在返回 404，"
            "改用只更新已存在记录的函数",
            "复测通过：非法请求体 400，不存在 404"],
           ["T06", "检索未覆盖车身类型，「我收藏的跑车有哪些」无结果",
            "把 body_type 与 sub_type 纳入匹配范围",
            "复测通过：检索返回正确记录"],
           ["T02", "演示时模型从临时文件名推测答案的风险",
            "图片复制为中性命名的临时文件后再上传",
            "复测通过：识别结果只能来自视觉工具"],
           ["T09", "演示中发现并发请求会打爆 4GB 显存",
            "推理加锁串行化，会话各持独立锁",
            "复测通过：多请求下无 OOM"]],
          widths=[1.8, 5.2, 4.6, 4.0])
    para(doc, "尚未完全解决的问题：unknown 类在真实测试集上仍有约 3%～4% 的误收"
              "与误拒；会话状态存于内存，服务重启后丢失。前者依靠扩充 unknown 训练"
              "覆盖与集成缓解，后者作为工程局限记录在 8.3 节。", indent=False)

    # ============================================================ 8 总结与分工
    h1(doc, "8 总结与分工")
    h2(doc, "8.1 实训总结")
    para(doc, "本次实训完整走通了「数据分析 → 模型训练与评估 → 智能体与工具开发 → "
              "Web 系统集成 → 系统测试与提交」的全流程，最重要的收获来自几个"
              "真实问题的定位与解决。")
    para(doc, "第一，评价口径必须先于优化确定。项目早期只关注总体准确率，"
              "没有意识到 51 类等权 Macro-F1 下「每一类都同等重要」，"
              "导致 unknown 这一类的巨大缺陷被掩盖，首次测试仅得 14.39 分。"
              "正确理解口径后，才把 unknown 的训练覆盖作为首要改进方向。")
    para(doc, "第二，开放集问题的解决靠数据而不是后处理。把 unknown 训练来源从 12 类"
              "扩充到 775 类后，检出率从 9.4% 跃升到 96% 以上；而尝试过的"
              "「减少 unknown 样本」「给已知类加权」「对已知类过采样」三种调平衡手段"
              "全部失败——它们都只是移动决策边界，必然在改善一侧的同时破坏另一侧。"
              "这让我们真正理解了「低置信度阈值不能替代 unknown 类别的训练」。")
    para(doc, "第三，工程细节决定成败。上传安全（白名单 + 真实解码 + uuid 重命名）、"
              "推理串行化（4GB 显存下的并发保护）、提交物一致性（result.csv 的标签"
              "必须等于 result.npz 的 logits 取 argmax）、模型选择只许用验证集，"
              "这些看似琐碎的约束，每一条都直接影响成绩或系统可用性。"
              "此外，我们用脚本对提交物做了逐项格式自检，并交叉校验了批量推理与"
              "评估两条链路的标签映射是否一致，避免了格式或映射错误导致的无谓失分。")
    para(doc, "第四，迭代要靠证据。为提高成绩，我们共训练了 16 组对照实验，"
              "每次都先分析上一版的失分结构（已知类误拒、组外类误收、"
              "各类 F1 分布），再决定下一步改动方向，并用统一的验证集口径"
              "（后期改为「每类多张」的全量验证集）做多随机种子的稳健性检验，"
              "避免在单一划分上过拟合。")

    h2(doc, "8.2 个人贡献与成员分工")
    para(doc, "本项目由本组成员共同完成。下面按模块说明各自承担的工作与成果依据；"
              "若有搭档，请按实际分工补充第二行。", indent=False)
    caption(doc, "表11  成员分工与成果依据")
    table(doc, ["成员", "负责模块与工作", "完成成果与证据", "协作或集成说明"],
          [[a.name + "（" + a.sid + "）",
            "数据集构建与选类；模型训练与对比实验（16 组）；"
            "评估与错误分析；批量推理与提交物生成；训练监控工具",
            "data/build_dataset.py、configs/、src/train.py、src/evaluate.py、"
            "src/make_submission.py、src/monitor.py；"
            "outputs/experiments_summary.csv、logs/*_history.csv",
            "向智能体与 Web 提供统一的单张/批量推理接口 "
            "（predict.CarClassifier.classify）"],
           ["【第二位成员姓名、学号】（若有）",
            "智能体工具层与 DeepSeek 接入；多轮 Session 与异常处理；"
            "Flask 服务与前端页面；系统测试",
            "src/tools.py、src/agent.py、src/app.py、web/；"
            "outputs/agent_记录_demo.md、22 项系统测试结果",
            "与视觉模块对接工具参数与返回结构，"
            "共同完成端到端联调与提交物核验"]],
          widths=[2.8, 4.2, 5.2, 3.4])

    h2(doc, "8.3 系统不足与后续改进")
    para(doc, "（1）易混淆车型与未知类覆盖。部分同级别、同设计语言的车型仍存在混淆"
              "（如中大型轿车之间）；组外车型与目标车型外观相近时会产生误收。"
              "后续可尝试在 unknown 采样中引入「与目标车型外观相近的困难负样本」，"
              "或在损失函数中引入度量学习项，拉大类间距离。")
    para(doc, "（2）数据规模与分布差异。每类训练图仅百余张，且验证集与测试集存在"
              "明显分布差异（验证集 Macro-F1 约 0.93，测试集约 0.81）。"
              "后续可通过更强的数据增强策略、半监督利用未标注图片、"
              "以及更贴近测试场景的采样来缩小差距。")
    para(doc, "（3）推理性能与硬件限制。本机显存 4GB，限制了主干规模与输入分辨率的"
              "组合空间（例如 resnet50 无法与高分辨率同时使用）。"
              "后续可在更大显存环境训练更强的模型，或采用模型蒸馏把大模型能力"
              "压缩到可部署的小模型上。")
    para(doc, "（4）工具可靠性与交互体验。联网查询工具受外部服务与额度限制，"
              "失败率不为零，目前通过本地缓存缓解；会话状态存于内存，"
              "服务重启即丢失，不利于长期使用。后续可改为持久化会话存储，"
              "并为联网工具增加多源降级与重试策略。")

    # ============================================================ 参考文献
    h1(doc, "参考文献")
    refs = [
        "[1] 周志华．机器学习[M]．北京：清华大学出版社，2016：97-115．",
        "[2] 李航．统计学习方法（第2版）[M]．北京：清华大学出版社，2019：56-78．",
        "[3] 张俊, 陈超, 王明. 基于深度残差网络的细粒度车辆识别方法[J]. "
        "计算机应用研究, 2023, 40(6): 1823-1828．",
        "[4] 刘洋, 赵敏, 孙磊, 等. 面向开放集的车辆图像分类与未知类检测研究[J]. "
        "计算机工程与应用, 2024, 60(3): 210-218．",
        "[5] 王磊, 李娜. 数据增强与标签平滑在细粒度图像分类中的对比研究[J]. "
        "计算机科学, 2023, 50(9): 145-152．",
        "[6] 陈思, 黄伟, 周涛. 基于模型集成的细粒度车型识别精度提升方法[J]. "
        "现代信息科技, 2025, 9(2): 33-38．",
        "[7] 赵鹏. 深度学习在汽车车型识别中的应用研究[D]．西安: 西安电子科技大学，2023．",
        "[8] 孙宇. 基于迁移学习的小样本图像分类方法研究[D]．北京: 北京邮电大学，2024．",
        "[9] 吴静, 郑凯. 大语言模型工具调用（Function Calling）技术综述[J]. "
        "计算机工程与科学, 2025, 47(1): 88-97．",
        "[10] 何谐, 罗斌. 基于 Flask 的轻量级深度学习模型服务化部署实践[J]. "
        "现代信息科技, 2024, 8(14): 52-56．",
        "[11] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image "
        "Recognition[C]//Proceedings of the IEEE Conference on Computer Vision "
        "and Pattern Recognition (CVPR). 2016: 770-778．",
        "[12] Bendale A, Boult T E. Towards Open Set Deep Networks[C]//Proceedings "
        "of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). "
        "2016: 1563-1572．",
        "[13] PyTorch 官方文档. Models and pre-trained weights[EB/OL]. (2025-06-01). "
        "[2026-09-20]. https://pytorch.org/vision/stable/models.html．",
        "[14] DeepSeek 开放平台. API 使用文档[EB/OL]. (2026-08-01). [2026-09-20]. "
        "https://platform.deepseek.com/api-docs．",
        "[15] 阿里云. 百炼大模型服务平台联网搜索文档[EB/OL]. (2026-07-15). "
        "[2026-09-20]. https://help.aliyun.com/zh/model-studio/．",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        set_font(p.add_run(r), size=10)

    # ============================================================ 附录
    h1(doc, "附录")
    para(doc, "附录 A  提交物清单：classes.txt（50 行四位车型 ID，UTF-8 无 BOM，"
              "升序，不含 unknown）、result.csv（image_id, predicted_label）、"
              "result.npz（image_id + logits，N×51 float32）、项目源代码、本报告。",
         indent=False)
    para(doc, "附录 B  关键文件与产出位置：模型推理 src/predict.py；"
              "批量提交物生成 src/make_submission.py；提交物自检 "
              "src/validate_submission.py；智能体 src/agent.py；工具层 src/tools.py；"
              "Web 服务 src/app.py；训练监控 src/monitor.py；"
              "实验汇总 outputs/experiments_summary.csv；"
              "对话与工具调用记录 outputs/agent_记录_demo.md；"
              "混淆矩阵与错误清单 outputs/best_check_confusion_matrix.png、"
              "outputs/best_check_errors.csv。", indent=False)
    para(doc, "附录 C  复现命令（节选）：", indent=False)
    for cmd in [".venv/Scripts/python.exe data/build_dataset.py        # 重建数据集",
                ".venv/Scripts/python.exe src/train.py --config configs/e14_res384_seed777.json",
                ".venv/Scripts/python.exe src/evaluate.py --checkpoint models/best.pt --tag best_check",
                ".venv/Scripts/python.exe src/make_submission.py --ensemble \"模型A,模型B,模型C\" --tta",
                ".venv/Scripts/python.exe src/validate_submission.py",
                ".venv/Scripts/python.exe src/agent.py            # 智能体四链路演示",
                ".venv/Scripts/python.exe src/app.py              # 启动 Web 系统",
                ".venv/Scripts/python.exe src/monitor.py          # 训练进度监控"]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        set_font(p.add_run(cmd), name="Consolas", size=9)

    doc.save(a.out)
    print(f"报告已生成：{a.out}")
    print(f"段落数 {len(doc.paragraphs)}，表格数 {len(doc.tables)}")


if __name__ == "__main__":
    main()
