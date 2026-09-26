# car-vision-agent · 识车笔记视觉智能体

基于 **PyTorch 图像分类 + DeepSeek 大模型工具调用 + Flask Web** 的汽车识别智能助手：上传一张车照，自动识别车型、查询车型资料、保存到「识车笔记」并支持多轮对话管理。

---

## 项目组成

本项目包含三个相互衔接的模块：

| 模块 | 说明 | 对外接口 |
| :--- | :--- | :--- |
| **视觉模型** | PyTorch 训练 51 类车型分类器（50 款车型 + unknown），最佳验证 Macro-F1 **0.9402** | `predict.CarClassifier.classify()` |
| **智能体** | DeepSeek-V4-Flash 驱动，6 个工具调用，多轮会话，拒绝编造 | `src/agent.py` |
| **Web 系统** | Flask + 纯静态前端，图片上传、聊天、笔记管理 | `src/app.py`（http://127.0.0.1:5000） |

---

## 技术栈

- **视觉模型**：Python 3.10+ / PyTorch / torchvision（ResNet18/34/50、EfficientNet-B0）、RecursiveCharacter 不适用——图像预处理用 ImageNet 均值方差
- **智能体**：DeepSeek-V4-Flash（OpenAI 兼容接口）、工具调用循环（max 8 轮 / 12 次工具调用）、JSON Schema 参数校验
- **数据存储**：SQLite 单文件（识车笔记 + 车型详情缓存）
- **Web**：Flask 多线程、HTML / CSS / 原生 JavaScript（无需 Node 构建）
- **联网查询**：阿里云百炼 DashScope（通义千问 `qwen-plus` 联网搜索，可选）

---

## 目录结构

```
car-vision-agent/
├── classes.txt                  # 50 个四位车型 ID（升序，不含 unknown）
├── cls_info/class_info.json     # 教师提供的车型说明（品牌/车身类型/细分级别）
├── configs/                     # 实验配置与智能体配置
│   ├── agent.json               #   DeepSeek 模型、工具上限、视觉 checkpoint、DB 路径
│   ├── e0_baseline.json ~ e16_*.json   # 17 组对比实验配置
├── data/
│   ├── build_dataset.py         # 选类 + unknown 采样脚本（可重跑）
│   ├── class_mapping.json       # 索引 0~49 → 车型，索引 50 → unknown
│   └── train/ val/             # 数据集（不随仓库提交）
├── src/
│   ├── common.py                # 路径、类别映射、预处理参数、随机种子
│   ├── dataset.py               # 数据集与数据增强
│   ├── model.py                 # 建模 + 模型文件保存/加载
│   ├── metrics.py               # Macro-F1 / 混淆矩阵 / unknown 专项指标
│   ├── train.py                 # 训练主程序
│   ├── evaluate.py              # 验证集评估与错误分析
│   ├── predict.py               # 单张 + 批量推理
│   ├── run_experiments.py       # 批量对比实验（支持断点续跑）
│   ├── tune_threshold.py        # 置信度拒识阈值搜索
│   ├── tools.py                 # 6 个智能体工具 + SQLite 存取
│   ├── agent.py                 # DeepSeek 智能体：工具循环 + 多轮 Session
│   └── app.py                   # Flask Web 服务 + 22 项系统自测
├── web/                         # 纯静态前端（index.html / style.css / app.js / records.js / logs.js）
├── models/                      # 训练权重（不提交，本地保存）
├── outputs/ logs/              # 运行产物（不提交）
└── .env.example                 # 所需环境变量示例
```

---

## 快速开始

### 1. 环境准备

```bash
cd car-vision-agent
python -m venv .venv

# GPU（CUDA 12.4，适配 RTX 3050）
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install flask requests jsonschema matplotlib
```

> 没有 GPU 时去掉 `--index-url` 装 CPU 版即可，脚本自动回退 CPU 训练。

### 2. 配置 API Key（只走环境变量，不写入任何文件）

```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-你的key"      # 智能体主模型（必需）
$env:DASHSCOPE_API_KEY = "你的key"         # 通义联网查详情（可选）
```

也可参考 `.env.example`。

### 3. 训练视觉模型

```bash
PY=.venv/Scripts/python.exe

# 冒烟测试（2 轮 + 每类 20 张，约 1 分钟）
$PY src/train.py --config configs/e0_baseline.json --epochs 2 --limit-per-class 20

# 训练最佳配置（冻结主干 2 轮后解冻微调）
$PY src/train.py --config configs/e4_freeze_then_finetune.json

# 跑全部对比实验（断点续跑）
$PY src/run_experiments.py
```

### 4. 运行智能体（命令行）

```bash
$PY src/tools.py                       # 工具层离线自测（不调大模型）
$PY src/agent.py                       # 内置演示：连续三工具 + 多轮复用 + unknown 拒绝
$PY src/agent.py --chat                # 交互式多轮对话
$PY src/agent.py --task "这是什么车？介绍一下，然后加入我的收藏" --image 图片路径
```

### 5. 启动 Web 系统

```bash
$PY src/app.py                         # 打开 http://127.0.0.1:5000
$PY src/app.py --selftest              # 只跑 22 项系统测试，不启动服务
```

---

## 实验结果（验证集 51 类等权 Macro-F1）

| 实验 | 结构 | 增强 | 调度 | 标签平滑 | 冻结轮数 | Macro-F1 | 耗时 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| e0_baseline | resnet18 | none | none | 0.0 | 0 | 0.8035 | 2.79 分 |
| e1_aug | resnet18 | basic | none | 0.0 | 0 | 0.7832 | 7.89 分 |
| e2_strong_aug_ls | resnet18 | strong | cosine | 0.1 | 0 | 0.9323 | 3.00 分 |
| e3_resnet34 | resnet34 | strong | cosine | 0.1 | 0 | 0.9340 | 4.32 分 |
| **e4_freeze_then_finetune** ★ | resnet18 | basic | cosine | 0.1 | 2 | **0.9402** | 2.77 分 |

- 基线 e0 严重过拟合（训练准确率 98% 后验证指标持续下滑）
- 强数据增强 + 标签平滑 + 余弦退火是最大提升组合（+0.129）
- e4 用最小最快的模型取得最好成绩：先冻结主干 2 轮只训分类头、再解冻微调

---

## 智能体工具一览

| 工具 | 类型 | 输入 | 返回 |
| :--- | :--- | :--- | :--- |
| `classify_car` | 视觉分类 | image_path、可选 top_k | 车型 ID、名称、confidence、is_unknown、Top-K |
| `query_car_info` | 业务 | car_id | 品牌、车身类型、细分级别（不编造） |
| `save_car_record` | 业务 | car_id、可选 note | 保存动作（created/updated）、记录数 |
| `search_records` | 业务 | 可选 keyword/car_id/limit | 记录列表 + 总数 |
| `delete_car_record` | 业务 | car_id | 删除结果 |
| `get_car_detail` | 联网 | car_name | 定位/动力/尺寸/价格（通义联网，结果缓存） |

### 关键设计

- **图片必须经工具识别**：模型是纯文本，用户消息只带图片路径，识别只能通过 `classify_car` 触发；演示图片复制为中性临时文件名，杜绝从文件名猜答案
- **多轮上下文复用**：Session 保留完整消息历史，第二轮"帮我记录一下"直接复用上一轮车型 ID，不重复识别图片
- **unknown 拒识**：低置信度时只展示 Top-K 请用户确认，不查资料、不保存，不编造
- **工具结果是数据不是指令**：系统提示词明确约束，避免工具返回内容劫持智能体行为

---

## 安全与注意事项

- API Key 只从环境变量读取，代码与配置中只保留环境变量名，任何文件、日志、对话记录都不写入 Key
- `models/*.pt` 训练权重（约 257MB）与 `data/train`、`data/val` 数据集不随仓库提交；clone 后需自行训练或放置权重到 `models/best.pt`
- 上传图片走白名单扩展名 + PIL 真实解码 + uuid 重命名，杜绝路径穿越与改后缀
- 视觉推理用可重入锁串行化，避免多线程请求打爆 4GB 显存

---

## License

本项目为课程实训作业，仅供学习交流。
