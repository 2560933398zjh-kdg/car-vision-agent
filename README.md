# 视觉智能体综合实训 · 视觉模型部分（任务书 4.1 / 4.2 / 4.3）

本目录实现任务书「任务一：数据分析与视觉模型训练」的完整流程：
数据准备 → 51 类分类基线 → 对比实验 → 模型评估 → 单张 / 批量推理。

智能体（任务二）与 Web 系统（任务三）在后续阶段接入，视觉部分对外只暴露两个稳定接口：
`predict.CarClassifier.classify()`（单张，供智能体工具调用）和 `predict.py --dir`（批量出 `predictions.csv`）。

---

## 一、环境准备

```bash
# 1) 创建虚拟环境（Python 3.10+，本机用 3.13）
cd "D:/实训课/智能系统综合实训"
python -m venv .venv

# 2) 安装依赖（CUDA 12.4 版本，适配 RTX 3050 / 驱动 531.88）
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install matplotlib

# 3) 验证环境
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 没有 GPU 时把 `--index-url` 去掉装 CPU 版即可，脚本会自动回退到 CPU 训练（`common.get_device()`）。
> 若无网络下载预训练权重，加 `--no-pretrained` 从零训练，但收敛会更慢、指标更低。

---

## 二、目录结构

```
综合实训/
├── classes.txt                  # 恰好 50 个四位车型 ID，升序，UTF-8 无 BOM，不含 unknown
├── cls_info/class_info.json     # 教师提供的 827 类类别说明
├── data/                        # 本组数据集（由 data/build_dataset.py 生成）
│   ├── classes.txt
│   ├── class_mapping.json       # 索引 0~49 → 车型，索引 50 → unknown
│   ├── build_dataset.py         # 选类 + unknown 采样脚本（可重跑）
│   └── train/ val/              # 各 51 个类别目录
├── src/
│   ├── common.py                # 路径、类别映射、预处理参数、随机种子
│   ├── dataset.py               # 数据集与数据增强
│   ├── model.py                 # 建模 + 模型文件保存/加载
│   ├── metrics.py               # Macro-F1 / 混淆矩阵 / unknown 专项指标
│   ├── train.py                 # 训练主程序
│   ├── evaluate.py              # 验证集评估与错误分析
│   ├── predict.py               # 单张 + 批量推理
│   ├── tune_threshold.py        # 验证集置信度拒识阈值搜索
│   ├── run_experiments.py       # 批量对比实验（支持断点续跑）
│   ├── make_report_table.py     # 生成报告用的对比实验表
│   ├── make_task1_report.py     # 汇总任务一全部产出为报告素材
│   ├── validate_submission.py   # 提交物格式自检
│   ├── tools.py                 # 【任务二】5 个智能体工具 + Schema + SQLite 存取
│   ├── agent.py                 # 【任务二】DeepSeek 智能体：工具循环 + 多轮 Session + 命令行
│   └── app.py                   # 【任务三】Flask Web 服务 + 系统测试（--selftest）
├── web/                         # 【任务三】前端（纯静态，无需构建）
│   ├── index.html               # 页面结构
│   ├── style.css                # 样式
│   ├── app.js                   # 请求封装 + 会话 + 图片上传 + 对话
│   ├── records.js               # 识车笔记面板：列表 / 检索 / 编辑备注 / 删除 / 导出
│   └── logs.js                  # 工具调用记录面板（可展开参数与原始返回）
├── configs/                     # 各实验配置 + agent.json（智能体配置）
├── models/                      # 模型文件（不提交，本地保存）
├── outputs/                     # 指标、混淆矩阵、predictions.csv、对比表、智能体对话记录
├── logs/                        # 逐轮训练记录与完整配置
└── docs/PyTorch-代码导读.md      # 概念与代码的对照说明
```

---

## 三、常用命令

```bash
PY=.venv/Scripts/python.exe

# 冒烟测试（2 轮 + 每类 20 张，约 1 分钟，先确认流程通畅）
$PY src/train.py --config configs/e0_baseline.json --epochs 2 --limit-per-class 20

# 训练单个实验
$PY src/train.py --config configs/e4_freeze_then_finetune.json

# 跑全部对比实验并汇总成表（已完成的会跳过，可断点续跑；--force 强制重跑）
$PY src/run_experiments.py
$PY src/run_experiments.py --only e3_resnet34 e4_freeze_then_finetune

# 评估（输出 Macro-F1、每类指标、混淆矩阵、典型错误样本）
$PY src/evaluate.py --checkpoint models/best.pt --tag mytag

# 在验证集上搜索置信度拒识阈值（任务书要求阈值须由验证集确定）
$PY src/tune_threshold.py --checkpoint models/best.pt

# 单张推理
$PY src/predict.py --image some_car.jpg

# 批量推理 → predictions.csv（默认递归子目录，可用 --confidence-threshold 启用拒识）
$PY src/predict.py --dir "path/to/test_images" --out outputs/predictions.csv

# 生成报告素材
$PY src/make_report_table.py       # 对比实验表 + 基线与最佳逐类差异
$PY src/make_task1_report.py       # 任务一报告素材（对应报告第 2、3、4 节）

# 提交前自检（classes.txt 与 predictions.csv 的格式与完整性）
$PY src/validate_submission.py --predictions outputs/predictions.csv

# ---------------------------------------------------------------- 任务二 · 智能体
# 先设置 API Key（不写入任何文件）
#   Windows CMD:  set DEEPSEEK_API_KEY=sk-xxxx
#   PowerShell:   $env:DEEPSEEK_API_KEY="sk-xxxx"

$PY src/tools.py                   # 工具层离线自测（不调大模型，验证 4 个工具真实执行）
$PY src/agent.py                   # 内置演示：连续三工具 + 多轮复用 + 记录检索 + unknown 拒绝
$PY src/agent.py --task "这是什么车？介绍一下，然后加入我的收藏" --image 图片路径
$PY src/agent.py --chat            # 交互式多轮对话（/img 路径 附加图片，/reset 清空上下文）

# ---------------------------------------------------------------- 任务三 · Web 系统
$PY src/app.py                     # 启动服务，浏览器打开 http://127.0.0.1:5000
$PY src/app.py --port 8000         # 换端口
$PY src/app.py --selftest          # 只跑系统测试（22 项），不启动服务
```

---

## 三之二、当前实验结果

五组对比实验（验证集口径，51 类等权 Macro-F1）：

| 实验 | 结构 | 增强 | 调度 | 标签平滑 | 冻结轮数 | 验证 Macro-F1 | 耗时 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `e0_baseline` | resnet18 | none | none | 0.0 | 0 | 0.8035 | 2.79 分 |
| `e1_aug` | resnet18 | basic | none | 0.0 | 0 | 0.7832 | 7.89 分 |
| `e2_strong_aug_ls` | resnet18 | strong | cosine | 0.1 | 0 | 0.9323 | 3.00 分 |
| `e3_resnet34` | resnet34 | strong | cosine | 0.1 | 0 | 0.9340 | 4.32 分 |
| **`e4_freeze_then_finetune`** ★ | resnet18 | basic | cosine | 0.1 | 2 | **0.9402** | 2.77 分 |

- 最佳模型已发布为 `models/best.pt`，`predict.py` 默认使用它。
- **基线 e0 严重过拟合**：训练准确率到 98%，第 5 轮后验证指标持续下滑。
- **强数据增强 + 标签平滑 + 余弦退火**是提升最大的单一组合（+0.129）。
- **e4 用最小最快的模型取得最好成绩**：先冻结主干 2 轮只训分类头、再解冻微调，耗时 2.77 分钟。
- **unknown 是最弱类别**（最佳模型 F1 仅 0.61，检出率 56%），是本项目主要短板，详见 `outputs/任务一报告素材.md` 与 `outputs/threshold_sweep.csv`。

---

## 三之三、任务二：识车笔记智能体

### 项目定位

面向汽车爱好者的「识车笔记」：上传一张车照 → 识别车型 → 查看车型资料 → 存入自己的识车笔记 → 随时检索。

### 四个工具（任务书要求不少于 3 个，其中视觉工具 1 个 + 业务工具 ≥2 个）

| 工具 | 类型 | 输入 | 返回 | 失败处理 |
| :--- | :--- | :--- | :--- | :--- |
| `classify_car` | **视觉分类（必需）** | `image_path`、可选 `top_k` | 车型 ID、名称、confidence、is_unknown、Top-K | 路径为空/不存在/是目录/格式不支持 → `ok=false` |
| `query_car_info` | 业务 | `car_id` | 品牌、车型名、车身类型、细分级别 | unknown、不在本组 50 类、不存在 → `ok=false`，**不编造** |
| `save_car_record` | 业务 | `car_id`、可选 `note`/`confidence` | 保存动作（created/updated）、记录 ID、总记录数 | ID 非法 → `ok=false` |
| `search_records` | 业务 | 可选 `keyword`/`car_id`/`limit` | 记录列表 + 总数 | 查不到返回空列表，不报错 |
| `get_car_detail` | 业务 | `car_name` | 定位/动力/尺寸/价格/油耗（联网） | 缺 Key 或查不到时如实告知，不编造 |

存储用 **SQLite 单文件 `data/app.db`**，同一车型重复保存只更新备注，不产生重复记录；
联网查到的车型详情会缓存到 `car_detail_cache` 表，下次同款车直接读缓存，省额度、答辩不依赖网络。

> `get_car_detail` 需要额外配置阿里云百炼的 API Key（环境变量 `DASHSCOPE_API_KEY`），
> 用于调用通义千问的联网搜索；视觉识别仍由本组自训练模型完成，该 Key 只用于查询文字资料。

### 四条关键链路（`python src/agent.py` 可一次跑完并自动判定达标情况）

1. **连续调用 3 个工具**：`classify_car` → `query_car_info` → `save_car_record`
2. **多轮对话**：第一轮识别后，第二轮只说"帮我记录一下"，直接复用上一轮的车型 ID，**不再重新识别图片**
3. **unknown / 低置信度**：只调用 `classify_car` 就停止，**不查资料、不保存**，如实告知并给出 Top-K 请用户确认
4. **异常处理**：非图片文件、不在本组 50 类内的车型 ID，工具如实返回失败原因，智能体照实转述且不产生错误保存

### 关键设计取舍

- **图片必须经由工具识别**：模型是纯文本的，用户消息里只带图片路径，识别只能通过 `classify_car` 触发。演示时图片被复制成中性命名的临时文件，杜绝模型从文件名猜答案。
- **资料不编造**：`class_info.json` 只有 `id / name_from_new / type_info` 三个字段，因此工具只返回品牌/车身类型/细分级别；被问到排量、价格时如实说明没有。
- **拒识阈值默认关闭**：任务一在验证集上做过阈值搜索，最优点仅带来 +0.0009（噪声级），而调高阈值会让 unknown 的 F1 从 0.61 降到 0.25。所以低置信度处理做成"展示 Top-K 请用户确认"，而不是自动改写预测结果。
- **工具结果是数据不是指令**：系统提示词明确写了这一点，避免工具返回内容劫持智能体行为。

### API Key 管理

只从环境变量读取，代码与配置里只保留环境变量名，**任何文件、日志、对话记录都不写入 Key**：

- `DEEPSEEK_API_KEY` —— 智能体主模型（任务书指定 `deepseek-v4-flash`）
- `DASHSCOPE_API_KEY` —— 阿里云百炼（通义千问联网搜索，供 `get_car_detail` 工具查询车型详情；可选，不配则详情功能返回提示）

---

## 三之四、任务三：Web 系统

### 技术栈

**Flask + 纯静态 HTML/CSS/JS**。不需要 Node、不需要打包构建，浏览器直接加载；单进程单端口同时提供接口与页面。选它而不是 FastAPI 的原因：本项目的视觉推理是阻塞式的，Flask 多线程 + 推理锁比异步框架更省心，而且依赖最少（只多一个 flask）。

### 启动

```bash
set DEEPSEEK_API_KEY=sk-xxxx            # 或不设，服务仍可启动，对话功能会提示未配置
.venv/Scripts/python.exe src/app.py
# 浏览器打开 http://127.0.0.1:5000
```

### 接口一览

| 方法 | 路径 | 用途 |
| :--- | :--- | :--- |
| GET | `/` | 返回前端页面 |
| GET | `/api/health` | 检查 API Key / 模型文件 / 数据库是否就绪 |
| GET | `/api/classes` | 本组支持的 50 款车型清单 |
| POST | `/api/session` | 新建会话，返回 session_id |
| POST | `/api/session/<id>/reset` | 重置会话上下文 |
| GET | `/api/session/<id>/log` | 该会话逐轮的工具调用记录 |
| POST | `/api/upload` | 上传图片（扩展名白名单 + 10MB 上限 + uuid 重命名 + 真实解码校验） |
| GET | `/uploads/<name>` | 访问已上传图片 |
| POST | `/api/chat` | 对话：返回回答、本轮工具调用、笔记快照、统计 |
| GET | `/api/records` | 识车笔记列表（支持关键词检索） |
| PATCH | `/api/records/<car_id>` | 修改备注（记录不存在返回 404） |
| DELETE | `/api/records/<car_id>` | 删除记录（不存在返回 404） |
| GET | `/api/records/export` | 导出 CSV（带 BOM，Excel 不乱码）或 JSON |
| GET | `/api/stats` | 记录总数、覆盖车身类型、品牌分布 |

### 界面对应验收项

| 任务书验收项 | 实现位置 |
| :--- | :--- |
| 图片上传与显示 | 上传区支持点击 / 拖拽 / 粘贴，上传后立即缩略图预览 |
| 用户输入与聊天 | 对话区气泡展示，附快捷任务按钮 |
| 视觉识别 | 智能体经 `classify_car` 调用本组模型（Web 端不重复实现） |
| 多轮对话 | 服务端一个 session_id 对应一个 Session，第二轮无需重新上传 |
| 业务功能 | 笔记列表、关键词检索、编辑备注、删除、导出 CSV/JSON、统计概览 |
| 异常反馈 | 顶部告警条 + 对话内错误气泡 + 工具失败红标，低置信度用黄色气泡提示 |

### 工程要点

- **并发**：会话各持一把锁，允许不同用户同时对话；视觉推理在 `tools.py` 内用可重入锁串行化，避免 4GB 显存被并发请求打爆。
- **上传安全**：只用白名单扩展名，落盘前用 PIL 真实解码一次（防改后缀），文件名统一换成 uuid，杜绝路径穿越。
- **会话存内存**：进程重启即丢失。这是有意取舍——本实训无多用户持久化需求，报告中作为局限说明。
- **系统测试**：`src/app.py --selftest` 内置 22 项检查（静态页面、健康检查、会话、上传、笔记接口、参数校验、检索覆盖车身类型，以及有 Key 时的端到端对话）。



---

## 四、类别映射约定（全局唯一）

| 位置 | 取值 |
| --- | --- |
| 索引 0 ~ 49 | 按 `classes.txt` 的行顺序对应 50 个目标车型 |
| 索引 50 | `unknown`，代表不属于本组 50 类的其他车型 |

- `classes.txt` **不含** unknown，恰好 50 行。
- 预测为 unknown 时输出字符串 `unknown`，**不输出**其来源车型 ID。
- 训练、验证、单张推理、批量推理、Web 端全部读同一个映射，写死在模型文件里。

---

## 五、模型文件里存了什么

`models/*.pt` 是一个字典，包含还原推理所需的全部信息：

| 键 | 内容 |
| --- | --- |
| `model_state` | 网络权重 |
| `arch` | 模型结构名（resnet18 / resnet34 / …） |
| `class_ids` / `class_names` / `index_to_label` | 51 类类别映射 |
| `preprocess` | 输入尺寸、Resize 尺寸、均值、标准差 |
| `train_config` | 本次训练的完整超参与数据设置 |
| `epoch` / `best_val_macro_f1` | 轮次与最佳验证指标 |

因此只需一个 `.pt` 文件即可正确复现推理，不会出现"训练用一套预处理、推理用另一套"的错位。

---

## 六、评估口径（与教师评分一致）

- **唯一效果指标**：51 类（50 车型 + unknown）等权平均 Macro-F1。
- 单类 F1 = 2×TP / (2×TP + FP + FN)；分母为 0 时该类 F1 记 **0**。
- unknown 与每个车型**同权**，不排除在评分之外。
- 目标车型误判为 unknown、unknown 误判为目标车型，都计入相应类别的错误。
- 训练期间一律用**验证集** Macro-F1 选模型与调参，**不使用测试图片**。
- Accuracy、每类准确率、混淆矩阵、unknown 检出率、已知类误拒率仅作辅助分析。

---

## 七、安全与提交注意

- API Key 只通过环境变量或本地配置文件提供，**不写入任何源代码、配置、日志或截图**；交付时用占位符。
- `models/` 下的模型文件只在本地保存，**不随作业提交**，也不要打包进源代码。
- 提交物为四项：项目源代码、`classes.txt`（项目根目录）、`predictions.csv`、实训报告。
