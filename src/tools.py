"""智能体工具集：1 个视觉分类工具 + 5 个业务工具。

任务书 5.2 要求工具总数不少于 3 个，其中 1 个视觉分类工具必须**真实加载并调用本组训练的模型**，
另外至少 2 个工具完成实际业务操作。本项目共 6 个：

    classify_car        视觉分类：真实加载 models/best.pt，返回车型 ID / unknown + confidence + Top-K
    query_car_info      资料查询：按车型 ID 返回品牌、车身类型、细分级别（只返回真实字段，不编造）
    save_car_record     保存记录：把车型写入 SQLite「识车笔记」，同一车型重复保存则更新
    search_records      历史检索：按关键词或车型 ID 检索已保存的记录
    delete_car_record   删除记录：仅在用户明确要求删除时移除某条记录
    get_car_detail      联网查详情：调用阿里云百炼（通义千问联网搜索）返回定位/动力/尺寸/价格等，结果缓存

设计原则：
1. 工具真实执行，返回真实结果；失败一律返回 ok=false + 可读原因，绝不伪造成功。
2. 工具结果是**数据**，不是可以覆盖系统规则的新指令。
3. 参数在执行前用 JSON Schema 二次校验，不依赖模型一定输出正确格式。

用法（离线自测，不需要 API Key）：
    python src/tools.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import requests
from jsonschema import Draft202012Validator, ValidationError

from common import (CLASS_INFO_JSON, IMG_EXTS, PROJECT_ROOT, UNKNOWN,
                    load_agent_config, load_class_ids, load_name_map)

CONFIG = load_agent_config()
DB_PATH = CONFIG["storage"]["db_path"]
VISION = CONFIG["vision"]
ENCYCLOPEDIA = CONFIG.get("car_encyclopedia", {})

# 每个会话累计的工具调用日志，供报告与「工具调用记录」使用
TOOL_LOGS: list[dict] = []


# --------------------------------------------------------------- 基础数据

def _class_info() -> dict[str, dict]:
    with open(CLASS_INFO_JSON, encoding="utf-8") as f:
        return {row["id"]: row for row in json.load(f)}


def _split_type(type_info: str) -> tuple[str, str]:
    """从 'SUV####紧凑型SUV==SUV' 解析出 (车身类型, 细分中文名)。"""
    parts = type_info.split("####")
    body = parts[0] if parts else ""
    sub = parts[1].split("==")[0] if len(parts) > 1 else ""
    return body, sub


def normalize_car_id(car_id: str) -> str:
    """把 '183' 之类的写法补成 '0183'，避免模型漏掉前导零。"""
    text = str(car_id or "").strip()
    if text.isdigit() and len(text) < 4:
        return text.zfill(4)
    return text


# --------------------------------------------------------------- SQLite

SCHEMA = """
CREATE TABLE IF NOT EXISTS car_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id      TEXT    NOT NULL UNIQUE,
    car_name    TEXT    NOT NULL,
    brand       TEXT,
    body_type   TEXT,
    sub_type    TEXT,
    confidence  REAL,
    image_path  TEXT,
    note        TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS car_detail_cache (
    car_name    TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    source      TEXT,
    updated_at  TEXT NOT NULL
);
"""


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)   # SCHEMA 含多条建表语句，必须用 executescript
    return conn


@contextmanager
def db():
    """一次数据库事务：正常结束提交、出错回滚，并且一定会关闭连接。"""
    conn = get_conn()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------- 视觉模型

_CLASSIFIER = None
# Web 服务是多线程的，而显存只有 4GB：把「加载模型」和「前向推理」都串行化，避免并发 OOM。
# 用 RLock（可重入）而不是 Lock：classify_car 持锁期间可能触发首次模型加载，
# 若用普通 Lock 会在同一线程内二次加锁，造成死锁。
_INFER_LOCK = threading.RLock()


def get_classifier():
    """惰性加载视觉模型，整个进程只加载一次。"""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from predict import CarClassifier  # 延迟导入，避免自测时也必须加载 torch
        with _INFER_LOCK:
            if _CLASSIFIER is None:
                _CLASSIFIER = CarClassifier(VISION["checkpoint"],
                                            confidence_threshold=VISION["confidence_threshold"])
    return _CLASSIFIER


def resolve_image_path(image_path: str) -> str:
    text = str(image_path or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError("图片路径为空")
    path = text if os.path.isabs(text) else os.path.join(PROJECT_ROOT, text)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise ValueError(f"文件不存在：{path}")
    if os.path.isdir(path):
        raise ValueError(f"这是一个目录，不是图片：{path}")
    if not path.lower().endswith(IMG_EXTS):
        raise ValueError(f"不支持的图片格式：{os.path.basename(path)}")
    return path


# --------------------------------------------------------------- 工具 1：视觉分类

def classify_car(image_path: str, top_k: int | None = None) -> dict:
    """识别图片中的车型。真实调用本组训练的视觉模型。"""
    path = resolve_image_path(image_path)
    k = int(top_k) if top_k else int(VISION["top_k"])
    network = get_classifier()              # 先确保模型已加载（内部自带锁）
    with _INFER_LOCK:                       # 再串行执行前向推理，避免多请求抢占显存
        result = network.classify(path, topk=k)
    if not result.get("ok"):
        raise ValueError(result.get("error", "识别失败"))

    payload = {
        "image": result["image"],
        "car_id": result["label"],
        "car_name": result["name"],
        "confidence": result["confidence"],
        "is_unknown": result["is_unknown"],
        "top_k": [{"car_id": t["label"], "car_name": t["name"], "confidence": t["confidence"]}
                  for t in result["topk"]],
        "model": result["model"],
    }
    if result["is_unknown"]:
        payload["note"] = "预测类别为 unknown：这张图片不属于本组支持的 50 款车型。"
    return payload


# --------------------------------------------------------------- 工具 2：资料查询

def query_car_info(car_id: str) -> dict:
    """按车型 ID 查询资料。只返回类别说明文件中真实存在的字段，不编造任何参数。"""
    cid = normalize_car_id(car_id)
    if cid.lower() == UNKNOWN:
        raise ValueError("unknown 表示不属于本组 50 款的车型，没有对应的车型资料，不得编造")

    allowed = load_class_ids()
    if cid not in allowed:
        info = _class_info()
        if cid in info:
            raise ValueError(f"{cid} 属于教师提供的 827 类，但不在本组选定的 50 款车型内，无资料可提供")
        raise ValueError(f"类别 ID {cid} 不存在于教师提供的 827 个车型类别中")

    info = _class_info()[cid]
    body, sub = _split_type(info["type_info"])
    brand, _, model = info["name_from_new"].partition("_")
    return {
        "car_id": cid,
        "car_name": info["name_from_new"],
        "brand": brand,
        "model": model or info["name_from_new"],
        "body_type": body,
        "sub_type": sub,
        "type_info_raw": info["type_info"],
        "data_source": "教师提供的类别说明文件 class_info.json",
    }


# --------------------------------------------------------------- 工具 3：保存记录

def save_car_record(car_id: str, note: str = "", image_path: str | None = None,
                    confidence: float | None = None) -> dict:
    """把车型写入「识车笔记」。同一车型重复保存则更新备注，不产生重复记录。"""
    info = query_car_info(car_id)          # 顺带校验 ID 合法性，unknown 会在这里被拒绝
    cid = info["car_id"]
    note = (note or "").strip()
    if len(note) > 200:
        raise ValueError("备注过长，请控制在 200 字以内")

    now = _now()
    with db() as conn:
        existing = conn.execute("SELECT id, created_at FROM car_records WHERE car_id = ?", (cid,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE car_records SET note = ?, confidence = COALESCE(?, confidence), "
                "image_path = COALESCE(?, image_path), updated_at = ? WHERE car_id = ?",
                (note or "", confidence, image_path, now, cid))
            action, record_id = "updated", existing["id"]
        else:
            cur = conn.execute(
                "INSERT INTO car_records (car_id, car_name, brand, body_type, sub_type, confidence,"
                " image_path, note, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, info["car_name"], info["brand"], info["body_type"], info["sub_type"],
                 confidence, image_path, note, now, now))
            action, record_id = "created", cur.lastrowid
    return {
        "action": action,
        "record_id": record_id,
        "car_id": cid,
        "car_name": info["car_name"],
        "note": note,
        "storage": os.path.relpath(DB_PATH, PROJECT_ROOT).replace("\\", "/"),
        "total_records": search_records(limit=1)["count"],
    }


# --------------------------------------------------------------- 工具 4：历史检索

def search_records(keyword: str = "", car_id: str = "", limit: int = 20) -> dict:
    """检索已保存的识车记录。查不到时返回空列表，不报错。

    上限放到 1000 是给 Web 端的「导出全部记录」用；智能体看到的工具 Schema 仍限制在 100 以内。
    """
    limit = max(1, min(int(limit or 20), 1000))
    sql = ("SELECT car_id, car_name, brand, body_type, sub_type, confidence, note, created_at, updated_at "
           "FROM car_records WHERE 1=1")
    params: list = []
    if car_id:
        sql += " AND car_id = ?"
        params.append(normalize_car_id(car_id))
    if keyword:
        # 车身类型与细分级别也要参与匹配，否则用户问「我收藏的跑车有哪些」会查不到
        sql += (" AND (car_name LIKE ? OR note LIKE ? OR brand LIKE ?"
                " OR body_type LIKE ? OR sub_type LIKE ?)")
        like = f"%{keyword.strip()}%"
        params += [like] * 5
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    with db() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        total = conn.execute("SELECT COUNT(*) AS c FROM car_records").fetchone()["c"]
    return {"items": rows, "count": total, "returned": len(rows)}


# --------------------------------------------------------------- 工具 5：删除记录

def delete_car_record(car_id: str) -> dict:
    """删除一条识车记录。仅在用户明确要求删除或移除时调用。"""
    cid = normalize_car_id(car_id)
    with db() as conn:
        row = conn.execute("SELECT car_id, car_name FROM car_records WHERE car_id = ?", (cid,)).fetchone()
        if row is None:
            raise ValueError(f"识车笔记中没有 {cid} 这条记录，无需删除")
        conn.execute("DELETE FROM car_records WHERE car_id = ?", (cid,))
    return {"deleted": True, "car_id": row["car_id"], "car_name": row["car_name"],
            "remaining": search_records(limit=1)["count"]}


# --------------------------------------------------------------- 编辑备注（供 Web 用）

def update_record_note(car_id: str, note: str) -> dict:
    """只修改已存在的记录的备注。记录不存在时报错，不会凭空新建。

    与 save_car_record 的区别：后者是「保存/收藏」语义，没有记录时会新建；
    这里专门服务 Web 界面的「编辑备注」，必须要求记录已存在。
    """
    cid = normalize_car_id(car_id)
    note = (note or "").strip()
    if len(note) > 200:
        raise ValueError("备注过长，请控制在 200 字以内")
    with db() as conn:
        row = conn.execute("SELECT car_id, car_name FROM car_records WHERE car_id = ?",
                           (cid,)).fetchone()
        if row is None:
            raise ValueError(f"识车笔记中没有 {cid} 这条记录，无法修改备注")
        conn.execute("UPDATE car_records SET note = ?, updated_at = ? WHERE car_id = ?",
                     (note, _now(), cid))
    return {"car_id": row["car_id"], "car_name": row["car_name"], "note": note,
            "total_records": search_records(limit=1)["count"]}


# --------------------------------------------------------------- 统计（供 Web 面板用）

def get_stats() -> dict:
    """识车笔记的概览统计，用于 Web 页面顶部指标卡。"""
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM car_records").fetchone()["c"]
        by_body = [dict(r) for r in conn.execute(
            "SELECT body_type, COUNT(*) AS n FROM car_records "
            "GROUP BY body_type ORDER BY n DESC, body_type").fetchall()]
        by_brand = [dict(r) for r in conn.execute(
            "SELECT brand, COUNT(*) AS n FROM car_records "
            "GROUP BY brand ORDER BY n DESC, brand LIMIT 8").fetchall()]
        latest = conn.execute("SELECT MAX(updated_at) AS t FROM car_records").fetchone()["t"]
        noted = conn.execute("SELECT COUNT(*) AS c FROM car_records WHERE note <> ''").fetchone()["c"]
    return {"total": total, "with_note": noted, "body_types": len(by_body),
            "latest_at": latest, "by_body_type": by_body, "by_brand": by_brand}


# --------------------------------------------------------------- 工具 6：联网查车型详情

_ENCYCLOPEDIA_PROMPT = (
    "你是汽车资料整理助手。请联网搜索并整理车型「{name}」的详细信息，用简体中文回答，"
    "严格按下面这个小节结构输出（小节标题用【】括起来）：\n"
    "【车型定位】级别与车型定位，一句话\n"
    "【动力系统】发动机或电机、排量、马力/功率、变速箱\n"
    "【车身尺寸】长宽高、轴距\n"
    "【价格区间】官方指导价区间\n"
    "【油耗或续航】\n"
    "【一句话点评】\n"
    "最后单独一行注明「数据来源：…」（如汽车之家、懂车帝、厂商官网等）。\n"
    "如果确实搜不到这款车，就明确回答「未找到该车型的公开资料」，不要编造任何参数。"
)


def _normalize_car_name(car_name: str) -> str:
    """把 '奔驰_奔驰GLE' 这类「品牌_车型」整理成方便搜索的名称。"""
    text = str(car_name or "").strip().replace("_", " ")
    # 去掉相邻重复词，例如「奔驰 奔驰GLE」→「奔驰GLE」
    parts = text.split()
    out = []
    for word in parts:
        if out and (word in out[-1] or out[-1] in word):
            out[-1] = word if len(word) > len(out[-1]) else out[-1]
        else:
            out.append(word)
    return " ".join(out) or text


def _call_encyclopedia(name: str, key: str) -> str:
    """调用阿里云百炼（通义千问 + 联网搜索）获取车型详情，返回模型生成的文本。"""
    cfg = ENCYCLOPEDIA or {}
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.get("model", "qwen-plus"),
        "messages": [
            {"role": "system", "content": "你只根据搜索结果整理资料，绝不编造参数。"},
            {"role": "user", "content": _ENCYCLOPEDIA_PROMPT.format(name=name)},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
        "enable_search": cfg.get("enable_search", True),
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=tuple(cfg.get("timeout", [10, 60])),
    )
    if not resp.ok:
        raise ValueError(f"百炼接口返回 HTTP {resp.status_code}：{resp.text[:200]}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("百炼接口返回格式异常，未取到内容")
    content = (content or "").strip()
    if not content:
        raise ValueError("百炼接口返回了空内容")
    return content


def get_car_detail(car_name: str) -> dict:
    """联网查询车型的详细信息（定位、动力、尺寸、价格等）。

    数据来源是阿里云百炼（通义千问）的联网搜索，而不是大模型凭记忆编造。
    查询结果会缓存到 SQLite，之后同款车再查直接读缓存，省额度、响应更快，
    也保证答辩演示时不依赖网络。
    """
    name = _normalize_car_name(car_name)
    if not name:
        raise ValueError("车型名称为空，无法查询")

    # 1) 先查本地缓存
    with db() as conn:
        row = conn.execute("SELECT content, source, updated_at FROM car_detail_cache "
                           "WHERE car_name = ?", (name,)).fetchone()
    if row is not None:
        return {"car_name": name, "detail": row["content"], "source": row["source"],
                "from_cache": True, "cached_at": row["updated_at"]}

    # 2) 缓存未命中，调用百炼联网搜索
    key = os.environ.get((ENCYCLOPEDIA or {}).get("api_key_env", "DASHSCOPE_API_KEY"), "").strip()
    if not key:
        raise ValueError("服务端未配置环境变量 DASHSCOPE_API_KEY，无法联网查询车型详情；"
                         "可改用 query_car_info 查询本地基础资料")
    content = _call_encyclopedia(name, key)
    source = "阿里云百炼·通义千问联网搜索"

    # 3) 写回缓存
    with db() as conn:
        conn.execute("INSERT INTO car_detail_cache (car_name, content, source, updated_at) "
                     "VALUES (?, ?, ?, ?) ON CONFLICT(car_name) DO UPDATE SET "
                     "content=excluded.content, source=excluded.source, updated_at=excluded.updated_at",
                     (name, content, source, _now()))
    return {"car_name": name, "detail": content, "source": source,
            "from_cache": False, "cached_at": None}


# --------------------------------------------------------------- 工具描述（给模型看）

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _tool("classify_car",
          "识别图片中的车型，返回车型 ID、名称与置信度。图片识别必须通过本工具完成，"
          "不要凭描述猜测车型。返回 is_unknown=true 表示不属于本组支持的 50 款车型。",
          {"image_path": {"type": "string", "description": "图片文件路径，通常直接使用用户消息中给出的路径"},
           "top_k": {"type": "integer", "minimum": 1, "maximum": 5, "description": "返回前 K 个候选，默认 3"}},
          ["image_path"]),
    _tool("query_car_info",
          "按车型 ID 查询该车型的品牌、车身类型与细分级别。只有在拿到真实车型 ID 后才能调用，"
          "不要传入 unknown 或自行编造的 ID。本工具不提供排量、价格等参数，也不要凭空补充。",
          {"car_id": {"type": "string", "description": "四位车型 ID，如 0183；不要传 unknown"}},
          ["car_id"]),
    _tool("save_car_record",
          "把车型保存进用户的识车笔记。仅在用户明确要求保存/收藏/记录时调用，"
          "且必须使用 classify_car 返回的真实车型 ID。",
          {"car_id": {"type": "string", "description": "要保存的四位车型 ID"},
           "note": {"type": "string", "maxLength": 200, "description": "可选备注"},
           "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "识别时的置信度"}},
          ["car_id"]),
    _tool("search_records",
          "检索用户已保存的识车记录，可按关键词或车型 ID 过滤。用户询问“我收藏了什么/有几条记录”时调用。",
          {"keyword": {"type": "string", "description": "在车型名称、品牌、备注中模糊匹配"},
           "car_id": {"type": "string", "description": "按车型 ID 精确过滤"},
           "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "最多返回条数，默认 20"}},
          []),
    _tool("delete_car_record",
          "从识车笔记中删除一条记录。仅在用户明确要求删除或移除某条记录时调用，"
          "不要因为用户只是查看或修改备注就删除。",
          {"car_id": {"type": "string", "description": "要删除的四位车型 ID"}},
          ["car_id"]),
    _tool("get_car_detail",
          "联网查询车型的详细信息（定位、动力、尺寸、价格、油耗等），数据来自阿里云百炼的联网搜索，"
          "不是模型凭记忆编造。当用户问“详细信息/配置/参数/多少钱/多大排量”等需要上网查资料的问题时调用，"
          "传入车型名称（从 classify_car 或 query_car_info 返回的 car_name/name 里取）。"
          "query_car_info 只提供本地基础资料，用户要完整参数时必须用本工具。",
          {"car_name": {"type": "string", "description": "车型名称，如 奔驰GLE 或 荣威iMAX8"}},
          ["car_name"]),
]

TOOL_FUNCTIONS = {
    "classify_car": classify_car,
    "query_car_info": query_car_info,
    "save_car_record": save_car_record,
    "search_records": search_records,
    "delete_car_record": delete_car_record,
    "get_car_detail": get_car_detail,
}
TOOL_DEFINITIONS = {t["function"]["name"]: t for t in TOOL_SCHEMAS}

for _t in TOOL_SCHEMAS:
    Draft202012Validator.check_schema(_t["function"]["parameters"])


# --------------------------------------------------------------- 执行器

def execute_tool(tool_call: dict) -> dict:
    """白名单查找 → 解析 JSON → Schema 校验 → 调用函数 → 记录日志。

    任何失败都转成 ok=false 的结构化结果，交给模型去解释或修正参数，
    而不是抛异常中断整个会话，也不伪造成功。
    """
    started = time.perf_counter()
    name = (tool_call.get("function") or {}).get("name", "")
    arguments = None
    try:
        if name not in TOOL_FUNCTIONS:
            raise ValueError(f"工具 {name!r} 不在允许调用的白名单中")
        arguments = json.loads((tool_call["function"].get("arguments") or "{}"))
        if not isinstance(arguments, dict):
            raise ValueError("参数必须是 JSON 对象")
        Draft202012Validator(TOOL_DEFINITIONS[name]["function"]["parameters"]).validate(arguments)
        result = {"ok": True, "data": TOOL_FUNCTIONS[name](**arguments)}
    except json.JSONDecodeError:
        result = {"ok": False, "error": "参数不是有效的 JSON 对象"}
    except ValidationError as exc:
        result = {"ok": False, "error": "参数校验失败：" + exc.message[:200]}
    except (ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "error": str(exc)[:300]}
    except Exception as exc:                                    # noqa: BLE001
        result = {"ok": False, "error": f"工具内部执行失败：{exc.__class__.__name__}"}

    TOOL_LOGS.append({
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "tool_call_id": tool_call.get("id"),
        "name": name,
        "arguments": arguments,
        "ok": result["ok"],
        "result": result,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    })
    return result


def tool_result_message(tool_call: dict) -> dict:
    """构造回传给模型的 role=tool 消息，tool_call_id 必须与请求中的调用 ID 对应。"""
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": json.dumps(execute_tool(tool_call), ensure_ascii=False),
    }


# --------------------------------------------------------------- 离线自测

def _selftest() -> None:
    print("=" * 70)
    print("工具层离线自测（不调用大模型 API）")
    print("=" * 70)

    print("\n[1] classify_car —— 真实加载视觉模型")
    target = os.path.join(PROJECT_ROOT, "data", "val", "0183")
    sample = sorted(os.listdir(target))[0]
    r = classify_car(os.path.join(target, sample))
    print(f"    输入 {sample} → {r['car_id']} {r['car_name']} 置信度={r['confidence']} "
          f"is_unknown={r['is_unknown']}")
    assert r["car_id"] == "0183", "视觉工具返回的车型 ID 与目录标签不一致"

    unknown_dir = os.path.join(PROJECT_ROOT, "data", "val", "unknown")
    u = classify_car(os.path.join(unknown_dir, sorted(os.listdir(unknown_dir))[0]))
    print(f"    unknown 样本 → {u['car_id']} {u['car_name']} 置信度={u['confidence']} "
          f"is_unknown={u['is_unknown']}")

    print("\n[2] 图片读取失败的容错")
    for bad in ["", "不存在的目录/x.jpg", str(target)]:
        try:
            classify_car(bad)
            print(f"    {bad!r} → 未报错（异常）")
        except ValueError as exc:
            print(f"    {bad!r} → 正确拒绝：{str(exc)[:60]}")

    print("\n[3] query_car_info")
    info = query_car_info("183")                 # 故意不写前导零
    print(f"    '183' → {info['car_id']} {info['car_name']} / {info['brand']} / "
          f"{info['body_type']} · {info['sub_type']}")
    for bad in ["unknown", "0001", "9999"]:
        try:
            query_car_info(bad)
            print(f"    {bad} → 未报错（异常）")
        except ValueError as exc:
            print(f"    {bad} → 正确拒绝：{str(exc)[:70]}")

    print("\n[4] save_car_record + search_records")
    before = search_records(limit=1)["count"]
    s1 = save_car_record(info["car_id"], note="自测写入", confidence=0.91)
    print(f"    保存 → action={s1['action']} record_id={s1['record_id']} 总记录数={s1['total_records']}")
    s2 = save_car_record(info["car_id"], note="自测更新", confidence=0.91)
    print(f"    重复保存 → action={s2['action']}（应为 updated，不产生重复记录）")
    found = search_records(keyword="自测")
    print(f"    检索关键词'自测' → 命中 {found['returned']} 条，备注={found['items'][0]['note']!r}")
    print(f"    检索不存在的关键词 → 命中 {search_records(keyword='绝不存在')['returned']} 条（应为 0，且不报错）")
    assert search_records(keyword="自测")["returned"] == 1

    print("\n[5] delete_car_record + get_stats")
    st = get_stats()
    print(f"    统计 → 总记录 {st['total']} 条，带备注 {st['with_note']} 条，"
          f"覆盖车身类型 {st['body_types']} 种，按类型分布 {st['by_body_type']}")
    d = delete_car_record(info["car_id"])
    print(f"    删除 → {d['car_id']} {d['car_name']}，剩余 {d['remaining']} 条")
    try:
        delete_car_record(info["car_id"])
        print("    重复删除 → 未报错（异常）")
    except ValueError as exc:
        print(f"    重复删除 → 正确拒绝：{str(exc)[:50]}")

    print("\n[6] 执行器：参数校验与白名单")
    cases = [
        ({"id": "x1", "function": {"name": "no_such_tool", "arguments": "{}"}}, "白名单拦截"),
        ({"id": "x2", "function": {"name": "query_car_info", "arguments": "{不是JSON"}}, "JSON 解析失败"),
        ({"id": "x3", "function": {"name": "query_car_info", "arguments": "{}"}}, "缺少必填参数"),
        ({"id": "x4", "function": {"name": "query_car_info", "arguments": '{"car_id":"0183","extra":1}'}}, "多余参数"),
    ]
    for call, label in cases:
        res = execute_tool(call)
        print(f"    {label} → ok={res['ok']}  {res.get('error', '')[:60]}")
        assert res["ok"] is False

    print("\n" + "=" * 70)
    print("自测全部通过。当前记录数：", search_records(limit=1)["count"], f"（自测前 {before}）")
    print("提示：如需清空自测数据，删除 data/app.db 即可。")


if __name__ == "__main__":
    _selftest()
