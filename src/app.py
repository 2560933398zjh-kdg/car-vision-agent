"""识车笔记 Web 服务（任务书任务三）。

后端只做三件事，业务逻辑全部复用任务二的 agent / tools，不重复实现：
    1. 提供静态页面（web/ 目录）与图片上传；
    2. 管理会话：一个 session_id 对应一个 agent.Session，多轮上下文因此得以保留；
    3. 把工具调用记录、识车笔记、统计以 JSON 形式暴露给前端。

并发处理：会话各自持一把锁，允许不同用户同时对话；
视觉推理的显存保护在 tools.py 内部完成（_INFER_LOCK 串行化），此处不重复加锁。

API Key 只从环境变量读取，本文件不包含任何密钥。

用法：
    python src/app.py                  # 启动服务，默认 http://127.0.0.1:5000
    python src/app.py --port 8000      # 换端口
    python src/app.py --selftest       # 跑系统测试用例，不启动服务
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tools
from agent import AgentError, Session, run_agent
from common import IMG_EXTS, PROJECT_ROOT, load_agent_config, load_class_ids

CONFIG = load_agent_config()
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.json.ensure_ascii = False

_SESSIONS: dict[str, dict] = {}
_REGISTRY_LOCK = threading.Lock()


# --------------------------------------------------------------- 会话

def get_session(sid: str) -> dict:
    with _REGISTRY_LOCK:
        entry = _SESSIONS.get(sid)
    if entry is None:
        raise KeyError(f"会话 {sid} 不存在，请刷新页面重新开始")
    return entry


@app.post("/api/session")
def api_new_session():
    sid = uuid.uuid4().hex[:8]
    with _REGISTRY_LOCK:
        _SESSIONS[sid] = {"session": Session(sid), "lock": threading.Lock(),
                          "created": datetime.now().isoformat(timespec="seconds")}
    return jsonify({"ok": True, "session_id": sid})


@app.post("/api/session/<sid>/reset")
def api_reset_session(sid):
    entry = get_session(sid)
    entry["session"].reset()
    tools.TOOL_LOGS.clear()
    return jsonify({"ok": True, "session_id": sid})


@app.get("/api/session/<sid>/log")
def api_session_log(sid):
    entry = get_session(sid)
    session: Session = entry["session"]
    return jsonify({"ok": True, "session_id": sid, "rounds": session.rounds,
                    "turns": session.transcript,
                    "tool_calls": sum(len(t["tools"]) for t in session.transcript)})


# --------------------------------------------------------------- 健康检查与元信息

@app.get("/api/health")
def api_health():
    key_env = CONFIG["api_key_env"]
    checks = {
        "api_key": bool(os.environ.get(key_env, "").strip()),
        "model_file": os.path.isfile(CONFIG["vision"]["checkpoint"]),
        "database_dir": os.path.isdir(os.path.dirname(CONFIG["storage"]["db_path"])),
        "classes_txt": os.path.isfile(os.path.join(PROJECT_ROOT, "classes.txt")),
    }
    ok = checks["api_key"] and checks["model_file"]
    missing = [k for k, v in checks.items() if not v]
    return jsonify({
        "ok": ok, "checks": checks, "missing": missing,
        "model": CONFIG["model"], "key_env": key_env,
        "vision_model": os.path.relpath(CONFIG["vision"]["checkpoint"], PROJECT_ROOT),
        "hint": "" if ok else f"缺少：{', '.join(missing)}；请先设置环境变量 {key_env}",
    })


@app.get("/api/classes")
def api_classes():
    ids = load_class_ids()
    names = tools.load_name_map()
    items = [{"car_id": cid, "car_name": names.get(cid, cid)} for cid in ids]
    return jsonify({"ok": True, "count": len(items), "items": items})


# --------------------------------------------------------------- 图片上传

@app.post("/api/upload")
def api_upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "没有收到文件，请重新选择图片"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in IMG_EXTS:
        return jsonify({"ok": False,
                        "error": f"不支持的格式 {ext or '（无扩展名）'}，请上传 "
                                 f"{'、'.join(e.lstrip('.') for e in IMG_EXTS)}"}), 400

    # 用 uuid 重命名，避免原始文件名带来的路径穿越与编码问题
    filename = f"{uuid.uuid4().hex}{ext}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()                       # 真正解码一次，确认是图片而不是改了后缀的文件
            size = im.size
    except Exception:
        # 清理失败不应影响给用户的答复：删不掉也照样返回「不是有效图片」
        try:
            os.remove(path)
        except OSError:
            pass
        return jsonify({"ok": False, "error": "文件内容不是有效的图片，请重新上传"}), 400

    return jsonify({"ok": True, "image_path": path, "url": f"/uploads/{filename}",
                    "filename": file.filename, "bytes": os.path.getsize(path),
                    "size": f"{size[0]}×{size[1]}"})


@app.get("/uploads/<path:name>")
def api_uploaded_file(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.errorhandler(413)
def too_large(_err):
    return jsonify({"ok": False,
                    "error": f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限，请压缩后再上传"}), 413


# --------------------------------------------------------------- 对话

@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    sid = (payload.get("session_id") or "").strip()
    message = (payload.get("message") or "").strip()
    image_path = payload.get("image_path") or None

    if not sid:
        return jsonify({"ok": False, "error": "缺少 session_id"}), 400
    if not message and not image_path:
        return jsonify({"ok": False, "error": "请输入内容，或先上传一张图片"}), 400

    try:
        entry = get_session(sid)
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404

    if not os.environ.get(CONFIG["api_key_env"], "").strip():
        return jsonify({"ok": False,
                        "error": f"服务端未配置环境变量 {CONFIG['api_key_env']}，无法调用大模型"}), 503

    session: Session = entry["session"]
    with entry["lock"]:                       # 同一会话串行，避免消息历史被并发写乱
        try:
            answer = run_agent(session, message or "这是什么车？", image_path)
        except AgentError as exc:
            return jsonify({"ok": False, "error": str(exc), "kind": "agent"}), 502
        except Exception as exc:              # noqa: BLE001
            return jsonify({"ok": False,
                            "error": f"服务内部错误：{exc.__class__.__name__}", "kind": "server"}), 500

    turn = session.transcript[-1] if session.transcript else {"tools": []}
    return jsonify({"ok": True, "answer": answer, "round": session.rounds,
                    "tool_calls": turn["tools"], "image": image_path,
                    "records": tools.search_records(limit=100)["items"],
                    "stats": tools.get_stats()})


# --------------------------------------------------------------- 识车笔记

@app.get("/api/records")
def api_records():
    keyword = request.args.get("keyword", "")
    body_type = request.args.get("body_type", "")
    result = tools.search_records(keyword=keyword, limit=1000)
    items = result["items"]
    if body_type:
        items = [r for r in items if r["body_type"] == body_type]
    return jsonify({"ok": True, "count": result["count"], "returned": len(items),
                    "items": items, "stats": tools.get_stats()})


@app.patch("/api/records/<car_id>")
def api_update_note(car_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "note" not in payload:
        return jsonify({"ok": False, "error": "请求体必须是 JSON 且包含 note 字段"}), 400
    try:
        result = tools.update_record_note(car_id, str(payload["note"]))
    except ValueError as exc:
        # 记录不存在属于「找不到」，参数非法属于「请求错误」，分开返回
        code = 404 if "没有" in str(exc) else 400
        return jsonify({"ok": False, "error": str(exc)}), code
    return jsonify({"ok": True, "record": result, "stats": tools.get_stats()})


@app.delete("/api/records/<car_id>")
def api_delete_record(car_id):
    try:
        result = tools.delete_car_record(car_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    return jsonify({"ok": True, "deleted": result, "stats": tools.get_stats()})


@app.get("/api/stats")
def api_stats():
    return jsonify({"ok": True, "stats": tools.get_stats()})


@app.get("/api/records/export")
def api_export():
    fmt = request.args.get("format", "csv").lower()
    items = tools.search_records(limit=1000)["items"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        buf = io.BytesIO(json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"))
        return send_file(buf, mimetype="application/json", as_attachment=True,
                         download_name=f"识车笔记_{stamp}.json")

    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["car_id", "car_name", "brand", "body_type", "sub_type",
                     "confidence", "note", "created_at", "updated_at"])
    for row in items:
        writer.writerow([row["car_id"], row["car_name"], row["brand"], row["body_type"],
                         row["sub_type"], row["confidence"], row["note"],
                         row["created_at"], row["updated_at"]])
    buf = io.BytesIO(("\ufeff" + text.getvalue()).encode("utf-8"))   # 带 BOM，Excel 打开不乱码
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=f"识车笔记_{stamp}.csv")


# --------------------------------------------------------------- 页面与错误

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.errorhandler(404)
def not_found(_err):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": f"接口不存在：{request.path}"}), 404
    return send_from_directory(WEB_DIR, "index.html")


@app.errorhandler(500)
def server_error(_err):
    return jsonify({"ok": False, "error": "服务内部错误，请查看服务端日志"}), 500


# --------------------------------------------------------------- 系统测试

def selftest() -> int:
    """系统测试：用 Flask 测试客户端跑一遍关键路径，输出通过/不通过。"""
    print("=" * 70)
    print("识车笔记 Web 服务 · 系统测试")
    print("=" * 70)
    has_key = bool(os.environ.get(CONFIG["api_key_env"], "").strip())
    results: list[tuple[str, bool, str]] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        results.append((label, bool(cond), detail))
        print(f"  [{'通过' if cond else '未通过'}] {label}" + (f"    （{detail}）" if detail else ""))

    client = app.test_client()

    print("\n[1] 静态页面与健康检查")
    r = client.get("/")
    check("首页可访问", r.status_code == 200 and "识车笔记" in r.get_data(as_text=True),
          f"HTTP {r.status_code}")
    r = client.get("/api/health")
    body = r.get_json()
    check("健康检查返回结构", "checks" in body and "ok" in body,
          f"API Key={'已配置' if body['checks']['api_key'] else '未配置'}，"
          f"模型文件={'存在' if body['checks']['model_file'] else '缺失'}")
    r = client.get("/api/classes")
    check("可获取 50 类清单", r.get_json()["count"] == 50, f"{r.get_json()['count']} 类")

    print("\n[2] 会话管理")
    r = client.post("/api/session")
    sid = r.get_json()["session_id"]
    check("新建会话", r.status_code == 200 and len(sid) == 8, f"session_id={sid}")
    check("重置会话", client.post(f"/api/session/{sid}/reset").status_code == 200)
    check("会话不存在时报错", client.post("/api/chat", json={"session_id": "nope",
                                                        "message": "hi"}).status_code == 404)

    print("\n[3] 图片上传")
    sample = None
    for cid in ("0183", "0432", "0009"):
        folder = os.path.join(PROJECT_ROOT, "data", "val", cid)
        if os.path.isdir(folder):
            sample = os.path.join(folder, sorted(os.listdir(folder))[0])
            break
    with open(sample, "rb") as f:
        r = client.post("/api/upload", data={"file": (f, os.path.basename(sample))},
                        content_type="multipart/form-data")
    up = r.get_json()
    check("上传有效图片", r.status_code == 200 and up.get("ok"), f"{up.get('size', '')}")
    uploaded_path = up.get("image_path")
    check("上传后可访问图片", client.get(up["url"]).status_code == 200, up.get("url", ""))

    buf = io.BytesIO(b"this is not an image")
    r = client.post("/api/upload", data={"file": (buf, "fake.jpg")}, content_type="multipart/form-data")
    check("伪装成图片的文件被拒绝", r.status_code == 400, r.get_json().get("error", "")[:40])
    buf = io.BytesIO(b"hello")
    r = client.post("/api/upload", data={"file": (buf, "note.txt")}, content_type="multipart/form-data")
    check("非图片扩展名被拒绝", r.status_code == 400, r.get_json().get("error", "")[:40])
    check("不传文件被拒绝", client.post("/api/upload", data={}).status_code == 400)

    print("\n[4] 识车笔记接口")
    r = client.get("/api/records")
    check("记录列表", r.status_code == 200 and "items" in r.get_json(),
          f"当前 {r.get_json()['count']} 条")
    r = client.get("/api/stats")
    check("统计接口", r.status_code == 200 and "total" in r.get_json()["stats"],
          f"覆盖车身类型 {r.get_json()['stats']['body_types']} 种")
    r = client.get("/api/records/export?format=csv")
    check("导出 CSV", r.status_code == 200 and r.data.startswith("\ufeff".encode("utf-8")),
          f"{len(r.data)} 字节")
    r = client.get("/api/records/export?format=json")
    check("导出 JSON", r.status_code == 200)
    r = client.delete("/api/records/9999")
    check("删除不存在的记录返回 404", r.status_code == 404, r.get_json().get("error", "")[:40])
    r = client.patch("/api/records/9999", json={"note": "x"})
    check("修改不存在记录的备注返回 404", r.status_code == 404, r.get_json().get("error", "")[:40])
    r = client.patch("/api/records/9999", data="not json",
                     content_type="application/json")
    check("非法请求体被拒绝（不静默建记录）", r.status_code == 400)

    print("\n[5] 参数校验")
    check("缺少 session_id 被拒绝",
          client.post("/api/chat", json={"message": "hi"}).status_code == 400)
    check("空消息被拒绝",
          client.post("/api/chat", json={"session_id": sid, "message": ""}).status_code == 400)
    check("未知接口返回 JSON 错误",
          client.get("/api/nope").status_code == 404)

    print("\n[5.1] 检索覆盖车身类型")
    tools.save_car_record("0183", note="系统测试")
    check("按车身类型关键词可检索到记录",
          tools.search_records(keyword="跑车")["returned"] == 1,
          f"共 {tools.search_records(limit=1000)['count']} 条记录")
    tools.delete_car_record("0183")

    print("\n[6] 端到端对话（需要 API Key）")
    if not has_key:
        print(f"  [跳过] 未设置环境变量 {CONFIG['api_key_env']}，联网用例未执行")
    else:
        r = client.post("/api/session")
        sid2 = r.get_json()["session_id"]
        r = client.post("/api/chat", json={
            "session_id": sid2,
            "message": "请识别这张图片里的车型，介绍一下它的资料，然后加入我的识车笔记。",
            "image_path": uploaded_path})
        body = r.get_json()
        names = [t["name"] for t in body.get("tool_calls", [])] if body.get("ok") else []
        check("连续调用 ≥3 个工具",
              {"classify_car", "query_car_info", "save_car_record"} <= set(names),
              " → ".join(names) or body.get("error", ""))
        check("对话返回识车笔记快照", r.status_code == 200 and "records" in body,
              f"{len(body.get('records', []))} 条")

        r = client.post("/api/chat", json={"session_id": sid2, "message": "帮我把它备注成“接口测试”。"})
        body = r.get_json()
        names = [t["name"] for t in body.get("tool_calls", [])]
        check("多轮复用识别结果（未重新识别）",
              "save_car_record" in names and "classify_car" not in names,
              " → ".join(names) or body.get("error", ""))

        r = client.post("/api/chat", json={"session_id": sid2, "message": "把它从笔记里删掉。"})
        body = r.get_json()
        names = [t["name"] for t in body.get("tool_calls", [])]
        check("按用户要求删除记录", "delete_car_record" in names,
              " → ".join(names) or body.get("error", ""))

        r = client.get(f"/api/session/{sid2}/log")
        check("可获取会话工具调用记录", r.get_json()["tool_calls"] >= 4,
              f"{r.get_json()['tool_calls']} 次调用")

    print("\n" + "=" * 70)
    failed = [r for r in results if not r[1]]
    print(f"共 {len(results)} 项：通过 {len(results) - len(failed)} 项，未通过 {len(failed)} 项")
    if failed:
        for label, _ok, detail in failed:
            print(f"  未通过：{label}  {detail}")
    return 0 if not failed else 1


# --------------------------------------------------------------- 启动

def main() -> None:
    p = argparse.ArgumentParser(description="识车笔记 Web 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--selftest", action="store_true", help="只跑系统测试，不启动服务")
    p.add_argument("--no-debug", action="store_true", help="关闭调试模式")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(selftest())

    key_env = CONFIG["api_key_env"]
    print("=" * 70)
    print("识车笔记 Web 服务")
    print("=" * 70)
    print(f"  模型          {CONFIG['model']}")
    print(f"  视觉模型      {os.path.relpath(CONFIG['vision']['checkpoint'], PROJECT_ROOT)}")
    print(f"  业务数据库    {os.path.relpath(CONFIG['storage']['db_path'], PROJECT_ROOT)}")
    print(f"  API Key 环境变量 {key_env}："
          f"{'已配置' if os.environ.get(key_env, '').strip() else '【未配置，对话功能不可用】'}")
    print(f"  访问地址      http://{a.host}:{a.port}")
    print("=" * 70)

    app.run(host=a.host, port=a.port, debug=not a.no_debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
