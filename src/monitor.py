"""训练进度监控程序。

启动后浏览器打开 http://127.0.0.1:8000 即可实时看到训练进度。
它会读取 train.py 每轮写出的 `logs/<name>_progress.json` 和 `<name>_history.csv`，
自动刷新进度条、逐轮曲线和关键指标，无需手动刷新页面。

用法：
    python src/monitor.py [--port 8000] [--logs 日志目录]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOGS_DIR, PROJECT_ROOT  # noqa: E402

PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练进度监控</title>
<style>
:root{--bg:#0f1420;--card:#1a2130;--line:#2a3345;--txt:#e8ecf4;--dim:#8b96ab;--acc:#4ade80;--warn:#facc15}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font-family:"Microsoft YaHei",system-ui,sans-serif;padding:24px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card .k{color:var(--dim);font-size:12px}.card .v{font-size:26px;font-weight:700;margin-top:6px}
.card .v small{font-size:13px;color:var(--dim);font-weight:400}
.bar-wrap{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:22px}
.bar-track{background:#0a0e18;border-radius:8px;height:18px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#3b82f6,#4ade80);border-radius:8px;transition:width .5s;width:0}
.legend{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;margin-top:8px}
svg{width:100%;height:230px;background:var(--card);border:1px solid var(--line);border-radius:12px}
.axis text{fill:var(--dim);font-size:11px}.grid-line{stroke:var(--line);stroke-width:1}
.line-f1{stroke:#4ade80;stroke-width:2;fill:none}.line-loss{stroke:#f87171;stroke-width:2;fill:none}
.dot{fill:#4ade80}
.idle{color:var(--warn);font-size:14px}
</style></head><body>
<h1>🚗 车型识别模型 · 训练进度监控</h1>
<div class="sub" id="sub">正在连接…</div>
<div class="grid">
  <div class="card"><div class="k">当前轮次 / 总轮数</div><div class="v" id="epoch">—</div></div>
  <div class="card"><div class="k">验证 Macro-F1（评分指标）</div><div class="v" id="f1">—</div></div>
  <div class="card"><div class="k">历史最佳 Macro-F1</div><div class="v" id="best">—</div></div>
  <div class="card"><div class="k">训练准确率</div><div class="v" id="acc">—</div></div>
  <div class="card"><div class="k">训练 loss</div><div class="v" id="loss">—</div></div>
  <div class="card"><div class="k">预计剩余时间</div><div class="v" id="eta">—</div></div>
</div>
<div class="bar-wrap">
  <div class="bar-track"><div class="bar-fill" id="fill"></div></div>
  <div class="legend"><span id="elapsed">已用时 —</span><span id="pct">0%</span></div>
</div>
<svg id="chart" viewBox="0 0 660 230" preserveAspectRatio="none"></svg>
<script>
let timer=null;
function fmt(s){s=Math.round(s);let m=Math.floor(s/60),ss=s%60;if(m>=60){let h=Math.floor(m/60);return h+'小时'+(m%60)+'分'}return m+'分'+ss+'秒'}
function draw(hist){
  const svg=document.getElementById('chart');if(!hist.length){return}
  const W=660,H=230,L=44,R=14,T=18,B=30;
  const n=hist.length,xs=i=>L+(W-L-R)*(i/(n-1));
  const f1s=hist.map(r=>r.val_macro_f1),ls=hist.map(r=>r.val_loss);
  const lo=Math.min(...ls),hi=Math.max(...ls);
  let s=`<g class="axis">`;
  for(let i=0;i<=4;i++){let y=T+(H-T-B)*i/4;s+=`<line class="grid-line" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/>`;
    s+=`<text x="${L-6}" y="${y+4}" text-anchor="end">${(hi-(hi-lo)*i/4).toFixed(2)}</text>`}
  s+=`</g><g class="axis">`;
  hist.forEach((r,i)=>{let x=xs(i);s+=`<text x="${x}" y="${H-10}" text-anchor="middle">${r.epoch}</text>`});
  s+=`</g>`;
  // loss 曲线（左轴归一化到图）
  let path='',p2='';
  hist.forEach((r,i)=>{let x=xs(i),y=T+(H-T-B)*((r.val_loss-lo)/(hi-lo||1));path+=(i?'L':'M')+x.toFixed(1)+','+y.toFixed(1)+' '});
  hist.forEach((r,i)=>{let x=xs(i),y=T+(H-T-B)*((r.val_macro_f1-0)/(1.0));p2+=(i?'L':'M')+x.toFixed(1)+','+y.toFixed(1)+' '});
  s+=`<path class="line-loss" d="${path}"/><path class="line-f1" d="${p2}"/>`;
  // F1 最后一个点
  let lx=xs(n-1),ly=T+(H-T-B)*((hist[n-1].val_macro_f1-0)/1.0);
  s+=`<circle class="dot" cx="${lx}" cy="${ly}" r="4"/>`;
  s+=`<text x="${W-R}" y="${T-4}" text-anchor="end" fill="#4ade80" font-size="11">Macro-F1=${hist[n-1].val_macro_f1.toFixed(4)}</text>`;
  svg.innerHTML=s;
}
async function tick(){
  try{
    const r=await fetch('/progress.json?t='+Date.now());
    if(!r.ok){document.getElementById('sub').textContent='未找到训练任务，等待训练启动…';return}
    const p=await r.json();
    if(!p){document.getElementById('sub').textContent='暂无进度数据';return}
    document.getElementById('sub').textContent=(p.finished?'✅ 训练已完成 · ':'⏳ 训练进行中 · ')+p.name+' · '+p.updated_at;
    document.getElementById('epoch').textContent=p.epoch+' / '+p.total_epochs;
    document.getElementById('f1').textContent=(p.val_macro_f1??'-').toFixed(4);
    document.getElementById('best').textContent=(p.best_val_macro_f1??'-').toFixed(4);
    document.getElementById('acc').innerHTML=(p.train_acc*100).toFixed(1)+'<small>%</small>';
    document.getElementById('loss').textContent=p.train_loss.toFixed(4);
    document.getElementById('eta').textContent=p.finished?'已完成':fmt(p.eta_seconds);
    let pct=p.finished?100:Math.round(p.epoch/p.total_epochs*100);
    document.getElementById('fill').style.width=pct+'%';
    document.getElementById('pct').textContent=pct+'%';
    document.getElementById('elapsed').textContent='已用时 '+fmt(p.elapsed_seconds)+' · 最佳出现在第 '+p.best_epoch+' 轮';
    const h=await (await fetch('/history.csv?t='+Date.now())).json();
    draw(h);
  }catch(e){document.getElementById('sub').textContent='连接失败：'+e.message}
}
tick();timer=setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _find_progress(self):
        files = sorted(glob.glob(os.path.join(self.logs_dir, "*_progress.json")),
                       key=os.path.getmtime, reverse=True)
        return files[0] if files else None

    def _find_history(self):
        files = sorted(glob.glob(os.path.join(self.logs_dir, "*_history.csv")),
                       key=os.path.getmtime, reverse=True)
        return files[0] if files else None

    def do_GET(self):
        if self.path.startswith("/progress.json"):
            p = self._find_progress()
            if not p:
                self.send_response(404); self.end_headers(); return
            with open(p, encoding="utf-8") as f:
                self._json(json.load(f))
        elif self.path.startswith("/history.csv"):
            h = self._find_history()
            if not h:
                self._json([]); return
            import csv
            rows = list(csv.DictReader(open(h, encoding="utf-8")))
            out = []
            for r in rows:
                out.append({k: (float(r[k]) if k not in ("epoch",) else int(r[k]))
                            for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_macro_f1")})
            self._json(out)
        else:
            data = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--logs", default=LOGS_DIR)
    a = p.parse_args()
    Handler.logs_dir = a.logs
    server = HTTPServer(("127.0.0.1", a.port), Handler)
    print(f"监控服务已启动：http://127.0.0.1:{a.port}")
    print(f"监控目录：{a.logs}（每 3 秒自动刷新）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
