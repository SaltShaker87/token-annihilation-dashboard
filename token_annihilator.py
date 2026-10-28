#!/usr/bin/env python3
"""Token Annihilation Dashboard: transparent Ollama token-counting proxy + tiny dashboard. Stdlib only."""
import http.client, http.server, json, os, re, sqlite3, subprocess, sys, threading, time, html
from datetime import datetime
from socketserver import ThreadingMixIn

HOME = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HOME, "stats.db")
PRICES = os.path.join(HOME, "prices.json")
UP_HOST, UP_PORT = "127.0.0.1", 11434
PROXY_BIND, PROXY_PORT = os.environ.get("PROXY_BIND", "0.0.0.0"), int(os.environ.get("PROXY_PORT", 11435))
def _tailscale_ip():
    try:
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        return ip[0] if ip else "127.0.0.1"
    except Exception:
        return "127.0.0.1"
UP_HOST = os.environ.get("UPSTREAM_HOST", UP_HOST); UP_PORT = int(os.environ.get("UPSTREAM_PORT", UP_PORT))
DASH_BIND, DASH_PORT = os.environ.get("DASH_BIND") or _tailscale_ip(), int(os.environ.get("DASH_PORT", 11480))
TRACK = ("/api/generate", "/api/chat", "/v1/chat/completions", "/v1/completions", "/api/embed", "/api/embeddings", "/v1/embeddings")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection"}
lock = threading.Lock()

def db():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, ts REAL, model TEXT, endpoint TEXT,
      input_tokens INTEGER, output_tokens INTEGER, duration_s REAL, eval_s REAL, status INTEGER, source TEXT, dedupe TEXT UNIQUE)""")
    return c

def log_row(**r):
    try:
        with lock:
            c = db(); c.execute("INSERT OR IGNORE INTO requests(ts,model,endpoint,input_tokens,output_tokens,duration_s,eval_s,status,source,dedupe) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (r.get("ts"), r.get("model"), r.get("endpoint"), r.get("inp"), r.get("out"), r.get("dur"), r.get("eval_s"), r.get("status"), r.get("source"), r.get("dedupe"))); c.commit(); c.close()
    except Exception as e:
        print("log error:", e, file=sys.stderr)

def parse_usage(endpoint, req_body, tail):
    model = inp = out = eval_s = None
    try: model = json.loads(req_body or b"{}").get("model")
    except Exception: pass
    text = tail.decode("utf-8", "replace")
    cands = [text] + [l[5:].strip() if l.startswith("data:") else l for l in reversed(text.splitlines())]
    for cand in cands:
        try: o = json.loads(cand)
        except Exception: continue
        if not isinstance(o, dict): continue
        model = o.get("model") or model
        if "prompt_eval_count" in o or "eval_count" in o:
            inp, out = o.get("prompt_eval_count"), o.get("eval_count")
            if o.get("eval_duration"): eval_s = o["eval_duration"] / 1e9
            break
        u = o.get("usage")
        if isinstance(u, dict):
            inp, out = u.get("prompt_tokens"), u.get("completion_tokens", 0); break
    return model, inp, out, eval_s

class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def handle_any(self):
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        up = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            up.request(self.command, self.path, body=body, headers=hdrs)
            r = up.getresponse()
        except Exception as e:
            self.send_error(502, "upstream error: %s" % e); return
        path = self.path.split("?")[0]
        track = self.command == "POST" and path in TRACK
        self.send_response(r.status, r.reason)
        chunked = r.getheader("Content-Length") is None and self.command != "HEAD" and r.status not in (204, 304)
        for k, v in r.getheaders():
            if k.lower() not in HOP: self.send_header(k, v)
        if chunked: self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        tail = bytearray()
        try:
            while True:
                chunk = r.read1(65536) if self.command != "HEAD" else b""
                if not chunk: break
                if chunked: self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else: self.wfile.write(chunk)
                self.wfile.flush()
                if track:
                    tail += chunk
                    if len(tail) > 4_000_000: del tail[:-262144]
            if chunked: self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        finally: up.close()
        if track:
            try:
                model, inp, out, eval_s = parse_usage(path, body, bytes(tail))
                log_row(ts=t0, model=model, endpoint=path, inp=inp, out=out, dur=time.time() - t0, eval_s=eval_s, status=r.status, source="proxy")
            except Exception as e: print("parse error:", e, file=sys.stderr)
    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_OPTIONS = do_PATCH = handle_any

GIN = re.compile(r'\[GIN\] (\d{4}/\d\d/\d\d - \d\d:\d\d:\d\d) \| (\d+) \|\s+([\d.]+)(ns|µs|us|ms|s|m)?\S* \|\s+(\S+) \| (\w+)\s+"([^"]+)"')
UNIT = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, None: 1}
def backfill(before=None):
    out = subprocess.run(["journalctl", "-u", "ollama", "--no-pager", "-o", "cat"], capture_output=True, text=True, errors="replace").stdout
    n = 0
    for line in out.splitlines():
        m = GIN.search(line)
        if not m: continue
        ts_s, status, dur, unit, ip, meth, path = m.groups()
        if meth != "POST" or path.split("?")[0] not in TRACK: continue
        ts = datetime.strptime(ts_s, "%Y/%m/%d - %H:%M:%S").timestamp()
        if before and ts >= before: continue
        log_row(ts=ts, model=None, endpoint=path, inp=None, out=None, dur=float(dur) * UNIT.get(unit, 1), status=int(status), source="backfill", dedupe=line[-160:]); n += 1
    print("backfill rows seen:", n)

def fmt(n): return "{:,}".format(int(n or 0))
def money(x): return "${:,.4f}".format(x) if x < 1 else "${:,.2f}".format(x)

def calendar(c):
    from datetime import date, timedelta
    rows = c.execute("SELECT date(ts,'unixepoch','localtime'), SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)) FROM requests WHERE source='proxy' GROUP BY 1").fetchall()
    per = {d: t or 0 for d, t in rows}
    today = date.today(); end = today + timedelta(days=(5 - today.weekday()) % 7)  # through Saturday
    start = end - timedelta(days=52 * 7 - 1 + 7); start -= timedelta(days=(start.weekday() + 1) % 7)  # Sunday
    mx = max(per.values() or [0]) or 0
    shades = ["#1b1f27", "#2a4d0a", "#3f7a00", "#5a9e00", "#76B900"]
    cells, months, d, col, lastm = [], [], start, 0, None
    while d <= end:
        r = (d.weekday() + 1) % 7
        if r == 0 and d.month != lastm: months.append(f"<text x={30+col*13} y=10 class=ct>{d:%b}</text>"); lastm = d.month
        if d <= today:
            t = per.get(d.isoformat(), 0)
            lvl = 0 if not t or not mx else min(4, 1 + int(4 * t / mx - 1e-9))
            cells.append(f"<rect x={30+col*13} y={16+r*13} width=11 height=11 rx=2 fill='{shades[lvl]}' data-t='{d:%a %b %d, %Y}: {fmt(t)} tokens annihilated'/>")
        if r == 6: col += 1
        d += timedelta(days=1)
    labels = "".join(f"<text x=0 y={25+r*13} class=ct>{n}</text>" for r, n in ((1, "Mon"), (3, "Wed"), (5, "Fri")))
    legend = "".join(f"<span class=sw style='background:{s}'></span>" for s in shades)
    return (f"<div class=cal><svg viewBox='0 0 {40+col*13} 110' width='100%'>{months_join(months)}{labels}{''.join(cells)}</svg>"
            f"<div class=legend>Mild <span>{legend}</span> Apocalyptic</div><div id=tip></div></div>")
def months_join(m): return "".join(m)

def page():
    c = db(); q = lambda s, *a: c.execute(s, a).fetchall()
    P = "source='proxy'"
    tin, tout, nreq = q(f"SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COUNT(*) FROM requests WHERE {P}")[0]
    tps = q(f"SELECT SUM(output_tokens)/SUM(eval_s) FROM requests WHERE {P} AND eval_s>0 AND output_tokens>0")[0][0]
    days = q(f"SELECT date(ts,'unixepoch','localtime') d, SUM(COALESCE(input_tokens,0)), SUM(COALESCE(output_tokens,0)) FROM requests WHERE {P} GROUP BY d ORDER BY d DESC LIMIT 30")
    models = q(f"SELECT COALESCE(model,'?'), COUNT(*), SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)) t FROM requests WHERE {P} GROUP BY 1 ORDER BY t DESC LIMIT 10")
    cal = calendar(c); c.close()
    try: prices = json.load(open(PRICES))["models"]
    except Exception: prices = []
    mx = max([a + b for _, a, b in days] or [1]) or 1
    dayrows = "".join(f"<tr><td>{d}</td><td class=bar><div style='width:{100*(a+b)/mx:.1f}%'></div></td><td>{fmt(a)}</td><td>{fmt(b)}</td></tr>" for d, a, b in days) or "<tr><td colspan=4 class=muted>The silence before the storm. Point a client at port %d and let the carnage begin.</td></tr>" % PROXY_PORT
    modrows = "".join(f"<tr><td>{html.escape(m)}</td><td>{fmt(n)}</td><td>{fmt(t)}</td></tr>" for m, n, t in models) or "<tr><td colspan=3 class=muted>No champions yet. The arena awaits.</td></tr>"
    sav = ""
    for p in prices:
        if p.get("input_per_mtok") is None or p.get("output_per_mtok") is None:
            sav += f"<div class=card><div class=lbl>{html.escape(p['label'])}</div><div class=big>n/a</div><div class=muted>price unverified</div></div>"; continue
        v = tin / 1e6 * p["input_per_mtok"] + tout / 1e6 * p["output_per_mtok"]
        sav += f"<div class=card><div class=lbl>Kept from {html.escape(p['label'])}</div><div class=big>{money(v)}</div><div class=muted>${p['input_per_mtok']}/${p['output_per_mtok']} per M in/out, now crying</div></div>"
    T = "DGX Token Annihilator Dashboard"
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=60><title>{T}</title><style>
body{{font-family:system-ui,sans-serif;background:#0b0d10;color:#e6e6e6;padding:24px;max-width:1000px;margin:auto}}
h1{{font-weight:800;margin:0 0 4px;color:#76B900;letter-spacing:.02em}} h2{{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:#9aa3b5;margin:28px 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}} .card{{background:#14171d;border:1px solid #232833;border-radius:12px;padding:14px}}
.big{{font-size:26px;font-weight:700;color:#76B900}} .lbl{{color:#9aa3b5;font-size:13px}} .muted{{color:#6b7285;font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td{{padding:6px 8px;border-bottom:1px solid #1f232d}} .bar div{{height:10px;background:linear-gradient(90deg,#3f7a00,#76B900);border-radius:5px}} .bar{{width:45%}}
.cal{{background:#14171d;border:1px solid #232833;border-radius:12px;padding:14px;position:relative}} .ct{{fill:#6b7285;font-size:9px}} rect:hover{{stroke:#e6e6e6;stroke-width:1}}
.legend{{text-align:right;font-size:12px;color:#6b7285;margin-top:6px}} .sw{{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 2px;vertical-align:middle}}
#tip{{position:fixed;display:none;background:#000;border:1px solid #76B900;color:#e6e6e6;font-size:12px;padding:4px 8px;border-radius:6px;pointer-events:none}}
</style></head><body><h1>{T}</h1><div class=muted>Every token counted by the proxy on port {PROXY_PORT} is a token the cloud will never bill you for. Last carnage report: {datetime.now():%b %d %H:%M}.</div>
<h2>Body count</h2><div class=grid><div class=card><div class=lbl>Tokens annihilated</div><div class=big>{fmt(tin+tout)}</div></div><div class=card><div class=lbl>Prompts devoured (input)</div><div class=big>{fmt(tin)}</div></div><div class=card><div class=lbl>Words unleashed (output)</div><div class=big>{fmt(tout)}</div></div><div class=card><div class=lbl>Requests vanquished</div><div class=big>{fmt(nreq)}</div></div><div class=card><div class=lbl>Annihilation speed (out tok/s)</div><div class=big>{f"{tps:.1f}" if tps else "–"}</div></div></div>
<h2>Cloud bills obliterated (at list price)</h2><div class=grid>{sav}</div>
<h2>Year of destruction</h2>{cal}
<h2>Daily devastation log</h2><table><tr class=muted><td>Day</td><td></td><td>In</td><td>Out</td></tr>{dayrows}</table>
<h2>Hall of fame: hungriest models</h2><table><tr class=muted><td>Model</td><td>Requests</td><td>Tokens</td></tr>{modrows}</table>
<p class=muted>No GPUs were harmed. Several cloud invoices were.</p>
<script>var t=document.getElementById('tip');document.querySelectorAll('rect[data-t]').forEach(function(r){{r.onmousemove=function(e){{t.textContent=r.dataset.t;t.style.display='block';t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px'}};r.onmouseleave=function(){{t.style.display='none'}}}});</script>
</body></html>"""

class Dash(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        try: b = page().encode(); code = 200
        except Exception as e: b = ("error: %s" % e).encode(); code = 500
        self.send_response(code); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

class TS(ThreadingMixIn, http.server.HTTPServer): daemon_threads = True; allow_reuse_address = True

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Token Annihilation Dashboard")
    ap.add_argument("--proxy-bind", default=PROXY_BIND); ap.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    ap.add_argument("--dash-bind", default=DASH_BIND); ap.add_argument("--dash-port", type=int, default=DASH_PORT)
    ap.add_argument("--backfill", nargs="?", const="", default=None, metavar="UNIX_TS_CUTOFF")
    a = ap.parse_args()
    PROXY_BIND, PROXY_PORT, DASH_BIND, DASH_PORT = a.proxy_bind, a.proxy_port, a.dash_bind, a.dash_port
    db().close()
    if a.backfill is not None:
        backfill(float(a.backfill) if a.backfill else None); sys.exit()
    threading.Thread(target=TS((DASH_BIND, DASH_PORT), Dash).serve_forever, daemon=True).start()
    print(f"proxy {PROXY_BIND}:{PROXY_PORT} -> {UP_HOST}:{UP_PORT}; dashboard http://{DASH_BIND}:{DASH_PORT}", flush=True)
    TS((PROXY_BIND, PROXY_PORT), Proxy).serve_forever()
