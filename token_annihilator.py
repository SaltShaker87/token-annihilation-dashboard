#!/usr/bin/env python3
"""Token Annihilation Dashboard: transparent token-counting proxies for Ollama, vLLM, and llama.cpp, plus one dashboard. Stdlib only.

The local servers are never modified. Each proxy forwards to its own upstream and reads the token receipt on the way back.
"""
import http.client, http.server, json, os, re, sqlite3, subprocess, sys, threading, time, html, urllib.parse
from datetime import datetime, date, timedelta
from socketserver import ThreadingMixIn

HOME = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("STATS_DB") or os.path.join(HOME, "stats.db")
PRICES = os.path.join(HOME, "prices.json")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection"}
# Inference routes whose replies carry a token receipt. Counting is the same on every proxy,
# so a client aimed at the wrong engine still gets a tally when the body is one we understand.
TRACK = {
    "/api/generate", "/api/chat", "/api/embed", "/api/embeddings",
    "/v1/chat/completions", "/v1/chat/completions/batch",
    "/v1/completions", "/v1/embeddings", "/v1/responses", "/v1/messages",
    "/v1/rerank", "/v1/reranking", "/v2/rerank", "/rerank", "/reranking",
    "/score", "/v1/score", "/pooling", "/classify", "/v1/classify",
    "/completion", "/infill", "/embedding", "/embeddings",
}
ALIASES = {"ollama": "ollama", "vllm": "vllm", "llamacpp": "llamacpp", "llama.cpp": "llamacpp", "llama-cpp": "llamacpp", "llama": "llamacpp"}
lock = threading.RLock()
ENGINES = []

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _tailscale_ip():
    try:
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        return ip[0] if ip else "127.0.0.1"
    except Exception:
        return "127.0.0.1"

PROXY_BIND = os.environ.get("PROXY_BIND", "0.0.0.0")
PROXY_PORT = _env_int("PROXY_PORT", 11435)
VLLM_PROXY_PORT = _env_int("VLLM_PROXY_PORT", 8001)
LLAMACPP_PROXY_PORT = _env_int("LLAMACPP_PROXY_PORT", 8081)
DASH_BIND = os.environ.get("DASH_BIND") or _tailscale_ip()
DASH_PORT = _env_int("DASH_PORT", 11480)

class Engine:
    def __init__(self, id, label, port, up_host, up_port, health):
        self.id, self.label, self.port = id, label, port
        self.up_host, self.up_port, self.health = up_host, up_port, health
        self.bind, self.listening, self.error = PROXY_BIND, False, None

def engine_catalog():
    return [
        Engine("ollama", "Ollama", PROXY_PORT, os.environ.get("UPSTREAM_HOST", "127.0.0.1"), _env_int("UPSTREAM_PORT", 11434), "/api/tags"),
        Engine("vllm", "vLLM", VLLM_PROXY_PORT, os.environ.get("VLLM_UPSTREAM_HOST", "127.0.0.1"), _env_int("VLLM_UPSTREAM_PORT", 8000), "/health"),
        Engine("llamacpp", "llama.cpp", LLAMACPP_PROXY_PORT, os.environ.get("LLAMACPP_UPSTREAM_HOST", "127.0.0.1"), _env_int("LLAMACPP_UPSTREAM_PORT", 8080), "/health"),
    ]

def select_engines(spec):
    wanted, unknown = [], []
    for part in (spec or "").split(","):
        key = ALIASES.get(part.strip().lower())
        if not part.strip():
            continue
        if not key:
            unknown.append(part.strip())
        elif key not in wanted:
            wanted.append(key)
    if unknown:
        print("unknown engines ignored:", ", ".join(unknown), file=sys.stderr)
    by_id = {e.id: e for e in engine_catalog()}
    return [by_id[i] for i in wanted if i in by_id]

def db():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, ts REAL, model TEXT, endpoint TEXT,
      input_tokens INTEGER, output_tokens INTEGER, duration_s REAL, eval_s REAL, status INTEGER, source TEXT, dedupe TEXT UNIQUE,
      backend TEXT)""")
    with lock:
        cols = {r[1] for r in c.execute("PRAGMA table_info(requests)")}
        if "backend" not in cols:
            try:
                c.execute("ALTER TABLE requests ADD COLUMN backend TEXT")
            except sqlite3.OperationalError:
                pass
            c.execute("UPDATE requests SET backend='ollama' WHERE backend IS NULL")
            c.commit()
    return c

def log_row(**r):
    try:
        with lock:
            c = db()
            c.execute("""INSERT OR IGNORE INTO requests(ts,model,endpoint,input_tokens,output_tokens,duration_s,eval_s,status,source,dedupe,backend)
              VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
              (r.get("ts"), r.get("model"), r.get("endpoint"), r.get("inp"), r.get("out"), r.get("dur"), r.get("eval_s"), r.get("status"), r.get("source"), r.get("dedupe"), r.get("backend")))
            c.commit(); c.close()
    except Exception as e:
        print("log error:", e, file=sys.stderr)

def _as_int(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    n = int(v)
    return n if n >= 0 else None

def _opt_num(d, key):
    if not isinstance(d, dict) or key not in d or d[key] is None:
        return None
    return _as_int(d[key])

def _usage_pair(u):
    """OpenAI uses prompt/completion tokens. Anthropic and the Responses API use input/output tokens."""
    if not isinstance(u, dict):
        return None
    if "prompt_tokens" in u or "completion_tokens" in u:
        return _opt_num(u, "prompt_tokens"), _opt_num(u, "completion_tokens")
    if "input_tokens" in u or "output_tokens" in u:
        return _opt_num(u, "input_tokens"), _opt_num(u, "output_tokens")
    if "total_tokens" in u:
        return _opt_num(u, "total_tokens"), 0
    return None

def _eval_seconds(o):
    if not isinstance(o, dict):
        return None
    if isinstance(o.get("eval_duration"), (int, float)) and not isinstance(o.get("eval_duration"), bool) and o["eval_duration"] > 0:
        return o["eval_duration"] / 1e9
    timings = o.get("timings")
    if isinstance(timings, dict) and isinstance(timings.get("predicted_ms"), (int, float)) and not isinstance(timings.get("predicted_ms"), bool) and timings["predicted_ms"] > 0:
        return timings["predicted_ms"] / 1000.0
    metrics = o.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("generation_time_ms"), (int, float)) and not isinstance(metrics.get("generation_time_ms"), bool) and metrics["generation_time_ms"] > 0:
        return metrics["generation_time_ms"] / 1000.0
    return None

def _sum_counts(items):
    inp = out = 0
    saw_i = saw_o = False
    ev = 0.0
    saw_e = False
    model = None
    hit = False
    for item in items:
        if not isinstance(item, dict):
            continue
        m, i, ou, e, h = extract_counts(item)
        if not h:
            continue
        hit = True
        if m and not model:
            model = m
        if i is not None:
            inp += i; saw_i = True
        if ou is not None:
            out += ou; saw_o = True
        if e:
            ev += e; saw_e = True
    if not hit:
        return None, None, None, None, False
    return model, (inp if saw_i else None), (out if saw_o else None), (ev if saw_e else None), True

def extract_counts(o):
    """Return model, input tokens, output tokens, generation seconds, and whether a receipt was found."""
    if isinstance(o, list):
        return _sum_counts(o)
    if not isinstance(o, dict):
        return None, None, None, None, False
    model = o.get("model") if isinstance(o.get("model"), str) else None
    containers = [o]
    for key in ("response", "message"):
        inner = o.get(key)
        if isinstance(inner, dict):
            containers.append(inner)
    if "prompt_eval_count" in o or "eval_count" in o:
        return model, _opt_num(o, "prompt_eval_count"), _opt_num(o, "eval_count"), _eval_seconds(o), True
    for c in containers:
        pair = _usage_pair(c.get("usage"))
        if pair is not None:
            m = c.get("model") if isinstance(c.get("model"), str) else model
            ev = _eval_seconds(o) or _eval_seconds(c)
            return m, pair[0], pair[1], ev, True
    if "tokens_evaluated" in o or "tokens_predicted" in o:
        inp, out = _opt_num(o, "tokens_evaluated"), _opt_num(o, "tokens_predicted")
        timings = o.get("timings") if isinstance(o.get("timings"), dict) else {}
        if out is None:
            out = _opt_num(timings, "predicted_n")
        if inp is None and _opt_num(timings, "prompt_n") is not None:
            inp = _opt_num(timings, "prompt_n") + (_opt_num(timings, "cache_n") or 0)
        return model, inp, out, _eval_seconds(o), True
    timings = o.get("timings")
    if isinstance(timings, dict) and ("prompt_n" in timings or "predicted_n" in timings):
        prompt, pred = _opt_num(timings, "prompt_n"), _opt_num(timings, "predicted_n")
        if prompt is not None or pred is not None:
            cache = _opt_num(timings, "cache_n") or 0
            return model, (prompt or 0) + cache, pred, _eval_seconds(o), True
    for key in ("responses", "data"):
        inner = o.get(key)
        if isinstance(inner, list):
            m, i, ou, e, hit = _sum_counts(inner)
            if hit:
                return m or model, i, ou, e or _eval_seconds(o), True
    return model, None, None, _eval_seconds(o), False

def json_values(text):
    raw = text.strip()
    if raw:
        try:
            return [json.loads(raw)]
        except Exception:
            pass
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("data:"):
            s = s[5:].strip()
        if not s or s == "[DONE]" or s.startswith(":"):
            continue
        try:
            vals.append(json.loads(s))
        except Exception:
            continue
    if len(vals) > 530:
        vals = vals[:30] + vals[-500:]
    return vals

def parse_usage(endpoint, req_body, tail):
    req_model = None
    try:
        req = json.loads(req_body or b"{}")
        if isinstance(req, dict) and isinstance(req.get("model"), str):
            req_model = req["model"]
    except Exception:
        pass
    model = inp = out = eval_s = None
    for o in reversed(json_values(tail.decode("utf-8", "replace"))):
        m, i, ou, e, hit = extract_counts(o)
        if m and model is None:
            model = m
        if not hit:
            continue
        if i is not None and inp is None:
            inp = i
        if ou is not None and out is None:
            out = ou
        if e is not None and eval_s is None:
            eval_s = e
    return model or req_model, inp, out, eval_s

def _keep_tail(tail):
    """Remember the start of a huge stream (prompt receipt) and the end (final receipt)."""
    if len(tail) > 4_000_000:
        head = bytes(tail[:65536])
        del tail[:-1_000_000]
        if tail[:len(head)] != head:
            tail[:0] = head

class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def handle_any(self):
        eng = self.server.engine
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        up = http.client.HTTPConnection(eng.up_host, eng.up_port, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            up.request(self.command, self.path, body=body, headers=hdrs)
            r = up.getresponse()
        except Exception as e:
            self.send_error(502, "upstream error: %s" % e); return
        path = self.path.split("?")[0]
        if len(path) > 1:
            path = path.rstrip("/")
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
                    _keep_tail(tail)
            if chunked: self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        finally: up.close()
        if track:
            try:
                model, inp, out, eval_s = parse_usage(path, body, bytes(tail))
                log_row(ts=t0, model=model, endpoint=path, inp=inp, out=out, dur=time.time() - t0, eval_s=eval_s, status=r.status, source="proxy", backend=eng.id)
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
        ts_s, status, dur, unit, ip, meth, raw = m.groups()
        path = raw.split("?")[0]
        if len(path) > 1: path = path.rstrip("/")
        if meth != "POST" or path not in TRACK: continue
        ts = datetime.strptime(ts_s, "%Y/%m/%d - %H:%M:%S").timestamp()
        if before and ts >= before: continue
        log_row(ts=ts, model=None, endpoint=raw, inp=None, out=None, dur=float(dur) * UNIT.get(unit, 1), status=int(status), source="backfill", dedupe=line[-160:], backend="ollama"); n += 1
    print("backfill rows seen:", n)

def fmt(n): return "{:,}".format(int(n or 0))
def money(x): return "${:,.4f}".format(x) if x < 1 else "${:,.2f}".format(x)

def _where(backend):
    if backend:
        return "source='proxy' AND backend=?", (backend,)
    return "source='proxy'", ()

def calendar(c, where, args):
    rows = c.execute("SELECT date(ts,'unixepoch','localtime'), SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)) FROM requests WHERE %s GROUP BY 1" % where, args).fetchall()
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
    return (f"<div class=cal><svg viewBox='0 0 {40+col*13} 110' width='100%'>{''.join(months)}{labels}{''.join(cells)}</svg>"
            f"<div class=legend>Mild <span>{legend}</span> Apocalyptic</div><div id=tip></div></div>")

def upstream_status(engines):
    out = {}
    def one(e):
        conn = None
        try:
            conn = http.client.HTTPConnection(e.up_host, e.up_port, timeout=0.8)
            conn.request("GET", e.health, headers={"Connection": "close"})
            r = conn.getresponse(); r.read(256)
            out[e.id] = r.status
        except Exception:
            out[e.id] = None
        finally:
            if conn:
                try: conn.close()
                except Exception: pass
    threads = []
    for e in engines:
        t = threading.Thread(target=one, args=(e,), daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join(1.0)
    return out

def _engine_phrase(e, code):
    if not e.listening:
        why = " (%s)" % html.escape(e.error) if e.error else ""
        return "Proxy on port %d did not start%s." % (e.port, why)
    dest = html.escape("%s:%d" % (e.up_host, e.up_port))
    if code is None:
        return "Proxy on port %d is open. Nothing is answering at %s yet." % (e.port, dest)
    return "Proxy on port %d is open. The server at %s is answering." % (e.port, dest)

STYLE = """
body{font-family:system-ui,sans-serif;background:#0b0d10;color:#e6e6e6;padding:24px;max-width:1000px;margin:auto}
.hero{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.mascot{width:96px;height:96px;flex:none;background:#000;border-radius:20px}
h1{font-weight:800;margin:0 0 4px;color:#76B900;letter-spacing:.02em} h2{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:#9aa3b5;margin:28px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px} .card{background:#14171d;border:1px solid #232833;border-radius:12px;padding:14px}
.big{font-size:26px;font-weight:700;color:#76B900} .lbl{color:#9aa3b5;font-size:13px} .muted{color:#6b7285;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:14px} td{padding:6px 8px;border-bottom:1px solid #1f232d} .bar div{height:10px;background:linear-gradient(90deg,#3f7a00,#76B900);border-radius:5px} .bar{width:40%}
.cal{background:#14171d;border:1px solid #232833;border-radius:12px;padding:14px;position:relative} .ct{fill:#6b7285;font-size:9px} rect:hover{stroke:#e6e6e6;stroke-width:1}
.legend{text-align:right;font-size:12px;color:#6b7285;margin-top:6px} .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 2px;vertical-align:middle}
#tip{position:fixed;display:none;background:#000;border:1px solid #76B900;color:#e6e6e6;font-size:12px;padding:4px 8px;border-radius:6px;pointer-events:none}
.nav a{color:#9aa3b5;margin-right:14px;text-decoration:none} .nav a.on{color:#76B900;font-weight:700}
a.card{display:block;text-decoration:none;color:inherit} a.card.on{border-color:#76B900}
"""

def page(backend=None):
    labels = {"ollama": "Ollama", "vllm": "vLLM", "llamacpp": "llama.cpp"}
    for e in ENGINES:
        labels[e.id] = e.label
    if backend not in labels:
        backend = None
    c = db(); q = lambda s, a=(): c.execute(s, a).fetchall()
    where, args = _where(backend)
    tin, tout, nreq = q(f"SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COUNT(*) FROM requests WHERE {where}", args)[0]
    tps = q(f"SELECT SUM(output_tokens)*1.0/SUM(eval_s) FROM requests WHERE {where} AND eval_s>0 AND output_tokens>0", args)[0][0]
    blank = q(f"SELECT COUNT(*) FROM requests WHERE {where} AND input_tokens IS NULL AND output_tokens IS NULL", args)[0][0]
    days = q(f"SELECT date(ts,'unixepoch','localtime') d, SUM(COALESCE(input_tokens,0)), SUM(COALESCE(output_tokens,0)) FROM requests WHERE {where} GROUP BY d ORDER BY d DESC LIMIT 30", args)
    models = q(f"SELECT COALESCE(backend,'?'), COALESCE(model,'?'), COUNT(*), SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)) t FROM requests WHERE {where} GROUP BY 1, 2 ORDER BY t DESC LIMIT 10", args)
    eng_rows = q("""SELECT backend, COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
        SUM(CASE WHEN eval_s>0 AND output_tokens>0 THEN output_tokens END)*1.0/SUM(CASE WHEN eval_s>0 AND output_tokens>0 THEN eval_s END)
        FROM requests WHERE source='proxy' GROUP BY backend""")
    cal = calendar(c, where, args); c.close()
    try:
        with open(PRICES) as f: prices = json.load(f)["models"]
    except Exception: prices = []
    estatus = upstream_status(ENGINES)
    by_eng = {bid: (n, i, o, spd) for bid, n, i, o, spd in eng_rows}
    cards = ""
    for e in ENGINES:
        n, i, o, spd = by_eng.get(e.id, (0, 0, 0, None))
        on = " on" if backend == e.id else ""
        speed = f"{spd:.1f} tok/s" if spd else "speed unknown"
        cards += (f"<a class='card{on}' href='/?backend={e.id}'><div class=lbl>{html.escape(e.label)}</div>"
                  f"<div class=big>{fmt((i or 0)+(o or 0))}</div>"
                  f"<div class=muted>{fmt(n)} requests · {speed}</div>"
                  f"<div class=muted>{_engine_phrase(e, estatus.get(e.id))}</div></a>")
    nav = "<div class=nav><a href='/' class='%s'>All engines</a>" % ("" if backend else "on")
    for e in ENGINES:
        nav += "<a href='/?backend=%s' class='%s'>%s</a>" % (e.id, "on" if backend == e.id else "", html.escape(e.label))
    nav += "</div>"
    scope = labels.get(backend, "Every engine")
    mx = max([a + b for _, a, b in days] or [1]) or 1
    ports = ", ".join("%s %d" % (e.label, e.port) for e in ENGINES) or "no proxy"
    dayrows = "".join(f"<tr><td>{d}</td><td class=bar><div style='width:{100*(a+b)/mx:.1f}%'></div></td><td>{fmt(a)}</td><td>{fmt(b)}</td></tr>" for d, a, b in days) or "<tr><td colspan=4 class=muted>The silence before the storm. Point a client at one of the proxy ports (%s) and let the carnage begin.</td></tr>" % html.escape(ports)
    modrows = "".join(f"<tr><td>{html.escape(labels.get(b, b))}</td><td>{html.escape(m)}</td><td>{fmt(n)}</td><td>{fmt(t)}</td></tr>" for b, m, n, t in models) or "<tr><td colspan=4 class=muted>No champions yet. The arena awaits.</td></tr>"
    sav = ""
    for p in prices:
        if p.get("input_per_mtok") is None or p.get("output_per_mtok") is None:
            sav += f"<div class=card><div class=lbl>{html.escape(p['label'])}</div><div class=big>n/a</div><div class=muted>price unverified</div></div>"; continue
        v = tin / 1e6 * p["input_per_mtok"] + tout / 1e6 * p["output_per_mtok"]
        sav += f"<div class=card><div class=lbl>Kept from {html.escape(p['label'])}</div><div class=big>{money(v)}</div><div class=muted>${p['input_per_mtok']}/${p['output_per_mtok']} per M in/out, now crying</div></div>"
    blank_note = ""
    if blank:
        noun = "request" if blank == 1 else "requests"
        blank_note = "<div class=muted>%s %s came through with no token receipt. A streamed chat has to send its usage total on the last chunk, or those tokens stay off the books.</div>" % (fmt(blank), noun)
    view = "Showing %s." % html.escape(scope if backend else "every engine")
    if backend:
        view += " <a href='/'>Show every engine</a>."
    T = "DGX Token Annihilator Dashboard"
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=60><link rel=icon href="/mascot.gif" type="image/gif"><title>{T}</title><style>{STYLE}</style></head><body><div class=hero><img class=mascot src="/mascot.gif" alt="A little chip breathing out dollar signs"><div><h1>{T}</h1><div class=muted>Every token that passes a proxy is a token the cloud will never bill you for. The local servers are left untouched. Last carnage report: {datetime.now():%b %d %H:%M}.</div></div></div>
{nav}
<h2>Engines of destruction</h2><div class=grid>{cards}</div>
<div class=muted style="margin-top:8px">Send each app to the proxy port, not the server port. {view}</div>
<h2>Body count</h2><div class=grid><div class=card><div class=lbl>Tokens annihilated</div><div class=big>{fmt(tin+tout)}</div></div><div class=card><div class=lbl>Prompts devoured (input)</div><div class=big>{fmt(tin)}</div></div><div class=card><div class=lbl>Words unleashed (output)</div><div class=big>{fmt(tout)}</div></div><div class=card><div class=lbl>Requests vanquished</div><div class=big>{fmt(nreq)}</div></div><div class=card><div class=lbl>Annihilation speed (out tok/s)</div><div class=big>{f"{tps:.1f}" if tps else "–"}</div></div></div>
{blank_note}
<h2>Cloud bills obliterated (at list price)</h2><div class=grid>{sav}</div>
<h2>Year of destruction</h2>{cal}
<h2>Daily devastation log</h2><table><tr class=muted><td>Day</td><td></td><td>In</td><td>Out</td></tr>{dayrows}</table>
<h2>Hall of fame: hungriest models</h2><table><tr class=muted><td>Engine</td><td>Model</td><td>Requests</td><td>Tokens</td></tr>{modrows}</table>
<p class=muted>No GPUs were harmed. Several cloud invoices were.</p>
<script>var t=document.getElementById('tip');document.querySelectorAll('rect[data-t]').forEach(function(r){{r.onmousemove=function(e){{t.textContent=r.dataset.t;t.style.display='block';t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px'}};r.onmouseleave=function(){{t.style.display='none'}}}});</script>
</body></html>"""

class Dash(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path.rstrip("/") == "/mascot.gif":
            try:
                with open(os.path.join(HOME, "mascot.gif"), "rb") as f: b = f.read()
                code, ctype = 200, "image/gif"
            except Exception as e:
                b, code, ctype = ("error: %s" % e).encode(), 404, "text/plain; charset=utf-8"
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "public, max-age=86400"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            return
        backend = None
        try:
            backend = (urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("backend") or [None])[0]
        except Exception:
            backend = None
        try: b = page(backend).encode(); code = 200
        except Exception as e: b = ("error: %s" % e).encode(); code = 500
        self.send_response(code); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

class TS(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    def __init__(self, addr, handler, engine=None):
        self.engine = engine
        super().__init__(addr, handler)

def serve():
    global ENGINES
    db().close()
    dash = TS((DASH_BIND, DASH_PORT), Dash)
    threading.Thread(target=dash.serve_forever, name="dashboard", daemon=True).start()
    for e in ENGINES:
        try:
            srv = TS((e.bind, e.port), Proxy, e)
            e.listening = True
            threading.Thread(target=srv.serve_forever, name="proxy-" + e.id, daemon=True).start()
            print(f"{e.id} proxy {e.bind}:{e.port} -> {e.up_host}:{e.up_port}", flush=True)
        except OSError as ex:
            e.listening, e.error = False, str(ex)
            print(f"{e.id} proxy not started on {e.bind}:{e.port}: {ex}", file=sys.stderr, flush=True)
    print(f"dashboard http://{DASH_BIND}:{DASH_PORT}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("stopped", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Token Annihilation Dashboard for Ollama, vLLM, and llama.cpp")
    ap.add_argument("--proxy-bind", default=PROXY_BIND)
    ap.add_argument("--proxy-port", type=int, default=PROXY_PORT, help="Ollama proxy port (default 11435)")
    ap.add_argument("--vllm-proxy-port", type=int, default=VLLM_PROXY_PORT, help="vLLM proxy port (default 8001)")
    ap.add_argument("--llamacpp-proxy-port", type=int, default=LLAMACPP_PROXY_PORT, help="llama.cpp proxy port (default 8081)")
    ap.add_argument("--dash-bind", default=DASH_BIND); ap.add_argument("--dash-port", type=int, default=DASH_PORT)
    ap.add_argument("--engines", default=os.environ.get("ENGINES", "ollama,vllm,llamacpp"), help="comma-separated: ollama,vllm,llamacpp")
    ap.add_argument("--backfill", nargs="?", const="", default=None, metavar="UNIX_TS_CUTOFF")
    a = ap.parse_args()
    PROXY_BIND = a.proxy_bind
    PROXY_PORT, VLLM_PROXY_PORT, LLAMACPP_PROXY_PORT = a.proxy_port, a.vllm_proxy_port, a.llamacpp_proxy_port
    DASH_BIND, DASH_PORT = a.dash_bind, a.dash_port
    ENGINES = select_engines(a.engines)
    for e in ENGINES:
        e.bind = PROXY_BIND
    if a.backfill is not None:
        db().close(); backfill(float(a.backfill) if a.backfill else None); sys.exit()
    serve()
