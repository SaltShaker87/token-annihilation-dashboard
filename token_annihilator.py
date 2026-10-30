#!/usr/bin/env python3
"""Token Annihilation Dashboard: transparent token-counting proxies (Ollama, vLLM, llama.cpp / any
OpenAI-compatible server) + llama.cpp log tailing + a tiny, gloriously dramatic dashboard. Stdlib only.
MIT licensed. https://github.com/SaltShaker87/token-annihilation-dashboard"""
import http.client, http.server, json, os, re, sqlite3, subprocess, sys, threading, time, html
from datetime import datetime
from socketserver import ThreadingMixIn

HOME = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HOME, "stats.db")
PRICES = os.path.join(HOME, "prices.json")
DB = os.environ.get("STATS_DB", DB)
UP_HOST, UP_PORT = os.environ.get("UPSTREAM_HOST", "127.0.0.1"), int(os.environ.get("UPSTREAM_PORT", 11434))   # Ollama
PROXY_BIND, PROXY_PORT = os.environ.get("PROXY_BIND", "0.0.0.0"), int(os.environ.get("PROXY_PORT", 11435))
# Older builds picked runtimes with ENGINES=ollama,vllm,llamacpp (+ VLLM_* / LLAMACPP_* ports). Still honoured.
ENGINES = [{"llama.cpp": "llamacpp", "llama-cpp": "llamacpp", "llama": "llamacpp"}.get(x.strip().lower(), x.strip().lower())
           for x in os.environ.get("ENGINES", "ollama,vllm,llamacpp").split(",") if x.strip()]
OLLAMA_PROXY = os.environ.get("OLLAMA_PROXY", "1" if "ollama" in ENGINES else "0") != "0"   # set 0 if you only run vLLM / llama.cpp
PROXY_SOURCE = os.environ.get("PROXY_SOURCE", "")   # force the label of the :11435 proxy (default: detect what answers upstream)

def _default_bind():
    try:
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        if ip: return ip[0]
    except Exception: pass
    return "127.0.0.1"
DASH_BIND, DASH_PORT = os.environ.get("DASH_BIND") or _default_bind(), int(os.environ.get("DASH_PORT", 11480))
# Extra OpenAI-compatible counting proxies (vLLM, llama-server, anything that returns a "usage" field):
#   OPENAI_PROXIES="vllm:8001=127.0.0.1:8000,llamacpp:8081=127.0.0.1:8080"   (source:listen_port=upstream_host:port)
# Unset = the defaults above (vLLM on 8001, llama.cpp on 8081), so several runtimes at once just work. Set it empty for none.
OPENAI_PROXIES = os.environ.get("OPENAI_PROXIES")
INJECT_USAGE = os.environ.get("INJECT_USAGE", "1") != "0"   # add stream_options.include_usage to streaming /v1 requests
TRACK = ("/api/generate", "/api/chat", "/v1/chat/completions", "/v1/completions", "/v1/responses", "/api/embed", "/api/embeddings", "/v1/embeddings")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection"}
lock = threading.Lock()
LIVE = {}            # request id -> {model, endpoint, out_est, t0}
LIVE_LOCK = threading.Lock()
LIVE_VER = [0]       # bumped on every change
DONE = []            # recent finished requests with exact counts: (ver, dict)
CLIENTS = [0]        # connected SSE clients
_rid = [0]
def live_bump():
    LIVE_VER[0] += 1

def db():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, ts REAL, model TEXT, endpoint TEXT,
      input_tokens INTEGER, output_tokens INTEGER, duration_s REAL, eval_s REAL, status INTEGER, source TEXT, dedupe TEXT UNIQUE)""")
    return c

# Rows that count. Only the token-less journal backfill is hidden; any future source (e.g. llama.cpp) is included automatically.
V = "source!='backfill'"
SOURCE_LABELS = {"proxy": "Ollama", "ollama": "Ollama", "llamacpp": "llama.cpp", "vllm": "vLLM", "nim": "NVIDIA NIM", "nemo": "NeMo",
                 "trtllm": "TensorRT-LLM", "triton": "Triton", "onnx": "ONNX Runtime", "sglang": "SGLang", "lmstudio": "LM Studio",
                 "openai": "OpenAI-compatible"}   # any other OPENAI_PROXIES name is shown as written
TOT = [0, 0]         # running all-time (input, output) for milestone checks
RUNTIMES = []        # every proxy we run: {"src", "listen", "up", "cls", "ok", "kind", "error"}

def init_db():
    c = db()
    c.executescript("""CREATE TABLE IF NOT EXISTS milestones(key TEXT PRIMARY KEY, kind TEXT, value REAL, reached_ts REAL, acked_ts REAL);
      CREATE TABLE IF NOT EXISTS trips(id INTEGER PRIMARY KEY, name TEXT, start_ts REAL, end_ts REAL);
      CREATE INDEX IF NOT EXISTS req_ts ON requests(ts); CREATE INDEX IF NOT EXISTS req_src ON requests(source);""")
    # The older multi-engine build logged every proxy as source='proxy' and kept the runtime in a `backend` column,
    # which made vLLM / llama.cpp traffic show up as Ollama. Move those rows to their real source (idempotent).
    if "backend" in [r[1] for r in c.execute("PRAGMA table_info(requests)")]:
        n = c.execute("UPDATE requests SET source=backend WHERE source='proxy' AND backend IN ('vllm','llamacpp')").rowcount
        if n: print("relabelled %d older vLLM / llama.cpp rows that were shown as Ollama" % n, flush=True)
    if not c.execute("SELECT 1 FROM trips WHERE end_ts IS NULL").fetchone():
        first = c.execute(f"SELECT MIN(ts) FROM requests WHERE {V}").fetchone()[0]
        c.execute("INSERT INTO trips(name,start_ts) VALUES(NULL,?)", (first or time.time(),))
    TOT[0], TOT[1] = c.execute(f"SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0) FROM requests WHERE {V}").fetchone()
    check_ms(c); c.commit(); c.close()

def hero_price():
    try:
        for p in json.load(open(PRICES))["models"]:
            if p.get("input_per_mtok") is not None and p.get("output_per_mtok") is not None: return p
    except Exception: pass
    return None

def usd(p, i, o): return (i or 0) / 1e6 * p["input_per_mtok"] + (o or 0) / 1e6 * p["output_per_mtok"] if p else 0.0

def ms_reached(tok, dollars):
    out = [("tok", v) for v in (1_000_000, 10_000_000) if tok >= v]
    v = 100_000_000
    while v <= tok: out.append(("tok", v)); v += 100_000_000
    v = 100
    while v <= dollars: out.append(("usd", v)); v += 100
    return out

def check_ms(c):
    new = []
    for kind, v in ms_reached(TOT[0] + TOT[1], usd(hero_price(), TOT[0], TOT[1])):
        key = "%s:%d" % (kind, v)
        if c.execute("INSERT OR IGNORE INTO milestones(key,kind,value,reached_ts) VALUES(?,?,?,?)", (key, kind, v, time.time())).rowcount:
            new.append({"key": key, "kind": kind, "value": v})
    return new

def log_row(**r):
    try:
        with lock:
            c = db(); cur = c.execute("INSERT OR IGNORE INTO requests(ts,model,endpoint,input_tokens,output_tokens,duration_s,eval_s,status,source,dedupe) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (r.get("ts"), r.get("model"), r.get("endpoint"), r.get("inp"), r.get("out"), r.get("dur"), r.get("eval_s"), r.get("status"), r.get("source"), r.get("dedupe")))
            new = []
            if cur.rowcount and r.get("source") != "backfill":
                TOT[0] += r.get("inp") or 0; TOT[1] += r.get("out") or 0
                new = check_ms(c)
            c.commit(); c.close()
            return new
    except Exception as e:
        print("log error:", e, file=sys.stderr)
    return []

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
        if isinstance(o.get("response"), dict):   # /v1/responses stream: the usage rides on response.completed
            o = o["response"]; model = o.get("model") or model
        u = o.get("usage")
        if isinstance(u, dict) and "input_tokens" in u:   # Responses API naming
            inp, out = u.get("input_tokens"), u.get("output_tokens", 0)
            break
        if isinstance(u, dict):
            inp, out = u.get("prompt_tokens"), u.get("completion_tokens", 0)
            tm = o.get("timings")   # llama-server extra
            if isinstance(tm, dict) and tm.get("predicted_ms"): eval_s = tm["predicted_ms"] / 1e3
            break
    return model, inp, out, eval_s

def add_usage_opt(path, body):
    """vLLM / llama-server only report usage on a stream when asked. Ask politely; never break the request."""
    if not INJECT_USAGE or not body or not path.startswith("/v1/") or "embed" in path or path.endswith("/responses"): return body
    try:
        j = json.loads(body)
        if isinstance(j, dict) and j.get("stream") and "stream_options" not in j:
            j["stream_options"] = {"include_usage": True}; return json.dumps(j).encode()
    except Exception: pass
    return body

class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    UPH, UPP, SRC = UP_HOST, UP_PORT, "proxy"
    def log_message(self, *a): pass
    def handle_any(self):
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        if self.command == "POST": body = add_usage_opt(self.path.split("?")[0], body)
        up = http.client.HTTPConnection(self.UPH, self.UPP, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "content-length"}
        if body is not None: hdrs["Content-Length"] = str(len(body))
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
        rid = None
        src = self.SRC
        streamed, t_first, t_last, nchunks = False, None, None, 0
        if track:
            try:
                m = None
                try: j = json.loads(body or b"{}"); m = j.get("model"); streamed = j.get("stream") is True
                except Exception: pass
                with LIVE_LOCK:
                    _rid[0] += 1; rid = _rid[0]
                    LIVE[rid] = {"model": m, "endpoint": path, "out_est": 0, "t0": t0, "src": src}; live_bump()
            except Exception: rid = None
        try:
            while True:
                chunk = r.read1(65536) if self.command != "HEAD" else b""
                if not chunk: break
                if chunked: self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else: self.wfile.write(chunk)
                self.wfile.flush()
                if track:
                    tail += chunk
                    t_last = time.time(); t_first = t_first or t_last; nchunks += 1
                    if rid is not None and CLIENTS[0]:
                        try:   # SSE (vLLM / llama-server) = one "data:" event per token; Ollama NDJSON = one line per token
                            LIVE[rid]["out_est"] += chunk.count(b"data:") or chunk.count(b"\n") or 1; live_bump()
                        except Exception: pass
                    if len(tail) > 4_000_000: del tail[:-262144]
            if chunked: self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        finally: up.close()
        if track:
            try:
                model, inp, out, eval_s = parse_usage(path, body, bytes(tail))
                if eval_s is None and streamed and (out or 0) >= 2 and nchunks >= 3 and t_last > t_first:
                    # vLLM reports no timings. The first chunk lands after prefill, so first->last chunk is decode time
                    # for out-1 tokens; scale it up to all out tokens.
                    eval_s = (t_last - t_first) * out / (out - 1)
                new_ms = log_row(ts=t0, model=model, endpoint=path, inp=inp, out=out, dur=time.time() - t0, eval_s=eval_s, status=r.status, source=src)
                with LIVE_LOCK:
                    LIVE.pop(rid, None); live_bump()
                    DONE.append((LIVE_VER[0], {"model": model, "endpoint": path, "in": inp or 0, "out": out or 0, "dur": round(time.time() - t0, 2), "eval_s": eval_s, "src": src, "ms": new_ms}))
                    del DONE[:-20]
            except Exception as e: print("parse error:", e, file=sys.stderr)
            finally:
                if rid is not None and rid in LIVE:
                    LIVE.pop(rid, None); live_bump()
    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_OPTIONS = do_PATCH = handle_any

def probe(host, port):
    """Who answers at host:port? 'ollama' / 'vllm' / 'llamacpp', 'up' (something else), or None (nothing)."""
    def get(path):
        c = http.client.HTTPConnection(host, port, timeout=2)
        try: c.request("GET", path); r = c.getresponse(); return r.status, r.read(1_000_000)
        finally: c.close()
    try:
        s, b = get("/api/version")
        if s == 200 and b'"version"' in b: return "ollama"
        s, b = get("/v1/models")
        try: owners = {str(m.get("owned_by", "")).lower() for m in json.loads(b).get("data", [])} if s == 200 else set()
        except Exception: owners = set()
        for k in ("vllm", "llamacpp"):
            if k in owners: return k
        return "up"
    except Exception:
        return None

def check_runtimes():
    for rt in RUNTIMES:
        if rt.get("error"): continue
        kind = probe(*rt["up"])
        if not rt["started"]:   # a default proxy claims its port only once the right runtime answers upstream
            if kind == rt["src"]: bind_proxy(rt)
            rt["ok"], rt["kind"] = rt["started"], kind
            continue
        rt["ok"], rt["kind"] = kind is not None, kind
        # the :11435 proxy is Ollama's by default, but people point it at vLLM too: label by what actually answers
        if rt["cls"] is Proxy and not PROXY_SOURCE and kind in ("ollama", "vllm", "llamacpp"):
            Proxy.SRC = rt["src"] = "proxy" if kind == "ollama" else kind

def runtimes_loop():
    while True:
        try: check_runtimes()
        except Exception as e: print("runtime check error:", e, file=sys.stderr)
        time.sleep(30)

def runtimes_info(c):
    """`used`: it has counted tokens before, so it being down is news (an unused default proxy being down is not)."""
    used = lambda s: bool(c.execute("SELECT 1 FROM requests WHERE source=? LIMIT 1", (s,)).fetchone())
    return [{"label": SOURCE_LABELS.get(rt["src"], rt["src"]), "listen": rt["listen"], "up": "%s:%d" % rt["up"],
             "ok": rt.get("ok"), "error": rt.get("error"), "used": used(rt["src"])} for rt in RUNTIMES]

def proxy_specs():
    """OPENAI_PROXIES if set; otherwise the older build's defaults / VLLM_* / LLAMACPP_* env vars."""
    if OPENAI_PROXIES is not None: return OPENAI_PROXIES
    e, out = os.environ.get, []
    if "vllm" in ENGINES:
        out.append("vllm:%s=%s:%s" % (e("VLLM_PROXY_PORT", "8001"), e("VLLM_UPSTREAM_HOST", "127.0.0.1"), e("VLLM_UPSTREAM_PORT", "8000")))
    if "llamacpp" in ENGINES and not LLAMA_LOG:   # tailing the log already counts it; don't count twice
        out.append("llamacpp:%s=%s:%s" % (e("LLAMACPP_PROXY_PORT", "8081"), e("LLAMACPP_UPSTREAM_HOST", "127.0.0.1"), e("LLAMACPP_UPSTREAM_PORT", "8080")))
    return ",".join(out)

LEGACY_ENV = ("ENGINES", "VLLM_PROXY_PORT", "VLLM_UPSTREAM_HOST", "VLLM_UPSTREAM_PORT", "LLAMACPP_PROXY_PORT", "LLAMACPP_UPSTREAM_HOST", "LLAMACPP_UPSTREAM_PORT")
def auto_proxies():
    """Nothing configured: the vLLM / llama.cpp proxies are only defaults, so wait until the right runtime is detected
    upstream. Otherwise a default could grab a port someone else uses, or forward (on PROXY_BIND) to an unrelated app on 8000/8080."""
    return OPENAI_PROXIES is None and not any(k in os.environ for k in LEGACY_ENV)

def start_proxy(cls, src, port, up, auto=False):
    rt = {"src": src, "listen": port, "up": up, "cls": cls, "ok": None, "kind": None, "started": False, "srv": None}
    RUNTIMES.append(rt)
    if auto: print(f"{SOURCE_LABELS.get(src, src)} proxy {PROXY_BIND}:{port} starts once {SOURCE_LABELS.get(src, src)} answers at {up[0]}:{up[1]}", flush=True)
    else: bind_proxy(rt)

def bind_proxy(rt):
    src, port, up = rt["src"], rt["listen"], rt["up"]
    try: srv = TS((PROXY_BIND, port), rt["cls"])
    except OSError as e:
        rt["error"] = "port %d busy" % port
        print(f"{SOURCE_LABELS.get(src, src)} proxy NOT started, port {port}: {e}", file=sys.stderr, flush=True); return
    rt["started"], rt["srv"] = True, srv
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"{SOURCE_LABELS.get(src, src)} proxy {PROXY_BIND}:{port} -> {up[0]}:{up[1]}", flush=True)

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

def stats(c, since=None):
    q = lambda s_, *a: c.execute(s_, a).fetchall()
    W = V + (" AND ts>=%f" % float(since) if since else "")   # time-filtered view
    hp = hero_price(); pi, po = (float(hp["input_per_mtok"]), float(hp["output_per_mtok"])) if hp else (0.0, 0.0)
    TOKX = "(COALESCE(input_tokens,0)+COALESCE(output_tokens,0))"
    COST = f"((COALESCE(input_tokens,0)*{pi}+COALESCE(output_tokens,0)*{po})/1e6)"
    LT = "'unixepoch','localtime'"
    R = []
    def add(icon, title, value=None, detail="", ts=None, date=None, hint="Not yet earned. Keep feeding the beast."):
        R.append({"icon": icon, "title": title, "value": value, "detail": detail, "ts": ts, "date": date, "hint": hint})
    def top(expr, where="1", asc=False):
        return c.execute(f"SELECT {expr} v, COALESCE(model,'?'), ts FROM requests WHERE {V} AND {where} AND ({expr}) IS NOT NULL ORDER BY v {'ASC' if asc else 'DESC'} LIMIT 1").fetchone()
    def grp(key, where="1", order="t"):
        return c.execute(f"SELECT {key} k, SUM({TOKX}) t, COUNT(*) n, SUM({COST}) s, COUNT(DISTINCT model) m FROM requests WHERE {V} AND {where} GROUP BY k ORDER BY {order} DESC LIMIT 1").fetchone()
    F = lambda n: "{:,}".format(int(n or 0))
    M = lambda x: "${:,.4f}".format(x) if x < 1 else "${:,.2f}".format(x)
    def pretty_day(d):
        try: return datetime.strptime(d, "%Y-%m-%d").strftime("%a %b {d}, %Y").format(d=datetime.strptime(d, "%Y-%m-%d").day)
        except Exception: return d
    SPD = "output_tokens*1.0/eval_s"; SPW = "eval_s>0 AND output_tokens>=20"
    r = top(SPD, SPW); add("🏎️", "Speed Demon", r and "%.1f tok/s" % r[0], r and f"{r[1]}, fastest single generation", r and r[2], hint="Needs a chat/generate run with 20+ output tokens.")
    r = top(SPD, SPW, asc=True); add("🐢", "Thoughtful Sloth", r and "%.1f tok/s" % r[0], r and f"{r[1]} took its sweet time", r and r[2])
    r = top("input_tokens", "input_tokens>0"); add("🍔", "Biggest Appetite", r and F(r[0]) + " tokens", r and f"Largest prompt devoured in one bite ({r[1]})", r and r[2])
    r = top("output_tokens", "output_tokens>0"); add("📜", "Epic Monologue", r and F(r[0]) + " tokens", r and f"Longest single answer ({r[1]})", r and r[2])
    r = top(TOKX, f"{TOKX}>0"); add("🦣", "Mammoth Request", r and F(r[0]) + " tokens", r and f"Biggest request, in + out ({r[1]})", r and r[2])
    r = top("duration_s", "duration_s>0"); add("⏳", "Marathon Generation", r and "%.1f s" % r[0], r and f"Longest single request ({r[1]})", r and r[2])
    r = top("duration_s", "duration_s>0 AND output_tokens>0", asc=True); add("⚡", "Blink and You Missed It", r and "%.2f s" % r[0], r and f"Quickest complete answer ({r[1]})", r and r[2])
    r = top(COST, f"{TOKX}>0") if hp else None; add("💸", "Priciest Request Dodged", r and M(r[0]), r and f"What {hp['label']} would have billed for one {r[1]} call", r and r[2])
    days = q(f"SELECT date(ts,{LT}) d, SUM({TOKX}), COUNT(*), SUM({COST}), COUNT(DISTINCT model) FROM requests WHERE {V} GROUP BY d ORDER BY d")
    if days:
        b = max(days, key=lambda x: x[1]); add("📅", "Day of Reckoning", F(b[1]) + " tokens", "Busiest day ever", date=pretty_day(b[0]))
    else: add("📅", "Day of Reckoning")
    r = grp(f"strftime('%Y-%m-%d %H:00',ts,{LT})"); add("🔥", "Rush Hour", r and F(r[1]) + " tokens", r and "Most tokens in a single clock hour", date=r and r[0])
    r = grp(f"date(ts,{LT},'-6 days','weekday 1')"); add("🗓️", "Week of Wrath", r and F(r[1]) + " tokens", "Busiest week", date=r and "week of " + pretty_day(r[0]))
    r = grp(f"strftime('%Y-%m',ts,{LT})"); add("🌕", "Month of Mayhem", r and F(r[1]) + " tokens", "Busiest calendar month", date=r and datetime.strptime(r[0], "%Y-%m").strftime("%B %Y"))
    if days:
        b = max(days, key=lambda x: x[2]); add("🔁", "Request Frenzy", F(b[2]) + " requests", "Most requests in one day", date=pretty_day(b[0]))
    else: add("🔁", "Request Frenzy")
    r = grp(f"strftime('%Y-%m-%d %H:00',ts,{LT})", order="n"); add("🐝", "Hive Mind Hour", r and F(r[2]) + " requests", "Most requests in one clock hour", date=r and r[0])
    # streaks
    best = cur = 0; bend = None; prev = None; ds = []
    for d in days:
        dt = datetime.strptime(d[0], "%Y-%m-%d").date(); ds.append(dt)
        cur = cur + 1 if prev and (dt - prev).days == 1 else 1; prev = dt
        if cur > best: best, bend = cur, dt
    add("🔗", "Iron Streak", best and f"{best} day{'s' if best != 1 else ''}", "Longest run of consecutive active days", date=bend and "ended " + bend.strftime("%b {}, %Y").format(bend.day))
    today = datetime.now().date(); live = 0
    if ds and (today - ds[-1]).days <= 1:
        live = 1
        for i in range(len(ds) - 1, 0, -1):
            if (ds[i] - ds[i - 1]).days == 1: live += 1
            else: break
    add("🕯️", "Current Streak", live and f"{live} day{'s' if live != 1 else ''}", "Don't let the flame go out", date=live and "still burning", hint="Use a model today (or yesterday) to light it.")
    if days and hp:
        b = max(days, key=lambda x: x[3]); add("💰", "Biggest Payday", M(b[3]), f"Most saved in one day vs {hp['label']}", date=pretty_day(b[0]))
        b = max(days, key=lambda x: x[4]); add("🧪", "Model Buffet", f"{b[4]} model{'s' if b[4] != 1 else ''}", "Most different models used in one day", date=pretty_day(b[0]))
    else: add("💰", "Biggest Payday"); add("🧪", "Model Buffet")
    r = grp(f"date(ts,{LT})", f"CAST(strftime('%H',ts,{LT}) AS INTEGER)<5"); add("🦉", "Night Owl", r and F(r[1]) + " tokens", "Most tokens annihilated between midnight and 5 AM", date=r and pretty_day(r[0]), hint="Requires questionable sleep hygiene.")
    r = grp(f"date(ts,{LT})", f"strftime('%w',ts,{LT}) IN ('0','6')"); add("🛋️", "Weekend Warrior", r and F(r[1]) + " tokens", "Busiest Saturday or Sunday", date=r and pretty_day(r[0]), hint="Touch grass? Never heard of it.")
    r = grp(f"strftime('%H',ts,{LT})")
    add("📺", "Prime Time", r and datetime.strptime(r[0], "%H").strftime("%I %p").lstrip("0"), r and f"Hour of the day with the most carnage ({F(r[1])} tokens all-time)")
    r = c.execute(f"SELECT ts, COALESCE(model,'?') FROM requests WHERE {V} ORDER BY ts LIMIT 1").fetchone()
    add("🐣", "First Blood", r and r[1], "The very first token annihilated", r and r[0])
    add("📆", "Days of Destruction", days and f"{len(days)} day{'s' if len(days) != 1 else ''}", "Total days with any annihilation", date=days and "since " + pretty_day(days[0][0]))
    # leaderboard (all-time for trophies, filtered for display)
    def board(where):
      L = []
      for m, n, ti, to, eo, es, bst, mo, f0, f1, src in q(f"""SELECT COALESCE(model,'?'), COUNT(*), SUM(COALESCE(input_tokens,0)), SUM(COALESCE(output_tokens,0)),
        SUM(CASE WHEN eval_s>0 AND output_tokens>0 THEN output_tokens ELSE 0 END), SUM(CASE WHEN eval_s>0 AND output_tokens>0 THEN eval_s ELSE 0 END),
        MAX(CASE WHEN {SPW} THEN {SPD} END), MAX(COALESCE(output_tokens,0)), MIN(ts), MAX(ts), GROUP_CONCAT(DISTINCT source)
        FROM requests WHERE {where} GROUP BY 1 ORDER BY SUM({TOKX}) DESC"""):
        L.append({"model": m, "n": n, "tin": ti, "tout": to, "tok": ti + to, "avg": eo / es if es else None, "best": bst, "maxout": mo, "first": f0, "last": f1,
                  "src": ", ".join(SOURCE_LABELS.get(x, x) for x in (src or "").split(",") if x)})
      return L
    L = board(V)
    if L:
        add("👑", "Reigning Champion", L[0]["model"], f"Hungriest model: {F(L[0]['tok'])} tokens", L[0]["last"])
        fa = [x for x in L if x["avg"]]; b = max(fa, key=lambda x: x["avg"]) if fa else None
        add("🚀", "Fastest Model", b and b["model"], b and "%.1f tok/s average" % b["avg"], b and b["last"])
        ch = [x for x in L if x["tout"]]; b = max(ch, key=lambda x: x["tout"] / x["n"]) if ch else None
        add("🗣️", "Chattiest Model", b and b["model"], b and f"{F(b['tout'] / b['n'])} output tokens per request on average", b and b["last"])
        b = max(L, key=lambda x: x["n"]); add("🧲", "Most Loyal Model", b["model"], f"{F(b['n'])} requests answered", b["last"])
    else:
        for t in ("👑 Reigning Champion", "🚀 Fastest Model", "🗣️ Chattiest Model", "🧲 Most Loyal Model"): add(*t.split(" ", 1))
    for x in R:
        if not x["value"]: x["value"] = x["ts"] = x["date"] = None; x["detail"] = ""
    sources = [{"src": s_, "label": SOURCE_LABELS.get(s_, s_), "n": n, "tok": t} for s_, n, t in q(f"SELECT source, COUNT(*), SUM({TOKX}) FROM requests WHERE {W} GROUP BY 1 ORDER BY 3 DESC")]
    ri, ro, rn = q(f"SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COUNT(*) FROM requests WHERE {W}")[0]
    re_ = q(f"SELECT COALESCE(SUM(output_tokens),0),COALESCE(SUM(eval_s),0) FROM requests WHERE {W} AND eval_s>0 AND output_tokens>0")[0]
    rng = {"since": since, "tin": ri, "tout": ro, "nreq": rn, "ev_out": re_[0], "ev_s": re_[1]}
    # trips
    def tt(a, b):
        i, o, n, eo, es = c.execute(f"""SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COUNT(*),
          COALESCE(SUM(CASE WHEN eval_s>0 AND output_tokens>0 THEN output_tokens END),0),COALESCE(SUM(CASE WHEN eval_s>0 AND output_tokens>0 THEN eval_s END),0)
          FROM requests WHERE {V} AND ts>=? AND ts<?""", (a, b or 1e13)).fetchone()
        return {"tin": i, "tout": o, "n": n, "usd": usd(hp, i, o), "ev_out": eo, "ev_s": es}
    trip = None; hist = []
    for tid, name, a, b in q("SELECT id,name,start_ts,end_ts FROM trips ORDER BY id DESC LIMIT 13"):
        d = {"id": tid, "name": name, "start_ts": a, "end_ts": b, **tt(a, b)}
        if b is None and trip is None: trip = d
        elif b is not None: hist.append(d)
    ms = [{"key": k, "kind": kd, "value": v, "reached_ts": rt, "acked": at is not None} for k, kd, v, rt, at in q("SELECT key,kind,value,reached_ts,acked_ts FROM milestones ORDER BY reached_ts, value")]
    return {"records": R, "leaderboard": board(W) if since else L, "rng": rng, "sources": sources, "trip": trip, "trips": hist[:12], "milestones": ms,
            "runtimes": runtimes_info(c)}

def page_data():
    c = db(); q = lambda s, *a: c.execute(s, a).fetchall()
    P = V
    tin, tout, nreq = q(f"SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COUNT(*) FROM requests WHERE {P}")[0]
    ev = q(f"SELECT COALESCE(SUM(output_tokens),0),COALESCE(SUM(eval_s),0) FROM requests WHERE {P} AND eval_s>0 AND output_tokens>0")[0]
    days = q(f"SELECT date(ts,'unixepoch','localtime') d, SUM(COALESCE(input_tokens,0)), SUM(COALESCE(output_tokens,0)) FROM requests WHERE {P} GROUP BY d ORDER BY d DESC LIMIT 400")
    st = stats(c)
    c.close()
    try: prices = [p for p in json.load(open(PRICES))["models"]]
    except Exception: prices = []
    return {"tin": tin, "tout": tout, "nreq": nreq, "ev_out": ev[0], "ev_s": ev[1], "days": days, "prices": prices,
            "today": datetime.now().strftime("%Y-%m-%d"), "port": PROXY_PORT, "st": st}

def api_stats(since=None):
    c = db()
    try: return stats(c, since)
    finally: c.close()

def trip_reset(name):
    name = (str(name or "").strip()[:60]) or None
    with lock:
        c = db(); now = time.time()
        c.execute("UPDATE trips SET end_ts=?, name=COALESCE(?,name) WHERE end_ts IS NULL", (now, name))
        c.execute("INSERT INTO trips(name,start_ts) VALUES(NULL,?)", (now,)); c.commit(); c.close()

def ms_ack(keys):
    keys = [str(k) for k in (keys or [])][:500]
    if not keys: return
    with lock:
        c = db(); c.executemany("UPDATE milestones SET acked_ts=? WHERE key=? AND acked_ts IS NULL", [(time.time(), k) for k in keys]); c.commit(); c.close()

def page():
    d = page_data()
    return PAGE.replace("__DATA__", json.dumps(d).replace("</", "<\\/"))

def live_state():
    now = time.time()
    with LIVE_LOCK:
        streams = [{"id": k, "model": v["model"], "endpoint": v["endpoint"], "out_est": v["out_est"], "elapsed": round(now - v["t0"], 1),
                    "src": SOURCE_LABELS.get(v.get("src"), v.get("src"))} for k, v in LIVE.items()]
        done = [(v, dict(x, src=SOURCE_LABELS.get(x.get("src"), x.get("src")))) for v, x in DONE]
    return streams, done

STATIC_TYPES = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4", ".webm": "video/webm"}

class Dash(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/events": return self.events()
        if p == "/api/stats":
            qs = dict(x.split("=", 1) for x in self.path.partition("?")[2].split("&") if "=" in x)
            try: days = int(qs.get("days", 0))
            except ValueError: days = 0
            return self.json(api_stats(time.time() - days * 86400 if days > 0 else None))
        if p.startswith("/static/"):
            name = os.path.basename(p)
            ct = STATIC_TYPES.get(os.path.splitext(name)[1])
            fp = os.path.join(HOME, "static", name)
            if ct and os.path.isfile(fp):
                b = open(fp, "rb").read(); code, extra = 200, {"Accept-Ranges": "bytes"}
                m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
                if m and (m.group(1) or m.group(2)):   # range requests so <video> loops/seeks happily
                    a = int(m.group(1)) if m.group(1) else len(b) - int(m.group(2)); z = int(m.group(2)) if m.group(1) and m.group(2) else len(b) - 1
                    a, z = max(0, a), min(z, len(b) - 1); extra["Content-Range"] = "bytes %d-%d/%d" % (a, z, len(b)); b = b[a:z + 1]; code = 206
                self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "max-age=86400"); self.send_header("Content-Length", str(len(b)))
                for k, v in extra.items(): self.send_header(k, v)
                self.end_headers(); self.wfile.write(b); return
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        try: b = page().encode(); code = 200
        except Exception as e: b = ("error: %s" % e).encode(); code = 500
        self.send_response(code); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        p = self.path.split("?")[0]
        n = min(int(self.headers.get("Content-Length") or 0), 8192)
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except Exception: body = {}
        if not isinstance(body, dict): body = {}
        try:
            if p == "/api/trip/reset": trip_reset(body.get("name")); return self.json(api_stats())
            if p == "/api/milestones/ack": ms_ack(body.get("keys")); return self.json({"ok": True})
        except Exception as e: return self.json({"error": str(e)}, 500)
        self.json({"error": "not found"}, 404)
    def events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
        self.close_connection = True
        CLIENTS[0] += 1
        try:
            last_ver, last_send = -1, 0.0
            _, done = live_state()
            seen = done[-1][0] if done else 0
            self.wfile.write(b"retry: 5000\n: hello\n\n"); self.wfile.flush()
            while True:
                time.sleep(0.25)                      # throttle: at most ~4 events/sec
                now = time.time()
                if LIVE_VER[0] != last_ver:
                    last_ver = LIVE_VER[0]
                    streams, done = live_state()
                    fin = [x for v, x in done if v > seen]
                    if done: seen = max(seen, done[-1][0])
                    msg = json.dumps({"streams": streams, "done": fin})
                    self.wfile.write(b"data: " + msg.encode() + b"\n\n"); self.wfile.flush(); last_send = now
                elif LIVE:                            # keep elapsed ticking while streaming
                    streams, _ = live_state()
                    self.wfile.write(b"data: " + json.dumps({"streams": streams, "done": []}).encode() + b"\n\n"); self.wfile.flush(); last_send = now
                elif now - last_send > 15:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush(); last_send = now
        except Exception:
            pass
        finally:
            CLIENTS[0] -= 1

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>DGX Token Annihilator Dashboard</title><link rel=icon type=image/png sizes=32x32 href="static/icon-32.png"><link rel=apple-touch-icon sizes=180x180 href="static/icon-180.png"><style>
:root{--g:#76B900}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:#0b0d10;color:#e6e6e6;margin:0;padding:20px 28px;max-width:1240px;margin:auto}
h1{font-weight:800;margin:0;color:var(--g);letter-spacing:.02em}.brand{display:flex;align-items:center;gap:14px}.brand img{border-radius:14px;display:block}.top{display:flex;justify-content:space-between;align-items:center}
.badge{border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;border:1px solid #2f4a10;color:#9fd14a;background:#13200a}.badge.off{color:#ffb84d;background:#2a1d00;border-color:#6b4a00}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#9aa3b5;margin:0 0 8px}h3{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#9aa3b5;margin:26px 0 8px}
.muted{color:#6b7285;font-size:12px}.card{background:#14171d;border:1px solid #232833;border-radius:14px;padding:14px;position:relative;overflow:hidden}
.row{display:grid;gap:14px;margin-top:14px}.r1{grid-template-columns:1.6fr 1fr 1fr;align-items:stretch}.r2{grid-template-columns:1.6fr 2fr}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.odo{--dw:36px;--dh:58px;display:flex;align-items:center;gap:3px;font:700 calc(var(--dh)*.93)/1 ui-monospace,Menlo,monospace;color:var(--g);margin:6px 0;max-width:100%}
.dig{height:var(--dh);width:var(--dw);flex:none;overflow:hidden;background:#0b0d10;border-radius:6px;border:1px solid #1f232d}.dig div{transition:transform .35s cubic-bezier(.2,.8,.2,1)}.dig span{display:block;height:var(--dh);line-height:var(--dh);text-align:center}
.sep{width:calc(var(--dw)/3);flex:none;height:var(--dh);display:flex;align-items:flex-end;justify-content:center;padding-bottom:calc(var(--dh)/10);font-size:calc(var(--dh)*.69);line-height:1;color:#3f7a00}
.unit{flex:none;margin-left:8px;font:800 13px system-ui,sans-serif;letter-spacing:.08em;color:#0b0d10;background:var(--g);border-radius:6px;padding:4px 7px}.words{color:#9fd14a;font-size:13px;font-weight:600}
.scroll{max-height:380px;overflow:auto;border-radius:10px;scrollbar-color:#2f4a10 #0b0d10}.scroll .muted td{position:sticky;top:0;background:#14171d;z-index:1}#ledger{max-height:420px}#ledger .muted td{background:#0b0d10}
.rts{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;margin-left:auto}.chip.down{border-style:dashed;color:#6b7285;border-color:#2a2f3a;background:none}.chip.busy{color:#ffb84d;border-color:#6b4a00;background:#2a1d00}
#shareDlg{max-width:min(92vw,820px)}#shareDlg img{display:block;width:100%;border-radius:10px;border:1px solid #2f4a10;margin:10px 0 14px}.dacts{display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap}
.trip .btn{margin-top:14px}
.big{font-size:30px;font-weight:700;color:var(--g);font-variant-numeric:tabular-nums}.med{font-size:24px;font-weight:700;color:var(--g);font-variant-numeric:tabular-nums}.lbl{color:#9aa3b5;font-size:13px}
.now{transition:opacity .4s,transform .4s}.now.off{opacity:.45;transform:scale(.99)}
.pulse{position:absolute;inset:0;border-radius:14px;pointer-events:none}.pulse.go{animation:pz .9s ease-out}
@keyframes pz{from{box-shadow:inset 0 0 0 0 rgba(118,185,0,.0),0 0 0 0 rgba(118,185,0,.75)}to{box-shadow:inset 0 0 40px rgba(118,185,0,.0),0 0 0 26px rgba(118,185,0,0)}}
.spark{position:absolute;width:6px;height:6px;border-radius:50%;background:var(--g);pointer-events:none;animation:fly .8s ease-out forwards}
@keyframes fly{to{transform:translate(var(--dx),var(--dy)) scale(.2);opacity:0}}
.breathe{animation:br 4s ease-in-out infinite}@keyframes br{50%{box-shadow:0 0 24px rgba(118,185,0,.18);border-color:#2f4a10}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px;margin-top:6px}.kv b{color:var(--g);font-variant-numeric:tabular-nums}
.ct{fill:#6b7285;font-size:9px}#today{transition:filter .4s}
.legend{text-align:right;font-size:12px;color:#6b7285;margin-top:4px}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 2px;vertical-align:middle}
#tip{position:fixed;display:none;background:#000;border:1px solid var(--g);font-size:12px;padding:4px 8px;border-radius:6px;pointer-events:none;z-index:9}
.idle{font-style:italic;color:#9aa3b5;margin-top:8px;min-height:1.2em}
table{width:100%;border-collapse:collapse;font-size:14px}td{padding:6px 8px;border-bottom:1px solid #1f232d;font-variant-numeric:tabular-nums}.bar div{height:10px;background:linear-gradient(90deg,#3f7a00,#76B900);border-radius:5px}.bar{width:45%}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}

.btn{background:#13200a;color:#c8f27a;border:1px solid #3f7a00;border-radius:10px;padding:7px 13px;font:600 13px system-ui,sans-serif;cursor:pointer}.btn:hover{background:#1d3310;border-color:var(--g)}.btn.red{background:#2a0d0d;color:#ffb3b3;border-color:#7a2020}
.acts{display:flex;gap:10px;align-items:center}
.r3{grid-template-columns:1fr 2fr}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 10px}.chip{font-size:12px;border:1px solid #2f4a10;background:#101a08;color:#c8f27a;border-radius:999px;padding:3px 10px}.chip.soon{border-style:dashed;color:#6b7285;border-color:#2a2f3a;background:none}
.rk{font-size:18px;width:34px;text-align:center}.lbm{font-weight:600}.lbs{font-size:11px;color:#6b7285}
.recs{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:12px}
.rec{background:linear-gradient(160deg,#16210c,#14171d 55%);border:1px solid #2f4a10;border-radius:14px;padding:12px;transition:transform .2s}.rec:hover{transform:translateY(-2px) rotate(-.4deg);box-shadow:0 0 18px rgba(118,185,0,.18)}
.ri{font-size:26px}.rt{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:#9aa3b5;margin-top:4px}.rv{font-size:22px;font-weight:800;color:var(--g);margin:4px 0;word-break:break-word}.rd{font-size:12px;color:#c9ced8}.rw{font-size:11px;color:#9fd14a;margin-top:6px}
.rec.locked{background:#111318;border:1px dashed #2a2f3a;filter:grayscale(1);opacity:.6}.rec.locked .rv{color:#4b5263}
.trip .med{font-size:28px}.th{font-size:12px;margin-top:10px;max-height:150px;overflow:auto}.th div{display:flex;justify-content:space-between;gap:8px;padding:3px 0;border-bottom:1px solid #1f232d}
dialog{background:#14171d;color:#e6e6e6;border:1px solid var(--g);border-radius:14px;padding:18px;max-width:380px}dialog::backdrop{background:rgba(0,0,0,.6)}dialog input{width:100%;margin:10px 0 14px;padding:8px;border-radius:8px;border:1px solid #2f4a10;background:#0b0d10;color:#e6e6e6;font-size:14px}
.ms{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 50% 45%,rgba(40,70,0,.92),rgba(0,0,0,.96));cursor:pointer;animation:fin .3s}.ms[hidden]{display:none}
@keyframes fin{from{opacity:0}}
#conf{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.msbox{position:relative;text-align:center;padding:20px;animation:pop .6s cubic-bezier(.2,1.6,.4,1)}@keyframes pop{from{transform:scale(.2) rotate(-12deg);opacity:0}}
.msg{width:180px;height:180px;border-radius:30px;animation:chomp 1s ease-in-out infinite;box-shadow:0 0 60px rgba(118,185,0,.6)}@keyframes chomp{50%{transform:scale(1.12) rotate(6deg)}}
.mskick{font:800 18px system-ui;letter-spacing:.3em;color:#9fd14a;margin-top:18px}.msbig{font:900 clamp(40px,9vw,110px)/1 system-ui;color:var(--g);text-shadow:0 0 30px rgba(118,185,0,.7);margin:10px 0}.mssub{font-size:20px;color:#e6e6e6;max-width:720px;margin:0 auto 22px}
.ms.huge .msbig{animation:shake .4s infinite,hue 2s linear infinite;font-size:clamp(50px,12vw,160px)}.ms.huge .msg{width:260px;height:260px;animation:spinz 1.6s ease-in-out infinite}
@keyframes shake{25%{transform:translate(3px,-2px) rotate(1deg)}75%{transform:translate(-3px,2px) rotate(-1deg)}}@keyframes hue{50%{color:#d4ff6b;text-shadow:0 0 60px #fff}}@keyframes spinz{50%{transform:scale(1.25) rotate(200deg)}100%{transform:rotate(360deg)}}
.orb{position:absolute;width:70px;height:70px;border-radius:16px;left:50%;top:40%;animation:orb 3s linear infinite;pointer-events:none}@keyframes orb{from{transform:rotate(var(--a)) translateX(min(42vw,460px)) rotate(calc(-1*var(--a)))}to{transform:rotate(calc(var(--a) + 360deg)) translateX(min(42vw,460px)) rotate(calc(-1*var(--a) - 360deg))}}
.seg{display:inline-flex;border:1px solid #2f4a10;border-radius:10px;overflow:hidden;margin-top:14px}.seg button{background:#0f1509;color:#9fd14a;border:0;border-right:1px solid #2f4a10;padding:7px 14px;font:600 13px system-ui,sans-serif;cursor:pointer}.seg button:last-child{border-right:0}.seg button.on{background:var(--g);color:#0b0d10}.seg button:hover:not(.on){background:#1d3310}
.fbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.fbar .muted{margin-top:14px}
.msvid{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.45;pointer-events:none}.msshade{position:absolute;inset:0;background:radial-gradient(circle at 50% 50%,rgba(0,0,0,.25),rgba(0,0,0,.85));pointer-events:none}
.tiny{background:none;border:none;color:#3f4757;font-size:11px;cursor:pointer}
@media (max-width:900px){.r3{grid-template-columns:1fr}}
@media (max-width:900px){.r1,.r2,.two{grid-template-columns:1fr}}
.seg{max-width:100%;overflow-x:auto}.seg button{white-space:nowrap}
@media (max-width:600px){body{padding:16px}.top{flex-wrap:wrap;gap:12px}h1{font-size:24px}.seg{display:flex}.seg button{flex:1;padding:7px 4px;font-size:12px}.rts{margin-left:0}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class=top><div class=brand><picture><source srcset="static/icon-128.png" media="(prefers-reduced-motion: reduce)"><img src="static/icon.webp" width=64 height=64 alt="Furious GPU chip devouring tokens"></picture><div><h1>DGX Token Annihilator Dashboard</h1><div class=muted id=sub></div></div></div><div class=acts><button class=btn id=share title="Preview a PNG of your carnage">📸 Share the carnage</button><span class=badge id=conn>Connecting to the extinction feed...</span></div></div>
<div class=fbar><div class=seg id=seg><button data-d=1>24 hours</button><button data-d=7>7 days</button><button data-d=30>30 days</button><button data-d=60>60 days</button><button data-d=90>90 days</button><button data-d=0>All time</button></div><span class=muted id=segSub></span><div class=rts id=rts></div></div>
<div class="row r1">
 <div class=card id=odoCard><h2 id=odoH>Tokens obliterated (all time, forever, no refunds)</h2><div class=odo id=odo></div><div class=words id=odoWords></div><div class=muted id=odoSub></div><div class=muted id=nextMs style="color:#9fd14a"></div>
  <div class=lbl id=rngLbl style="margin-top:14px;color:#9fd14a"></div><div class=grid style="margin-top:6px"><div><div class=lbl>Prompts devoured (input)</div><div class=med id=tin></div></div><div><div class=lbl>Vocabulary purged (output)</div><div class=med id=tout></div></div><div><div class=lbl>Lineages erased (requests)</div><div class=med id=nreq></div></div><div><div class=lbl>Avg extermination speed</div><div class=med id=avg></div></div></div></div>
 <div class=card><h2>Rate of extermination, right now</h2><svg viewBox="0 0 200 120" width=100%><path d="M20 105 A80 80 0 0 1 180 105" stroke="#232833" stroke-width=14 fill=none stroke-linecap=round /><path id=arc d="M20 105 A80 80 0 0 1 180 105" stroke="#76B900" stroke-width=14 fill=none stroke-linecap=round pathLength=100 stroke-dasharray="0 100"/><text x=100 y=92 text-anchor=middle font-size=30 font-weight=700 fill="#76B900" id=tps>0.0</text><text x=100 y=112 text-anchor=middle class=ct style="font-size:11px">tokens / sec (live estimate)</text></svg>
  <svg viewBox="0 0 240 40" width=100% height=40 preserveAspectRatio=none><polyline id=spk fill=none stroke="#76B900" stroke-width=1.5 points=""/></svg><div class=muted>last 60 seconds of mayhem</div></div>
 <div class=card id=savCard><h2>Cloud invoices eradicated <span id=savRng style="text-transform:none;letter-spacing:0;color:#9fd14a"></span></h2><div class=big id=sav></div><div class=muted id=savSub></div><div class=kv style="margin-top:10px" id=savList></div><div class=muted style="margin-top:8px">List prices from prices.json. The cloud is in mourning.</div></div>
</div>
<div class="row r2">
 <div class="card now off breathe" id=now><div class=pulse id=pulse></div><h2 id=nowH>Now annihilating</h2>
  <div class=kv><span>Model</span><b id=nm>–</b><span>Prompt devoured</span><b id=ni>–</b><span>Words unleashed</span><b id=no>–</b><span>Elapsed</span><b id=ne>–</b></div><div class=idle id=idle></div></div>
 <div class=card><h2>Year of mass extinction events</h2><svg id=cal width=100%></svg><div class=legend>Mild <span id=leg></span> Apocalyptic</div></div>
</div>
<div class="row r3">
 <div class="card trip"><h2>Current killing spree</h2><div class=muted id=tripSince></div>
  <div class=grid style="margin-top:10px;grid-template-columns:1fr 1fr"><div><div class=lbl>Tokens slain this spree</div><div class=med id=tripTok></div></div><div><div class=lbl>Victims (requests)</div><div class=med id=tripReq></div></div><div><div class=lbl>Saved this spree</div><div class=med id=tripSav></div></div><div><div class=lbl>Avg slaughter speed</div><div class=med id=tripAvg></div></div></div>
  <button class="btn red" id=tripBtn>☠ End this spree</button>
  <h2 style="margin-top:16px">The mass graves (past sprees)</h2><div class=th id=tripHist></div></div>
 <div class=card><h2>Model leaderboard: apex predators <span id=lbRng style="text-transform:none;letter-spacing:0;color:#9fd14a"></span></h2><div class=chips id=srcs></div><div class=scroll><table id=modT></table></div></div>
</div>
<h3>Trophy case: hall of total eradication (all time) <span id=recN style="color:#9fd14a;text-transform:none;letter-spacing:0"></span></h3><div class=recs id=recs></div>
<h3>Extinction events survived</h3><div class=chips id=msList></div>
<div><h3>Daily obliteration ledger <span id=logRng style="text-transform:none;letter-spacing:0;color:#9fd14a"></span></h3><div class=scroll id=ledger><table id=daysT></table></div></div>
<dialog id=tripDlg><form method=dialog><b style="color:var(--g)">End this killing spree?</b><div class=muted style="margin-top:6px">The current spree gets buried in the mass graves and a fresh one begins. All-time totals stay untouched.</div><input id=tripName maxlength=60 placeholder="Name this spree (optional), e.g. Fine-tune round 2"><div style="display:flex;gap:10px;justify-content:flex-end"><button class=btn value=cancel>Never mind</button><button class="btn red" value=ok>Bury it</button></div></form></dialog>
<dialog id=shareDlg><b style="color:var(--g)">Your carnage, ready to share</b><img id=shareImg alt="Share card: tokens obliterated, cloud invoices eradicated and trophies"><div class=dacts><button class=btn id=shareNative hidden>📤 Share</button><button class=btn id=shareCopy hidden>📋 Copy image</button><button class=btn id=shareDl>⬇ Download PNG</button><button class=btn id=shareX>Close</button></div><div class=muted id=shareMsg style="margin-top:8px;text-align:right"></div></dialog>
<div class=ms id=ms hidden><video class=msvid id=msVid muted loop playsinline preload=none poster="static/icon-180.png"><source src="static/splash.mp4" type="video/mp4"></video><div class=msshade></div><canvas id=conf></canvas><div id=orbs></div><div class=msbox><img src="static/icon.webp" class=msg alt=""><div class=mskick id=msKick></div><div class=msbig id=msBig></div><div class=mssub id=msSub></div><button class=btn id=msX>Bask in glory (dismiss)</button></div></div>
<p class=muted>No GPUs were harmed. Entire bloodlines of cloud invoices were. <button class=tiny id=msTest title="Preview the 1 billion celebration (does not count)">🎉 test the party</button></p>
<div id=tip></div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id),fmt=n=>Math.round(n||0).toLocaleString('en-US'),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const IDLE=["The GPU is napping. Somewhere, a cloud invoice sleeps uneasily.","Idle. Sharpening the tensor cores.","No tokens in sight. The silence is deafening (and free).","Waiting for prey. Feed me a prompt on port "+D.port+".","Plotting the next mass extinction event.","Resting between purges. The vocabulary trembles.","Dormant. Ancient token lineages enjoy a brief reprieve."];
let tin=D.tin,tout=D.tout,nreq=D.nreq,evOut=D.ev_out,evS=D.ev_s,est=0,streams=[],idleI=0;
$('sub').textContent="Every token erased here is a token the cloud will never bill you for. Total eradication, locally sourced.";
// odometer: grows a reel whenever the count gains a digit (always one spare zero up front), shrinks the reels to fit
// the card, and once even 18px reels won't fit it rolls in thousands/millions/... with a unit badge.
const ODO=$('odo'),UNITS=['','THOUSAND','MILLION','BILLION','TRILLION','QUADRILLION'];let DIG=0,SC=0,DH=58,reels=[],odoN=0;
const odoW=(dig,dw)=>dig*(dw+5)+Math.floor((dig-1)/3)*(dw/3+3);   // reel width + border + gap, plus the comma columns
const odoDw=(dig,avail)=>(avail-odoW(dig,0))/(odoW(dig,1)-odoW(dig,0));   // widest reel that fits
function odoBuild(dig,sc){DIG=dig;SC=sc;ODO.innerHTML=Array.from({length:dig},(_,i)=>(i&&(dig-i)%3==0?'<span class=sep>,</span>':'')+'<div class=dig><div>'+[...'0123456789'].map(d=>'<span>'+d+'</span>').join('')+'</div></div>').join('')+(sc?`<span class=unit>${UNITS[sc]}</span>`:'');reels=[...ODO.querySelectorAll('.dig>div')]}
function odoFit(n){const avail=ODO.parentElement.clientWidth-30||560;let sc=0,dig;
 for(;;sc++){dig=Math.max(9-3*sc,String(Math.floor(n/1e3**sc)).length+1);if(odoW(dig,18)+(sc?110:0)<=avail||sc>=UNITS.length-1)break}
 const dw=Math.max(18,Math.min(36,Math.floor(odoDw(dig,avail-(sc?110:0)))));DH=Math.round(dw*1.6);ODO.style.setProperty('--dw',dw+'px');ODO.style.setProperty('--dh',DH+'px');
 if(dig!=DIG||sc!=SC)odoBuild(dig,sc)}
const WORD=[[1e15,'quadrillion'],[1e12,'trillion'],[1e9,'billion'],[1e6,'million'],[1e3,'thousand']];
function words(n){for(const [v,w] of WORD)if(n>=v)return '≈ '+(n/v).toLocaleString('en-US',{maximumFractionDigits:n/v<10?2:1})+' '+w+' tokens';return ''}
function setOdo(n){n=Math.floor(n);odoN=n;if(String(Math.floor(n/1e3**SC)).length+1>DIG||!DIG)odoFit(n);
 const s=String(Math.floor(n/1e3**SC)).padStart(DIG,'0').slice(-DIG);reels.forEach((r,i)=>r.style.transform=`translateY(${-DH*s[i]}px)`);
 $('odoWords').textContent=words(n);ODO.title=fmt(n)+' tokens'}
if(window.ResizeObserver)new ResizeObserver(()=>{if(DIG){odoFit(odoN);setOdo(odoN)}}).observe($('odoCard'));
// calendar
const shades=["#1b1f27","#2a4d0a","#3f7a00","#5a9e00","#76B900"];$('leg').innerHTML=shades.map(s=>`<span class=sw style="background:${s}"></span>`).join('');
const per={};D.days.forEach(([d,a,b])=>per[d]=a+b);
const iso=d=>d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
const [ty,tm,td]=D.today.split('-').map(Number),today=new Date(ty,tm-1,td);let start=new Date(today);start.setDate(start.getDate()-364-today.getDay());
const days=[];for(let d=new Date(start);d<=today;d.setDate(d.getDate()+1))days.push({d:new Date(d),t:per[iso(d)]||0});
const T=days[days.length-1];let cutoff='';let mx=Math.max(1,...days.map(x=>x.t)),cal='';
days.forEach((x,i)=>{const c=Math.floor(i/7),r=x.d.getDay();if(r==0&&x.d.getDate()<=7)cal+=`<text x=${30+c*13} y=10 class=ct>${x.d.toLocaleString('en',{month:'short'})}</text>`;cal+=`<rect ${i==days.length-1?'id=today':''} data-i=${i} x=${30+c*13} y=${16+r*13} width=11 height=11 rx=2 />`});
cal+=[[1,'Mon'],[3,'Wed'],[5,'Fri']].map(([r,n])=>`<text x=0 y=${25+r*13} class=ct>${n}</text>`).join('');
const C=$('cal');C.setAttribute('viewBox',`0 0 ${40+Math.ceil(days.length/7)*13} 110`);C.innerHTML=cal;const rects=[...C.querySelectorAll('rect')];
const lvl=t=>!t?0:Math.min(4,1+Math.floor(4*t/mx-1e-9));
function paint(){rects.forEach(r=>{const x=days[r.dataset.i];r.setAttribute('fill',shades[lvl(x.t)]);r.setAttribute('opacity',cutoff&&iso(x.d)<cutoff?.25:1)})}
const tip=$('tip');rects.forEach(r=>{r.onmousemove=e=>{const x=days[r.dataset.i];tip.textContent=x.d.toDateString()+': '+fmt(x.t)+' tokens annihilated';tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px'};r.onmouseleave=()=>tip.style.display='none'});
paint();
// tables
function tables(){const DD=cutoff?D.days.filter(x=>x[0]>=cutoff):D.days,dm=Math.max(1,...DD.map(([,a,b])=>a+b));
 $('daysT').innerHTML='<tr class=muted><td>Day</td><td></td><td>In</td><td>Out</td></tr>'+(DD.map(([d,a,b])=>`<tr><td>${d}</td><td class=bar><div style="width:${(100*(a+b)/dm).toFixed(1)}%"></div></td><td>${fmt(a)}</td><td>${fmt(b)}</td></tr>`).join('')||`<tr><td colspan=4 class=muted>The silence before the storm. Nothing in this window. Point a client at the proxy and let the purge begin.</td></tr>`);
}
tables();
const priced=D.prices.filter(p=>p.input_per_mtok!=null&&p.output_per_mtok!=null),hero=priced[0];
$('savSub').textContent=hero?`kept out of ${hero.label}'s pockets ($${hero.input_per_mtok} / $${hero.output_per_mtok} per M)`:'no prices in prices.json';
const cost=(p,i,o)=>i/1e6*p.input_per_mtok+o/1e6*p.output_per_mtok,money=x=>'$'+(x<100?x.toFixed(4):x.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}));

// ===== leaderboard, records, trips, milestones, share card =====
let ST=D.st;
const when=ts=>ts?new Date(ts*1000).toLocaleString('en-US',{month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit'}):'';
const ago=ts=>{const s=Date.now()/1000-ts;return s<3600?Math.max(1,Math.round(s/60))+' min ago':s<86400?Math.round(s/3600)+' h ago':Math.round(s/86400)+' days ago'};
function renderLB(){const L=ST.leaderboard,md=['🥇','🥈','🥉'],mx=Math.max(1,...L.map(x=>x.tok));
 const have=new Set(ST.sources.map(s=>s.src));
 $('srcs').innerHTML=ST.sources.map(s=>`<span class=chip>${esc(s.label)}: ${fmt(s.tok)} tokens, ${fmt(s.n)} req</span>`).join('')+(ST.sources.length?'':'<span class="chip soon">No sources in this window. The arena is silent.</span>');
 $('modT').innerHTML='<tr class=muted><td>#</td><td>Model</td><td></td><td>Tokens</td><td>Req</td><td>Avg tok/s</td><td>Best tok/s</td><td>Biggest out</td></tr>'+(L.map((x,i)=>`<tr><td class=rk>${md[i]||i+1}</td><td><div class=lbm>${esc(x.model)}</div><div class=lbs>${esc(x.src)} · last seen ${ago(x.last)}</div></td><td class=bar style="width:18%"><div style="width:${(100*x.tok/mx).toFixed(1)}%"></div></td><td>${fmt(x.tok)}</td><td>${fmt(x.n)}</td><td>${x.avg?x.avg.toFixed(1):'–'}</td><td>${x.best?x.best.toFixed(1):'–'}</td><td>${fmt(x.maxout)}</td></tr>`).join('')||'<tr><td colspan=8 class=muted>No apex predators in this window. The arena awaits.</td></tr>')}
function renderRec(){const R=ST.records,got=R.filter(r=>r.value!=null).length;$('recN').textContent=`(${got} of ${R.length} trophies unlocked)`;
 $('recs').innerHTML=R.map(r=>r.value==null?`<div class="rec locked"><div class=ri>🔒</div><div class=rt>${esc(r.title)}</div><div class=rv>???</div><div class=rd>${esc(r.hint||'')}</div></div>`:`<div class=rec><div class=ri>${r.icon}</div><div class=rt>${esc(r.title)}</div><div class=rv>${esc(r.value)}</div><div class=rd>${esc(r.detail||'')}</div><div class=rw>🏆 ${esc(r.ts?when(r.ts):(r.date||''))}</div></div>`).join('')}
function renderTrip(){const t=ST.trip;if(!t)return;$('tripTok').textContent=fmt(t.tin+t.tout);$('tripReq').textContent=fmt(t.n);$('tripSav').textContent=hero?money(cost(hero,t.tin,t.tout)):'–';
 $('tripAvg').textContent=t.ev_s>0?(t.ev_out/t.ev_s).toFixed(1)+' tok/s':'–';
 $('tripSince').textContent='Rampaging since '+when(t.start_ts)+' ('+ago(t.start_ts)+')';
 $('tripHist').innerHTML=ST.trips.map(p=>`<div><span>${esc(p.name||'Unnamed spree')}<span class=muted> · ${new Date(p.start_ts*1000).toLocaleDateString('en-US',{month:'short',day:'numeric'})} to ${new Date(p.end_ts*1000).toLocaleDateString('en-US',{month:'short',day:'numeric'})}</span></span><b style="color:var(--g)">${fmt(p.tin+p.tout)} tok${hero?' · '+money(p.usd):''}</b></div>`).join('')||'<div class=muted>Empty. End a spree to bury it here.</div>'}
function renderRt(){$('rts').innerHTML=(ST.runtimes||[]).filter(r=>r.ok||r.used&&(r.error||r.ok===false)).map(r=>r.error?`<span class="chip busy" title="${esc(r.error)}">⚠ ${esc(r.label)}: ${esc(r.error)}</span>`
  :`<span class="chip${r.ok?'':' down'}" title="clients use :${r.listen}, forwarded to ${esc(r.up)}">${r.ok?'●':'○'} ${esc(r.label)} :${r.listen}${r.ok?'':' (nobody home)'}</span>`).join('')}
const bigNum=v=>v>=1e9?+(v/1e9).toFixed(1)+' BILLION':v>=1e6?+(v/1e6).toFixed(1)+' MILLION':fmt(v);
function msName(m){return m.kind=='usd'?'$'+fmt(m.value)+' saved':bigNum(m.value).toLowerCase()+' tokens'}
function renderMs(){$('msList').innerHTML=ST.milestones.map(m=>`<span class=chip title="${esc(when(m.reached_ts))}">${m.kind=='usd'?'💰':m.value>=1e9?'🌋':'🎉'} ${esc(msName(m))}</span>`).join('')||'<span class="chip soon">None yet. First stop: 1 million tokens. Then $100 saved.</span>'}
function nextMilestone(){const tok=tin+tout,sv=hero?cost(hero,tin,tout):0;const nt=tok<1e6?1e6:tok<1e7?1e7:(Math.floor(tok/1e8)+1)*1e8,ns=(Math.floor(sv/100)+1)*100;
 $('nextMs').textContent=`Next party: ${bigNum(nt).toLowerCase()} tokens (${fmt(nt-tok)} to go)`+(hero?` · $${fmt(ns)} saved ($${(ns-sv).toFixed(2)} to go)`:'')}
function renderAll(){renderLB();renderRec();renderTrip();renderMs();renderRt();nextMilestone()}
let rT=null,lastR=0;function refreshSoon(){if(rT)return;rT=setTimeout(()=>{rT=null;lastR=Date.now();refreshNow()},Math.max(3000,10000-(Date.now()-lastR)))}
// trip reset
$('tripBtn').onclick=()=>{$('tripName').value='';$('tripDlg').showModal()};
$('tripDlg').onclose=async()=>{if($('tripDlg').returnValue!='ok')return;try{await fetch('api/trip/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('tripName').value})});await refreshNow()}catch(_){alert('Reset failed. The gremlin is confused.')}};
// milestone celebrations
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches,Q=[];let showing=false,confId=0,seenMs=new Set(ST.milestones.filter(m=>m.acked).map(m=>m.key));
const TOKL=["The cloud providers have been notified. They are weeping.","Your GPU would like a raise.","Somewhere, a data center invoice just evaporated.","Entire token lineages, erased from the timeline.","Vocabulary purged. The dictionary has filed a missing persons report.","A mass extinction event, and the GPU did not even break a sweat.","Local inference: undefeated. Total eradication confirmed."],USDL=["That money stays in your pocket. The cloud stays hungry.","Treat yourself. Or buy more VRAM. Same thing.","Accountants hate this one weird trick: owning a GPU.","Cloud invoices eradicated down to the last decimal."];
function celebrate(m){if(m.key&&seenMs.has(m.key))return;if(m.key)seenMs.add(m.key);Q.push(m);if(!showing)nextMs()}
const vid=$('msVid');
function nextMs(){confId++;$('conf').getContext('2d').clearRect(0,0,1e5,1e5);const m=Q.shift();if(!m){showing=false;$('ms').hidden=true;$('orbs').innerHTML='';try{vid.pause()}catch(_){}return}showing=true;
 if(!RM){vid.muted=true;const pp=vid.play();if(pp)pp.catch(()=>{})}
 const huge=m.kind=='tok'&&m.value>=1e9&&m.value%1e9==0,mid=m.kind=='tok'&&m.value>=1e8;
 $('ms').className='ms'+(huge?' huge':'');$('msKick').textContent=huge?'⚠️ TOTAL ERADICATION ACHIEVED ⚠️':m.kind=='usd'?'CLOUD INVOICES ERADICATED':'MASS EXTINCTION EVENT';
 $('msBig').textContent=m.kind=='usd'?'$'+fmt(m.value)+' SAVED':bigNum(m.value)+' TOKENS';
 const L=m.kind=='usd'?USDL:TOKL;$('msSub').textContent=(huge?'A BILLION. With a B. Whole lineages of tokens, wiped from history. ':'')+L[Math.floor(Math.random()*L.length)]+(m.test?' (Test run, nothing counted.)':'');
 $('orbs').innerHTML=huge&&!RM?Array.from({length:8},(_,i)=>`<img class=orb src="static/icon.webp" style="--a:${i*45}deg;margin:-35px">`).join(''):'';
 $('ms').hidden=false;$('msX').focus();confetti(huge?700:mid?300:180,huge?14000:6000);
 if(m.key)fetch('api/milestones/ack',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:[m.key,...(m.also||[])]})}).then(refreshSoon).catch(()=>{})}
function confetti(n,ms){if(RM)return;const cv=$('conf'),x=cv.getContext('2d');cv.width=innerWidth;cv.height=innerHeight;const cols=['#76B900','#9fd14a','#ffffff','#3f7a00','#d4ff6b','#ffd400'];
 const P=Array.from({length:n},()=>({x:Math.random()*cv.width,y:-20-Math.random()*cv.height,vx:(Math.random()-.5)*3,vy:2+Math.random()*4,r:Math.random()*6.3,vr:(Math.random()-.5)*.3,s:5+Math.random()*8,c:cols[Math.random()*cols.length|0]}));
 const end=performance.now()+ms,id=confId;
 (function f(t){if(id!==confId)return;x.clearRect(0,0,cv.width,cv.height);const go=t<end;let alive=false;
  for(const p of P){p.x+=p.vx;p.y+=p.vy;p.r+=p.vr;if(p.y>cv.height+20){if(go){p.y=-20;p.x=Math.random()*cv.width}else continue}alive=true;x.save();x.translate(p.x,p.y);x.rotate(p.r);x.fillStyle=p.c;x.fillRect(-p.s/2,-p.s/4,p.s,p.s/2);x.restore()}
  if(alive)requestAnimationFrame(f)})(performance.now())}
$('msX').onclick=e=>{e.stopPropagation();nextMs()};$('ms').onclick=nextMs;addEventListener('keydown',e=>{if(showing&&(e.key=='Escape'||e.key=='Enter'))nextMs()});
$('msTest').onclick=()=>celebrate({kind:'tok',value:1e9,test:true});
// milestones reached while nobody was watching: party once for the biggest of each kind, then mark all seen
['tok','usd'].forEach(k=>{const p=ST.milestones.filter(m=>m.kind==k&&!m.acked);if(p.length){const top=p.reduce((a,b)=>b.value>a.value?b:a);celebrate({...top,also:p.map(m=>m.key)});p.forEach(m=>seenMs.add(m.key))}});
// share card
function fitText(x,t,w){if(x.measureText(t).width<=w)return t;while(t.length>1&&x.measureText(t+'…').width>w)t=t.slice(0,-1);return t+'…'}
$('share').onclick=async()=>{const W=1200,H=630,cv=document.createElement('canvas');cv.width=W;cv.height=H;const x=cv.getContext('2d'),G='#76B900',SF='system-ui,sans-serif';
 x.fillStyle='#0b0d10';x.fillRect(0,0,W,H);const g=x.createRadialGradient(150,100,10,150,100,700);g.addColorStop(0,'rgba(118,185,0,.28)');g.addColorStop(1,'rgba(118,185,0,0)');x.fillStyle=g;x.fillRect(0,0,W,H);
 x.strokeStyle=G;x.lineWidth=6;x.strokeRect(16,16,W-32,H-32);
 const img=new Image();img.src='static/icon-180.png';try{await img.decode();x.drawImage(img,52,44,120,120)}catch(_){}
 x.fillStyle=G;x.font='800 46px '+SF;x.fillText('DGX Token Annihilator',196,98);x.fillStyle='#9aa3b5';x.font='24px '+SF;x.fillText('Local mass extinction report · '+new Date().toLocaleDateString('en-US',{month:'long',day:'numeric',year:'numeric'}),196,138);
 x.font='600 20px '+SF;x.fillText('TOKENS OBLITERATED',56,228);x.fillStyle=G;let fs=84;do x.font=`800 ${fs}px ui-monospace,Menlo,monospace`;while(x.measureText(fmt(tin+tout)).width>600&&(fs-=4)>30);x.fillText(fmt(tin+tout),56,310);
 x.fillStyle='#9aa3b5';x.font='600 20px '+SF;x.fillText(fitText(x,'CLOUD INVOICES ERADICATED'+(hero?' (vs '+hero.label.toUpperCase()+')':''),600),56,370);x.fillStyle='#fff';x.font='800 54px '+SF;x.fillText(hero?money(cost(hero,tin,tout)):'–',56,428);
 const L=ST.leaderboard[0];x.fillStyle='#c9ced8';x.font='22px '+SF;
 [['Lineages erased',fmt(nreq)],['Avg speed',evS>0?(evOut/evS).toFixed(1)+' tok/s':'–'],['Top model',L?L.model:'–']].forEach(([a,b],i)=>{x.fillStyle='#9aa3b5';x.fillText(a,56,488+i*36);x.fillStyle=G;x.font='700 22px '+SF;x.fillText(fitText(x,b,330),300,488+i*36);x.font='22px '+SF});
 x.fillStyle='#9aa3b5';x.font='600 20px '+SF;x.fillText('TROPHY CASE',700,228);
 ST.records.filter(r=>r.value!=null).slice(0,6).forEach((r,i)=>{const y=270+i*52;x.font='26px '+SF;x.fillText(r.icon,700,y);x.fillStyle='#9aa3b5';x.font='16px '+SF;x.fillText(r.title.toUpperCase(),745,y-8);x.fillStyle=G;x.font='700 22px '+SF;x.fillText(fitText(x,r.value,380),745,y+16);x.fillStyle='#9aa3b5'});
 x.fillStyle='#6b7285';x.font='italic 18px '+SF;x.fillText('No GPUs were harmed. Entire bloodlines of cloud invoices were.',56,H-40);
 cv.toBlob(showShare,'image/png')};
// share preview: look first, then download / copy / native share (copy + share need https or localhost)
let shareBlob=null,shareUrl='';const shareName=()=>'token-carnage-'+D.today+'.png';
function showShare(b){shareBlob=b;if(shareUrl)URL.revokeObjectURL(shareUrl);shareUrl=URL.createObjectURL(b);$('shareImg').src=shareUrl;$('shareMsg').textContent='';
 const file=new File([b],shareName(),{type:'image/png'});
 $('shareNative').hidden=!(navigator.canShare&&navigator.canShare({files:[file]}));$('shareCopy').hidden=!(navigator.clipboard&&window.ClipboardItem);
 $('shareDlg').showModal()}
$('shareDl').onclick=()=>{const a=document.createElement('a');a.href=shareUrl;a.download=shareName();document.body.appendChild(a);a.click();a.remove()};
$('shareCopy').onclick=async()=>{try{await navigator.clipboard.write([new ClipboardItem({'image/png':shareBlob})]);$('shareMsg').textContent='Copied. Paste it somewhere the cloud can see it.'}catch(_){$('shareMsg').textContent='The clipboard refused. Download it instead.'}};
$('shareNative').onclick=async()=>{try{await navigator.share({files:[new File([shareBlob],shareName(),{type:'image/png'})],title:'Token carnage report'})}catch(_){}};
$('shareX').onclick=()=>$('shareDlg').close();$('shareDlg').onclick=e=>{if(e.target==$('shareDlg'))$('shareDlg').close()};
// ===== time filter (24h/7/30/60/90/all). The odometer follows the window (live increments land in RG); trophies stay all-time. =====
let RG=ST.rng,rDays=+(localStorage.getItem('tad.range')||0);
const RNAME=d=>d==1?'last 24 hours':d?'last '+d+' days':'all time';
function setRange(d,fetchIt){rDays=d;localStorage.setItem('tad.range',d);
 document.querySelectorAll('#seg button').forEach(b=>b.classList.toggle('on',+b.dataset.d==d));
 if(d){const c=new Date(today);c.setDate(c.getDate()-(d==1?1:d-1));cutoff=iso(c)}else cutoff='';   // rolling 24h touches yesterday too
 $('odoH').textContent=d?'Tokens obliterated ('+RNAME(d)+', no refunds)':'Tokens obliterated (all time, forever, no refunds)';
 $('rngLbl').textContent='In the '+RNAME(d)+':';['savRng','lbRng','logRng'].forEach(i=>$(i).textContent='('+RNAME(d)+')');
 $('segSub').textContent=d?'Showing the '+RNAME(d)+'. The trophies never forget.':'Showing everything since the first drop of blood.';
 paint();tables();draw();if(fetchIt)refreshNow()}
async function refreshNow(){try{ST=await(await fetch('api/stats?days='+rDays,{cache:'no-store'})).json();RG=ST.rng;renderAll();draw()}catch(_){}}
document.querySelectorAll('#seg button').forEach(b=>b.onclick=()=>setRange(+b.dataset.d,true));
renderAll();setRange(rDays,rDays!=0);
// state from SSE
let curTps=0,hist=Array(60).fill(0),secOut=0,prevEst=0,arr=[],primed=false;   // primed: false until the first SSE message after (re)connect sets the out_est baseline   // arr: [time, tokens] arrivals for the gauge
const WIN=3000;   // gauge = tokens/sec over a trailing 3 s wall-clock window
function onMsg(m){streams=m.streams||[];
 for(const f of (m.done||[])){tin+=f.in;tout+=f.out;nreq++;if(f.eval_s&&f.out){evOut+=f.out;evS+=f.eval_s;RG.ev_out+=f.out;RG.ev_s+=f.eval_s}RG.tin+=f.in;RG.tout+=f.out;RG.nreq++;T.t+=f.in+f.out;
  const dd=D.days.find(x=>x[0]==D.today);if(dd){dd[1]+=f.in;dd[2]+=f.out}else D.days.unshift([D.today,f.in,f.out]);
  if(ST.trip){ST.trip.tin+=f.in;ST.trip.tout+=f.out;ST.trip.n++;if(f.eval_s&&f.out){ST.trip.ev_out+=f.out;ST.trip.ev_s+=f.eval_s}}
  for(const k of (f.ms||[]))celebrate(k);
  showDone(f)}
 if(m.done&&m.done.length){tables();renderTrip();nextMilestone();refreshSoon();if(T.t>mx){mx=T.t;paint()}}
 const e=streams.reduce((a,s)=>a+s.out_est,0),de=primed?Math.max(0,e-prevEst):0;primed=true;secOut+=de;   // the first message carries every token the open streams produced BEFORE this page connected; counting it as a 3 s arrival spiked the gauge (80 then settling at 30)
 if(de)arr.push([performance.now(),de]);prevEst=streams.length?e:0;est=e;
 if(streams.length){const s=streams[streams.length-1];$('now').classList.remove('off','breathe');$('nowH').textContent=streams.length>1?`Now annihilating (${streams.length} at once!)`:'Now annihilating';$('nm').innerHTML=esc(s.model||'?')+(s.src?` <span class=lbs>on ${esc(s.src)}</span>`:'');$('ni').textContent='counting...';$('no').textContent='~'+fmt(s.out_est);$('ne').textContent=s.elapsed.toFixed(1)+' s';$('idle').textContent=''}
 draw()}
function showDone(f){$('nowH').textContent='Last annihilation complete';$('nm').innerHTML=esc(f.model||'?')+(f.src?` <span class=lbs>on ${esc(f.src)}</span>`:'');$('ni').textContent=fmt(f.in);$('no').textContent=fmt(f.out);$('ne').textContent=f.dur.toFixed(1)+' s';
 if(!matchMedia('(prefers-reduced-motion: reduce)').matches&&!document.hidden){const c=$('now'),p=$('pulse');p.classList.remove('go');void p.offsetWidth;p.classList.add('go');for(let i=0;i<14;i++){const s=document.createElement('i');s.className='spark';const a=i/14*6.28;s.style.left='85%';s.style.top='50%';s.style.setProperty('--dx',Math.cos(a)*60+'px');s.style.setProperty('--dy',Math.sin(a)*40+'px');c.appendChild(s);setTimeout(()=>s.remove(),850)}}}
function draw(){const live=streams.length>0,total=(rDays?RG.tin+RG.tout:tin+tout)+est;setOdo(total);   // window total + live increments (RG ticks on every finished request); all time = lifetime count
 $('odoSub').textContent=live?`including ~${fmt(est)} in flight (estimate until the stream finishes)`:'exact counts, straight from your inference backends';
 $('tin').textContent=fmt(RG.tin);$('tout').textContent=fmt(RG.tout);$('nreq').textContent=fmt(RG.nreq);$('avg').textContent=RG.ev_s>0?(RG.ev_out/RG.ev_s).toFixed(1)+' tok/s':'–';
 if(hero)$('sav').textContent=money(cost(hero,RG.tin,RG.tout));
 $('savList').innerHTML=D.prices.slice(1).map(p=>`<span>vs ${esc(p.label)}</span><b>${p.input_per_mtok==null?'n/a':money(cost(p,RG.tin,RG.tout))}</b>`).join('');
 const tr=$('today');tr.setAttribute('fill',shades[lvl(T.t+est)]);tr.style.filter=`drop-shadow(0 0 ${live?6:3}px #76B900)`;
 $('odoCard').classList.toggle('breathe',!live);$('savCard').classList.toggle('breathe',!live);
 if(!live){$('now').classList.add('off','breathe');if(!$('idle').textContent)$('idle').textContent=IDLE[idleI++%IDLE.length]}}
// animation loop: only gauge easing + sparkline; paused when hidden
let last=performance.now(),lastSec=last,running=!document.hidden;
function frame(t){if(!running)return;const gap=(t-last)/1000,dt=Math.min(.2,gap);last=t;
 // tokens/sec over the REAL elapsed time: rAF can be throttled to well under 1 fps, so a "1 s" bucket may span many seconds
 if(t-lastSec>=1000){const el=(t-lastSec)/1000;lastSec=t;hist.push(secOut/el);hist.shift();secOut=0;const m=Math.max(1,...hist);$('spk').setAttribute('points',hist.map((v,i)=>`${i*4},${38-v/m*34}`).join(' '))}
 // Rate over a trailing wall-clock window, not the last 1 s bucket or "is a stream open right now": SSE ticks land unevenly
 // in buckets and back-to-back requests leave brief no-stream gaps, which made the target flip to 0 and back (the sweep).
 while(arr.length&&t-arr[0][0]>WIN)arr.shift();
 const target=arr.reduce((a,x)=>a+x[1],0)/(WIN/1000);const k=gap>2?1:Math.min(1,dt*3);   // after a long stall, snap instead of sweeping
 const n=curTps+(target-curTps)*k;
 if(Math.abs(n-curTps)>.05||Math.abs(n)<.05&&curTps!==0){curTps=Math.abs(n)<.05?0:n;$('tps').textContent=curTps.toFixed(1);$('arc').setAttribute('stroke-dasharray',Math.min(100,curTps)+' 100')}
 requestAnimationFrame(frame)}
document.addEventListener('visibilitychange',()=>{running=!document.hidden;if(running){last=lastSec=performance.now();secOut=0;requestAnimationFrame(frame)}});   // drop tokens counted while hidden from the sparkline bucket; the gauge window is wall-clock, so it shows the real current rate
setInterval(()=>{if(!streams.length&&!document.hidden)$('idle').textContent=IDLE[idleI++%IDLE.length]},20000);
setInterval(()=>{if(!document.hidden)refreshNow()},60000);   // keeps runtime chips and "last seen" honest when idle
// SSE with 60s full-refresh fallback
let fb=null;function connect(){const es=new EventSource('events');
 es.onopen=()=>{primed=false;$('conn').textContent='Live: extinction feed connected';$('conn').classList.remove('off');if(fb){clearTimeout(fb);fb=null}};
 es.onmessage=e=>{try{onMsg(JSON.parse(e.data))}catch(_){}};
 es.onerror=()=>{$('conn').textContent='Feed lost. Refreshing every 60s like a caveman';$('conn').classList.add('off');if(!fb)fb=setTimeout(()=>location.reload(),60000)}}
if(window.EventSource)connect();else setTimeout(()=>location.reload(),60000);
draw();requestAnimationFrame(frame);
</script></body></html>"""

# --- llama.cpp: read-only tail of llama-server's own log (e.g. `llama-server ... 2>&1 | tee server.log`).
# Never connects to the server. Set LLAMA_LOG to the log path to enable; empty = disabled. ---
LLAMA_LOG = os.path.expanduser(os.environ.get("LLAMA_LOG", ""))
LLAMA_POLL_S = int(os.environ.get("LLAMA_POLL_S", 15))
LL_T = re.compile(rb'^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) ')
LL_PE = re.compile(rb'task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens')
LL_EV = re.compile(rb'task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens')
LL_TOT = re.compile(rb'task (\d+) \|\s+total time =\s*([\d.]+) ms')
LL_REL = re.compile(rb'task (\d+) \| stop processing: n_tokens = (\d+)')
LL_LOAD = re.compile(rb"load_model: loading model '([^']+)'")

def ll_state(c, path, ino=None, off=None):
    c.execute("CREATE TABLE IF NOT EXISTS llama_state(path TEXT PRIMARY KEY, ino INTEGER, off INTEGER)")
    if ino is None:
        r = c.execute("SELECT ino, off FROM llama_state WHERE path=?", (path,)).fetchone()
        return r if r else (None, 0)
    c.execute("INSERT OR REPLACE INTO llama_state(path,ino,off) VALUES(?,?,?)", (path, ino, off))

def llama_scan(path=None, anchors=None):
    """Read new llama-server log lines and log one row per finished request. anchors: optional list of
    session start epochs (one per server start in the file); otherwise times are inferred from the file mtime."""
    path = path or LLAMA_LOG
    if not path: return 0
    try: st = os.stat(path)
    except OSError: return 0
    c = db(); ino, off = ll_state(c, path); c.close()
    if ino != st.st_ino or st.st_size < off: off = 0
    if st.st_size == off: return 0
    with open(path, "rb") as f:
        f.seek(off); data = f.read()
    end = data.rfind(b"\n")
    if end < 0: return 0
    data = data[:end + 1]
    sess, model, pend, rows, pos, last_rel = 0, os.environ.get("LLAMA_MODEL", "llama.cpp"), {}, [], off, {}
    models = {}
    for line in data.splitlines(keepends=True):
        lo = pos; pos += len(line)
        m = LL_T.match(line)
        rel = (int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6) if m else None
        lm = LL_LOAD.search(line)
        if lm:
            if rows or pend: sess += 1
            model = os.path.splitext(os.path.basename(lm.group(1).decode("utf-8", "replace")))[0]
            pend = {}
        models[sess] = model
        if rel is not None: last_rel[sess] = rel
        for rx, k in ((LL_PE, "pe"), (LL_EV, "ev"), (LL_TOT, "tot")):
            x = rx.search(line)
            if x: pend.setdefault(x.group(1), {})[k] = x.groups()[1:]; break
        else:
            x = LL_REL.search(line)
            if x and x.group(1) in pend and "ev" in pend[x.group(1)]:
                t = pend.pop(x.group(1)); ev_ms, ev = float(t["ev"][0]), int(t["ev"][1])
                rows.append({"sess": sess, "rel": rel, "model": models[sess], "out": ev, "inp": max(0, int(x.group(2)) - ev),
                             "dur": float(t["tot"][0]) / 1e3 if "tot" in t else None, "eval_s": ev_ms / 1e3, "lo": lo, "end": pos})
    if not rows: return 0
    # session anchors: explicit (backfill) or chained back from file mtime (live)
    if anchors:
        base = len(anchors) - 1 - sess
        anc = {s: anchors[base + s] for s in range(sess + 1) if 0 <= base + s < len(anchors)}
    else:
        anc, t = {}, st.st_mtime
        for s in range(sess, -1, -1):
            anc[s] = t - last_rel.get(s, 0); t = anc[s]
    new_tot, fin = [], []
    for r in rows:
        ts = anc.get(r["sess"], time.time()) + (r["rel"] or 0)
        ms = log_row(ts=min(ts, time.time()), model=r["model"], endpoint="llama-server", inp=r["inp"], out=r["out"], dur=r["dur"],
                     eval_s=r["eval_s"], status=200, source="llamacpp", dedupe="llamacpp:%d:%d" % (st.st_ino, r["lo"]))
        fin.append({"model": r["model"], "endpoint": "llama-server", "in": r["inp"], "out": r["out"], "dur": round(r["dur"] or 0, 2), "eval_s": r["eval_s"], "src": "llamacpp", "ms": ms})
    c = db(); ll_state(c, path, st.st_ino, rows[-1]["end"]); c.commit(); c.close()
    if not anchors:
        with LIVE_LOCK:
            for x in fin[-20:]: live_bump(); DONE.append((LIVE_VER[0], x))
            del DONE[:-20]
    return len(rows)

def llama_loop():
    while True:
        try: llama_scan()
        except Exception as e: print("llama scan error:", e, file=sys.stderr)
        time.sleep(LLAMA_POLL_S)

def llama_backfill():
    """One-shot import of everything already in LLAMA_LOG. Optional LLAMA_STARTS: comma-separated epoch
    seconds of each server start in the file, for exact timestamps (else inferred from the file mtime)."""
    anchors = [float(x) for x in os.environ.get("LLAMA_STARTS", "").split(",") if x.strip()] or None
    print("llama rows logged:", llama_scan(anchors=anchors))


class TS(ThreadingMixIn, http.server.HTTPServer): daemon_threads = True; allow_reuse_address = True

if __name__ == "__main__":
    init_db()
    if "--backfill" in sys.argv:
        backfill(float(sys.argv[2]) if len(sys.argv) > 2 else None); sys.exit()
    if "--llama-backfill" in sys.argv:
        llama_backfill(); sys.exit()
    if LLAMA_LOG:
        threading.Thread(target=llama_loop, daemon=True).start(); print("tailing llama.cpp log", LLAMA_LOG, flush=True)
    for spec in filter(None, (x.strip() for x in proxy_specs().split(","))):
        src, _, rest = spec.partition(":"); lp, _, upstream = rest.partition("="); uh, _, upp = upstream.rpartition(":")
        H = type("Proxy_" + src, (Proxy,), {"UPH": uh or "127.0.0.1", "UPP": int(upp), "SRC": src})
        start_proxy(H, src, int(lp), (H.UPH, H.UPP), auto=auto_proxies())
    print(f"dashboard http://{DASH_BIND}:{DASH_PORT}", flush=True)
    if OLLAMA_PROXY:
        if PROXY_SOURCE: Proxy.SRC = "proxy" if PROXY_SOURCE == "ollama" else PROXY_SOURCE
        start_proxy(Proxy, Proxy.SRC, PROXY_PORT, (UP_HOST, UP_PORT))
    threading.Thread(target=runtimes_loop, daemon=True).start()
    TS((DASH_BIND, DASH_PORT), Dash).serve_forever()
