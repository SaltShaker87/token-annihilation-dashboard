#!/usr/bin/env python3
"""Stdlib checks for the Ollama, vLLM and llama.cpp token counting. Run: python3 test_annihilator.py"""
import http.client, http.server, json, os, socket, sqlite3, subprocess, sys, tempfile, threading, time, unittest
from socketserver import ThreadingMixIn

os.environ.setdefault("DASH_BIND", "127.0.0.1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import token_annihilator as ta


def parse(body, req=b"{}"):
    if isinstance(body, str): body = body.encode()
    return ta.parse_usage("/test", req, body)


class ParseTests(unittest.TestCase):
    def test_ollama_ndjson_and_pretty_json(self):
        body = '\n'.join(['{"model":"llama3","done":false,"message":{"content":"Hi"}}',
                          '{"model":"llama3","done":true,"prompt_eval_count":10,"eval_count":20,"eval_duration":2000000000}'])
        self.assertEqual(parse(body), ("llama3", 10, 20, 2.0))
        self.assertEqual(parse('{\n  "model": "m",\n  "prompt_eval_count": 1,\n  "eval_count": 2\n}\n')[:3], ("m", 1, 2))

    def test_openai_sse_ignores_null_usage_and_done(self):
        body = ('data: {"model":"streamy","usage":null}\n\n'
                'data: {"model":"streamy","usage":{"prompt_tokens":3,"completion_tokens":4}}\n\ndata: [DONE]\n')
        self.assertEqual(parse(body, b'{"model":"from-req"}'), ("streamy", 3, 4, None))

    def test_llamacpp_openai_timings(self):
        body = '{"model":"gguf","usage":{"prompt_tokens":8,"completion_tokens":6},"timings":{"predicted_ms":250}}'
        self.assertEqual(parse(body), ("gguf", 8, 6, 0.25))

    def test_embeddings_use_the_usage_object_not_the_vectors(self):
        body = '{"model":"e","data":[{"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":6,"total_tokens":6}}'
        self.assertEqual(parse(body), ("e", 6, 0, None))   # our version records 0 output tokens

    def test_responses_api_usage(self):
        self.assertEqual(parse('{"model":"r","usage":{"input_tokens":9,"output_tokens":11,"total_tokens":20}}'), ("r", 9, 11, None))
        body = ('event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}\n\n'
                'event: response.completed\ndata: {"type":"response.completed","response":{"model":"rs","usage":{"input_tokens":2,"output_tokens":3}}}\n\n')
        self.assertEqual(parse(body), ("rs", 2, 3, None))
        self.assertEqual(ta.add_usage_opt("/v1/responses", b'{"stream":true}'), b'{"stream":true}')   # never inject there

    def test_request_model_fallback(self):
        self.assertEqual(parse('{"id":"x"}', b'{"model":"asked"}'), ("asked", None, None, None))

    def test_stream_usage_injection(self):
        out = json.loads(ta.add_usage_opt("/v1/chat/completions", b'{"stream":true}'))
        self.assertEqual(out["stream_options"], {"include_usage": True})
        self.assertEqual(ta.add_usage_opt("/v1/chat/completions", b'{"stream":true,"stream_options":{}}'), b'{"stream":true,"stream_options":{}}')
        self.assertEqual(ta.add_usage_opt("/api/chat", b'{"stream":true}'), b'{"stream":true}')
        self.assertEqual(ta.add_usage_opt("/v1/chat/completions", b'not json'), b'not json')


class LlamaLogTests(unittest.TestCase):
    def test_log_tail_counts_requests(self):
        tmp = tempfile.mkdtemp(); log = os.path.join(tmp, "server.log")
        with open(log, "w") as f:
            f.write("llama_model_loader: load_model: loading model '/models/my-gguf.gguf'\n"
                    "slot print_timing: id  0 | task 7 | prompt eval time =      10.00 ms /    12 tokens\n"
                    "slot print_timing: id  0 | task 7 |        eval time =     200.00 ms /    30 tokens\n"
                    "slot print_timing: id  0 | task 7 |       total time =     210.00 ms\n"
                    "slot      release: id  0 | task 7 | stop processing: n_tokens = 42, truncated = 0\n")
        old = ta.DB
        try:
            ta.DB = os.path.join(tmp, "s.db"); ta.init_db()
            self.assertEqual(ta.llama_scan(log), 1)
            self.assertEqual(ta.llama_scan(log), 0)   # no double counting
            r = sqlite3.connect(ta.DB).execute("SELECT source, model, input_tokens, output_tokens, eval_s FROM requests").fetchall()
            self.assertEqual(r, [("llamacpp", "my-gguf", 12, 30, 0.2)])
        finally: ta.DB = old


class MigrationTests(unittest.TestCase):
    def test_older_build_rows_get_their_real_runtime(self):
        tmp = tempfile.mkdtemp(); old = ta.DB
        try:
            ta.DB = os.path.join(tmp, "s.db"); c = ta.db()
            c.execute("ALTER TABLE requests ADD COLUMN backend TEXT")
            c.executemany("INSERT INTO requests(ts,model,input_tokens,output_tokens,source,backend) VALUES(?,?,?,?,?,?)",
                          [(1, "qwen", 1, 2, "proxy", "vllm"), (2, "small", 1, 2, "proxy", "ollama"), (3, "gguf", 1, 2, "proxy", "llamacpp"), (4, None, None, None, "backfill", "ollama")])
            c.commit(); c.close()
            ta.init_db(); ta.init_db()   # idempotent
            rows = sqlite3.connect(ta.DB).execute("SELECT model, source FROM requests ORDER BY ts").fetchall()
            self.assertEqual(rows, [("qwen", "vllm"), ("small", "proxy"), ("gguf", "llamacpp"), (None, "backfill")])
        finally: ta.DB = old

    def test_older_build_env_vars_still_start_proxies(self):
        saved = (ta.OPENAI_PROXIES, ta.ENGINES, ta.LLAMA_LOG, dict(os.environ))
        try:
            ta.OPENAI_PROXIES, ta.ENGINES, ta.LLAMA_LOG = None, ["ollama", "vllm", "llamacpp"], ""
            os.environ.update(VLLM_PROXY_PORT="9001", VLLM_UPSTREAM_PORT="9000")
            self.assertEqual(ta.proxy_specs(), "vllm:9001=127.0.0.1:9000,llamacpp:8081=127.0.0.1:8080")
            ta.LLAMA_LOG = "/tmp/x.log"; self.assertEqual(ta.proxy_specs(), "vllm:9001=127.0.0.1:9000")   # log tail counts llama.cpp
            ta.OPENAI_PROXIES = ""; self.assertEqual(ta.proxy_specs(), "")   # explicitly empty = none
            ta.OPENAI_PROXIES = None; self.assertFalse(ta.auto_proxies())   # VLLM_* set: the user asked for it, start now
            for k in ta.LEGACY_ENV: os.environ.pop(k, None)
            self.assertTrue(ta.auto_proxies())   # nothing configured: defaults wait for detection
        finally:
            ta.OPENAI_PROXIES, ta.ENGINES, ta.LLAMA_LOG = saved[:3]; os.environ.clear(); os.environ.update(saved[3])


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _reply(self, body, status=200):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        g = getattr(self.server, "gets", {}).get(self.path)
        self._reply(g) if g else self._reply('{"error":"nope"}', 404)
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)); self.server.seen.append((self.path, body))
        q = self.server.queue.get(self.path.split("?")[0])
        item = q.pop(0) if q else '{"ok":true}'
        if isinstance(item, list):   # a slow token stream, one SSE event at a time
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for part in item:
                p = part.encode(); self.wfile.write(b"%x\r\n%s\r\n" % (len(p), p)); self.wfile.flush(); time.sleep(0.05)
            self.wfile.write(b"0\r\n\r\n"); return
        self._reply(item)


class Server(ThreadingMixIn, http.server.HTTPServer): daemon_threads = True; allow_reuse_address = True


def req(port, method, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
    r = c.getresponse(); d = r.read(); c.close(); return r.status, d


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="annihilator-"); cls.db = os.path.join(cls.tmp, "stats.db")
        cls.up = {"ollama": 19134, "vllm": 19100, "llamacpp": 19180}
        cls.px = {"ollama": 18135, "vllm": 18136, "llamacpp": 18137}; cls.dash = 18180
        queues = {
            "ollama": {"/api/chat": ['{"model":"llama3","done":false}\n{"model":"llama3","done":true,"prompt_eval_count":10,"eval_count":20,"eval_duration":2000000000}']},
            "vllm": {"/v1/chat/completions": [
                '{"model":"meta-llama","usage":{"prompt_tokens":5,"completion_tokens":7}}',
                'data: {"model":"streamy","usage":null}\n\ndata: {"model":"streamy","usage":{"prompt_tokens":3,"completion_tokens":4}}\n\ndata: [DONE]\n',
                '{"id":"blank"}',
                ['data: {"model":"slowpoke","choices":[{"delta":{"content":"a"}}]}\n\n'] * 4 +
                ['data: {"model":"slowpoke","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":5}}\n\n', 'data: [DONE]\n\n']]},
            "llamacpp": {"/v1/chat/completions": ['{"model":"gguf","usage":{"prompt_tokens":8,"completion_tokens":6},"timings":{"predicted_ms":250}}']},
        }
        gets = {"ollama": {"/api/version": '{"version":"0.12.0"}', "/api/tags": '{"models":[]}'},
                "vllm": {"/v1/models": '{"object":"list","data":[{"id":"qwen","owned_by":"vllm"}]}'},
                "llamacpp": {"/v1/models": '{"object":"list","data":[{"id":"gguf","owned_by":"llamacpp"}]}'}}
        cls.servers = {}
        for k, port in cls.up.items():
            s = Server(("127.0.0.1", port), Handler); s.queue = queues[k]; s.seen = []; s.gets = gets[k]
            threading.Thread(target=s.serve_forever, daemon=True).start(); cls.servers[k] = s
        env = dict(os.environ, STATS_DB=cls.db, DASH_BIND="127.0.0.1", DASH_PORT=str(cls.dash), PROXY_BIND="127.0.0.1",
                   PROXY_PORT=str(cls.px["ollama"]), UPSTREAM_HOST="127.0.0.1", UPSTREAM_PORT=str(cls.up["ollama"]), LLAMA_LOG="",
                   OPENAI_PROXIES="vllm:%d=127.0.0.1:%d,llamacpp:%d=127.0.0.1:%d" % (cls.px["vllm"], cls.up["vllm"], cls.px["llamacpp"], cls.up["llamacpp"]))
        cls.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "token_annihilator.py")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        cls.boot = ""; deadline = time.time() + 8
        while time.time() < deadline and "Ollama proxy" not in cls.boot: cls.boot += cls.proc.stdout.readline()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try: cls.proc.wait(timeout=3)
        except subprocess.TimeoutExpired: cls.proc.kill()
        cls.proc.stdout.close()
        for s in cls.servers.values(): s.shutdown(); s.server_close()

    def test_startup_lists_three_proxies(self):
        self.assertIn("vLLM proxy 127.0.0.1:18136 -> 127.0.0.1:19100", self.boot)
        self.assertIn("llama.cpp proxy 127.0.0.1:18137 -> 127.0.0.1:19180", self.boot)
        self.assertIn("Ollama proxy 127.0.0.1:18135 -> 127.0.0.1:19134", self.boot)

    def test_counts_each_engine_and_serves_dashboard(self):
        self.assertEqual(req(self.px["ollama"], "POST", "/api/chat", {"model": "llama3"})[0], 200)
        self.assertEqual(req(self.px["vllm"], "POST", "/v1/chat/completions", {"model": "meta-llama"})[0], 200)
        st, streamed = req(self.px["vllm"], "POST", "/v1/chat/completions", {"model": "streamy", "stream": True})
        self.assertIn(b"[DONE]", streamed)
        self.assertIn(b'"include_usage": true', self.servers["vllm"].seen[-1][1])   # usage was requested upstream
        self.assertEqual(req(self.px["vllm"], "POST", "/v1/chat/completions", {"model": "a<b>"})[0], 200)
        self.assertEqual(req(self.px["llamacpp"], "POST", "/v1/chat/completions", {"model": "gguf"})[0], 200)
        st, tags = req(self.px["ollama"], "GET", "/api/tags"); self.assertEqual(st, 200)   # untracked passthrough
        deadline = time.time() + 3; rows = []
        while time.time() < deadline and len(rows) < 5:
            rows = sqlite3.connect(self.db).execute("SELECT source, model, input_tokens, output_tokens, eval_s FROM requests ORDER BY id").fetchall(); time.sleep(0.05)
        self.assertEqual(rows, [("proxy", "llama3", 10, 20, 2.0), ("vllm", "meta-llama", 5, 7, None), ("vllm", "streamy", 3, 4, None),
                                ("vllm", "a<b>", None, None, None), ("llamacpp", "gguf", 8, 6, 0.25)])
        st, page = req(self.dash, "GET", "/"); self.assertEqual(st, 200); text = page.decode()
        self.assertIn('src="static/icon.webp"', text); self.assertIn("static/splash.mp4", text)
        self.assertIn("esc(x.model)", text)   # model names are HTML-escaped client-side; embedded JSON escapes "</"
        self.assertIn("id=updBtn", text); self.assertIn(">Update</button>", text)
        self.assertIn('"version": "0.0.0"', text); self.assertIn('"boot":', text)
        st, img = req(self.dash, "GET", "/static/icon-32.png"); self.assertTrue(img.startswith(b"\x89PNG"))
        j = json.loads(req(self.dash, "GET", "/api/stats")[1])
        labels = {s["label"]: s["tok"] for s in j["sources"]}
        self.assertEqual(labels, {"Ollama": 30, "vLLM": 19, "llama.cpp": 14})
        self.assertEqual(len(j["records"]), 27)
        self.assertEqual(j["rng"]["nreq"], 5)
        j7 = json.loads(req(self.dash, "GET", "/api/stats?days=7")[1]); self.assertEqual(j7["rng"]["nreq"], 5)
        junk = json.loads(req(self.dash, "GET", "/api/stats?days=%3Cscript%3E")[1]); self.assertIsNone(junk["rng"]["since"])
        rts = {r["label"]: r for r in j["runtimes"]}
        self.assertEqual(set(rts), {"Ollama", "vLLM", "llama.cpp"}); self.assertTrue(all(r["ok"] for r in rts.values()))

    def test_probe_tells_runtimes_apart(self):
        self.assertEqual(ta.probe("127.0.0.1", self.up["ollama"]), "ollama")
        self.assertEqual(ta.probe("127.0.0.1", self.up["vllm"]), "vllm")
        self.assertEqual(ta.probe("127.0.0.1", self.up["llamacpp"]), "llamacpp")
        self.assertIsNone(ta.probe("127.0.0.1", 1))

    def test_default_proxy_pointed_at_vllm_is_labelled_vllm(self):
        saved = list(ta.RUNTIMES)
        try:
            ta.RUNTIMES[:] = [{"src": "proxy", "listen": 1, "up": ("127.0.0.1", self.up["vllm"]), "cls": ta.Proxy, "started": True}]
            ta.check_runtimes(); self.assertEqual(ta.Proxy.SRC, "vllm")
            ta.RUNTIMES[0]["up"] = ("127.0.0.1", self.up["ollama"])
            ta.check_runtimes(); self.assertEqual(ta.Proxy.SRC, "proxy")
        finally: ta.RUNTIMES[:] = saved; ta.Proxy.SRC = "proxy"

    def test_default_proxy_waits_for_the_right_runtime(self):
        saved, bind = list(ta.RUNTIMES), ta.PROXY_BIND
        try:
            ta.RUNTIMES[:] = []; ta.PROXY_BIND = "127.0.0.1"
            H = type("Proxy_vllm", (ta.Proxy,), {"UPH": "127.0.0.1", "UPP": self.up["llamacpp"], "SRC": "vllm"})
            ta.start_proxy(H, "vllm", 18190, ("127.0.0.1", self.up["llamacpp"]), auto=True)   # something else answers there
            ta.check_runtimes(); rt = ta.RUNTIMES[0]
            self.assertFalse(rt["started"]); self.assertFalse(rt["ok"])
            with self.assertRaises(OSError): req(18190, "GET", "/v1/models")   # port left alone
            rt["up"] = ("127.0.0.1", self.up["vllm"]); ta.check_runtimes()   # vLLM shows up later
            self.assertTrue(rt["started"]); self.assertTrue(rt["ok"])
        finally:
            for rt in ta.RUNTIMES:
                if rt.get("srv"): rt["srv"].shutdown(); rt["srv"].server_close()
            ta.RUNTIMES[:] = saved; ta.PROXY_BIND = bind

    def test_streamed_vllm_gets_speed(self):
        while self.servers["vllm"].queue["/v1/chat/completions"][:-1]: self.servers["vllm"].queue["/v1/chat/completions"].pop(0)   # if run alone
        st, body = req(self.px["vllm"], "POST", "/v1/chat/completions", {"model": "slowpoke", "stream": True})
        self.assertEqual(st, 200); self.assertIn(b"[DONE]", body)
        deadline = time.time() + 3; r = None
        while time.time() < deadline and not r:
            r = sqlite3.connect(self.db).execute("SELECT input_tokens, output_tokens, eval_s FROM requests WHERE model='slowpoke'").fetchone(); time.sleep(0.05)
        self.assertEqual(r[:2], (2, 5)); self.assertTrue(0.1 < r[2] < 2, r)



class MascotTests(unittest.TestCase):
    def test_brand_uses_the_inhaling_robot(self):
        p = ta.PAGE
        self.assertIn('id=mascotVid', p)
        self.assertIn('id=mascotStill', p)
        self.assertIn('<video id=mascotVid muted playsinline loop preload=none poster="static/mascot-idle.png"', p)
        self.assertLess(p.index('static/mascot-inhale.webm'), p.index('static/mascot-inhale.mp4'))
        self.assertIn('type="video/webm"', p)
        self.assertIn('type="video/mp4"', p)
        self.assertIn('srcset="static/mascot-idle.webp"', p)
        self.assertIn('alt="Angry chip robot inhaling tokens"', p)
        self.assertIn('mascotSync()', p)
        self.assertIn('setTimeout(go,1500)', p)
        self.assertIn('if(mascotOn)', p)   # an SSE tick must not restart the loop at frame 0
        self.assertIn('mascotIdle(true)', p)   # a dropped feed goes idle immediately
        self.assertIn(".brand .mascot video{display:none!important}", p)
        self.assertNotIn("static/icon-128.png", p)
        self.assertIn('src="static/icon.webp"', p)   # milestone box still uses the old art
        self.assertIn('class=orb src="static/icon.webp"', p)
        self.assertIn('poster="static/icon-180.png"', p)   # share / celebration video untouched

    def test_mascot_files_are_served_with_their_types(self):
        srv = ta.TS(("127.0.0.1", 0), ta.Dash)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
        port = srv.server_address[1]
        expect = [
            ("/static/mascot-idle.png", "image/png", b"\x89PNG"),
            ("/static/mascot-idle.webp", "image/webp", b"RIFF"),
            ("/static/mascot-inhale.webp", "image/webp", b"RIFF"),
            ("/static/mascot-inhale.webm", "video/webm", b"\x1a\x45\xdf\xa3"),
            ("/static/mascot-inhale.mp4", "video/mp4", b"ftyp"),
        ]
        for path, ctype, magic in expect:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", path)
            r = c.getresponse(); body = r.read(); c.close()
            self.assertEqual(r.status, 200, path)
            self.assertEqual(r.getheader("Content-Type"), ctype, path)
            self.assertIn(magic, body[:16], (path, body[:16]))
            if ctype == "image/webp":
                self.assertIn(b"WEBP", body[:16], path)



class OdometerTests(unittest.TestCase):
    # runs the page's real odometer code in node against a fake DOM and reads back what would be drawn
    def run_odo(self, seq, width=900):
        import shutil
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        p = ta.PAGE
        js = p[p.index("const ODO=$('odo')"):p.index("if(window.ResizeObserver)")]
        stub = """
const W=%d;let HTML='';const cls=new Set();
const odo={style:{setProperty(){}},parentElement:{clientWidth:W},classList:{toggle(c,on){on?cls.add(c):cls.delete(c)}},
 set innerHTML(v){HTML=v},get innerHTML(){return HTML},title:'',
 querySelectorAll(){return Array.from({length:(HTML.match(/class=dig>/g)||[]).length},()=>({style:{},parentElement:{classList:{add(){},remove(){}},offsetWidth:0}}))}};
const words_el={textContent:''};const $=id=>id=='odo'?odo:words_el;const fmt=n=>Math.round(n).toLocaleString('en-US');
""" % width
        drive = """const out=[];for(const n of %s){setOdo(n);const shown=reels.map(r=>Math.round(-parseFloat(r.style.transform.slice(11))/DH)).join('');
 const k=HTML.indexOf('<sup>'),sup=k<0?null:parseInt(HTML.slice(k+5));
 out.push({reels:reels.length,shown,sci:cls.has('sci'),exp:sup,commas:HTML.split('>,<').length-1,title:odo.title})}
console.log(JSON.stringify(out))""" % json.dumps(seq)
        out = subprocess.run([node, "-e", stub + js + drive], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_reels_show_exactly_the_digits_of_the_number(self):
        nums = [0, 7, 12935412, 999999999, 1000000000, 1234567890, 123456789012]
        for n, r in zip(nums, self.run_odo(nums)):
            self.assertFalse(r["sci"], n)
            self.assertEqual(r["shown"], str(n))           # no leading zeros, nothing cut off
            self.assertEqual(r["commas"], (len(str(n)) - 1) // 3)

    def test_count_shrinks_back_when_the_range_gets_smaller(self):
        r = self.run_odo([98765432109, 12935412, 5])
        self.assertEqual([x["shown"] for x in r], ["98765432109", "12935412", "5"])

    def test_huge_numbers_switch_to_scientific_notation(self):
        self.assertFalse(self.run_odo([4512345678901])[0]["sci"])   # 4.5T still fits as reels
        r = self.run_odo([45123456789012, 1234567890123456])
        self.assertEqual((r[0]["sci"], r[0]["shown"], r[0]["exp"]), (True, "4512", 13))
        self.assertEqual((r[1]["sci"], r[1]["shown"], r[1]["exp"]), (True, "1234", 15))
        self.assertEqual(r[1]["title"], "1,234,567,890,123,456 tokens")   # exact count still available

    def test_switching_ranges_flips_notation_both_ways(self):
        r = self.run_odo([2345678901234567, 3456789, 2345678901234999])
        self.assertEqual([x["sci"] for x in r], [True, False, True])
        self.assertEqual(r[1]["shown"], "3456789")
        self.assertEqual(r[2]["exp"], 15)

    def test_narrow_card_uses_notation_before_reels_get_unreadable(self):
        n = 123456789012
        self.assertFalse(self.run_odo([n], width=900)[0]["sci"])
        self.assertTrue(self.run_odo([n], width=200)[0]["sci"])

class GaugeTests(unittest.TestCase):
    def test_first_sse_message_only_sets_the_baseline(self):
        # tokens already streamed before the page connected must not count as a 3 s arrival (gauge overshoot on load)
        self.assertIn("de=primed?Math.max(0,e-prevEst):0;primed=true;", ta.PAGE)
        self.assertIn("es.onopen=()=>{primed=false;", ta.PAGE)


class Hold(http.server.BaseHTTPRequestHandler):
    """Upstream that reads the request, then waits, so a test can look at the live card mid-flight."""
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, body):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send(getattr(self.server, "get_body", '{"ok":true}'))
    def do_POST(self):
        self.server.seen.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        self.server.arrived.set()
        self.server.gate.wait(4)
        item = self.server.reply
        if isinstance(item, list):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for part in item:
                p = part.encode(); self.wfile.write(b"%x\r\n%s\r\n" % (len(p), p)); self.wfile.flush(); time.sleep(0.03)
            self.wfile.write(b"0\r\n\r\n"); return
        self._send(item)


class InflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="inflight-")
        self.saved = (ta.DB, tuple(ta.TOT), dict(ta.LIVE), list(ta.DONE), ta.LIVE_VER[0], ta._rid[0], ta.CLIENTS[0])
        ta.DB = os.path.join(self.tmp, "s.db")
        ta.LIVE.clear(); ta.DONE.clear(); ta.init_db()
        self.servers, self.gate = [], threading.Event()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self.gate.set()
        for s in self.servers:
            try: s.shutdown(); s.server_close()
            except Exception: pass
        db, tot, live, done, ver, rid, clients = self.saved
        ta.DB = db; ta.TOT[:] = tot; ta.LIVE.clear(); ta.LIVE.update(live); ta.DONE[:] = done
        ta.LIVE_VER[0] = ver; ta._rid[0] = rid; ta.CLIENTS[0] = clients

    def _pair(self, reply):
        up = Server(("127.0.0.1", 0), Hold)
        up.gate, up.arrived, up.seen, up.reply, up.get_body = self.gate, threading.Event(), [], reply, '{"models":[]}'
        threading.Thread(target=up.serve_forever, daemon=True).start()
        H = type("InflightProxy", (ta.Proxy,), {"UPH": "127.0.0.1", "UPP": up.server_address[1], "SRC": "vllm"})
        px = ta.TS(("127.0.0.1", 0), H)
        threading.Thread(target=px.serve_forever, daemon=True).start()
        self.servers.extend((up, px))
        return up, px

    def _post(self, port, path, payload):
        box = {}
        def run():
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
                raw = json.dumps(payload).encode()
                c.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
                r = c.getresponse(); box["res"] = (r.status, r.read()); c.close()
            except Exception as e:
                box["err"] = e
        t = threading.Thread(target=run); t.start(); return t, box

    def _wait_clear(self):
        deadline = time.time() + 2
        while time.time() < deadline:
            if not ta.live_state()[0]: return True
            time.sleep(0.02)
        return False

    def test_nonstream_is_on_the_card_until_the_exact_counts_land(self):
        self.assertIn("s.wait?'waiting for the full answer...'", ta.PAGE)
        answer = '{"model":"quiet","usage":{"prompt_tokens":5,"completion_tokens":7}}'
        up, px = self._pair(answer)
        t, box = self._post(px.server_address[1], "/v1/chat/completions", {"model": "quiet", "stream": False})
        self.assertTrue(up.arrived.wait(3))
        streams, _ = ta.live_state()
        self.assertEqual([(s["model"], s["endpoint"], s["wait"], s["out_est"]) for s in streams], [("quiet", "/v1/chat/completions", True, 0)])
        self.gate.set(); t.join(5)
        self.assertNotIn("err", box); self.assertEqual(box["res"], (200, answer.encode()))
        self.assertTrue(self._wait_clear())
        row = sqlite3.connect(ta.DB).execute("SELECT source, model, input_tokens, output_tokens FROM requests").fetchone()
        self.assertEqual(row, ("vllm", "quiet", 5, 7))

    def test_openai_request_without_stream_is_in_flight_immediately(self):
        answer = '{"model":"plain","usage":{"prompt_tokens":1,"completion_tokens":2}}'
        up, px = self._pair(answer)
        t, box = self._post(px.server_address[1], "/v1/chat/completions", {"model": "plain"})
        self.assertTrue(up.arrived.wait(3))
        streams, _ = ta.live_state()
        self.assertEqual(len(streams), 1); self.assertTrue(streams[0]["wait"]); self.assertEqual(streams[0]["model"], "plain")
        self.gate.set(); t.join(5)
        self.assertEqual(box["res"][0], 200)
        self.assertTrue(self._wait_clear())

    def test_ollama_default_stays_quiet_until_the_answer_starts(self):
        # Ollama streams unless the client says stream:false, so a plain /api/chat is not shown early.
        answer = '{"model":"llama3","done":true,"prompt_eval_count":4,"eval_count":6,"eval_duration":1000000000}'
        up, px = self._pair(answer)
        t, box = self._post(px.server_address[1], "/api/chat", {"model": "llama3"})
        self.assertTrue(up.arrived.wait(3))
        self.assertEqual(ta.live_state()[0], [])
        self.gate.set(); t.join(5)
        self.assertEqual(box["res"][0], 200)
        self.assertTrue(self._wait_clear())
        row = sqlite3.connect(ta.DB).execute("SELECT model, input_tokens, output_tokens FROM requests").fetchone()
        self.assertEqual(row, ("llama3", 4, 6))

    def test_stream_is_registered_the_old_way_and_chunks_count(self):
        ta.CLIENTS[0] = 1   # the live estimate only ticks while someone is watching the card
        parts = ['data: {"model":"streamy","choices":[{"delta":{"content":"a"}}]}\n\n'] * 8
        parts += ['data: {"model":"streamy","usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n', 'data: [DONE]\n\n']
        up, px = self._pair(parts)
        t, box = self._post(px.server_address[1], "/v1/chat/completions", {"model": "streamy", "stream": True})
        self.assertTrue(up.arrived.wait(3))
        self.assertEqual(ta.live_state()[0], [])   # still hidden until the upstream starts answering
        self.gate.set()
        seen = None; deadline = time.time() + 2
        while time.time() < deadline:
            streams, _ = ta.live_state()
            if streams and streams[0]["out_est"] > 0:
                seen = streams[0]; break
            time.sleep(0.02)
        self.assertIsNotNone(seen)
        self.assertFalse(seen["wait"]); self.assertEqual(seen["model"], "streamy"); self.assertGreater(seen["out_est"], 0)
        t.join(5)
        self.assertEqual(box["res"][0], 200); self.assertIn(b"[DONE]", box["res"][1])
        self.assertTrue(self._wait_clear())

    def test_dead_upstream_does_not_leave_the_card_stuck(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
        H = type("DeadProxy", (ta.Proxy,), {"UPH": "127.0.0.1", "UPP": port, "SRC": "vllm"})
        px = ta.TS(("127.0.0.1", 0), H)
        threading.Thread(target=px.serve_forever, daemon=True).start(); self.servers.append(px)
        st, _ = req(px.server_address[1], "POST", "/v1/chat/completions", {"model": "gone", "stream": False})
        self.assertEqual(st, 502)
        self.assertEqual(ta.live_state()[0], [])

    def test_get_passthrough_is_not_a_live_request(self):
        up, px = self._pair('{"ok":true}')
        st, body = req(px.server_address[1], "GET", "/api/tags")
        self.assertEqual((st, body), (200, b'{"models":[]}'))
        self.assertEqual(ta.live_state()[0], [])


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def git(cwd, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="Test",
               GIT_COMMITTER_EMAIL="t@example.com", GIT_TERMINAL_PROMPT="0")
    p = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError("%s: %s" % (" ".join(args), p.stderr or p.stdout))
    return p.stdout.strip()


def commit_ver(repo, ver, body, msg):
    with open(os.path.join(repo, "VERSION"), "w", encoding="utf-8") as f:
        f.write(ver + "\n")
    with open(os.path.join(repo, "token_annihilator.py"), "w", encoding="utf-8") as f:
        f.write(body)
    with open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("stats.db*\n__pycache__/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", msg)


def make_install(root):
    remote, inst = os.path.join(root, "remote"), os.path.join(root, "install")
    os.makedirs(remote)
    git(remote, "init", "-b", "main")
    commit_ver(remote, "1.0.0", "print('one')\n", "v1.0.0")
    git(remote, "tag", "v1.0.0")
    commit_ver(remote, "1.1.0", "print('two')\n", "v1.1.0")
    git(remote, "tag", "-a", "v1.1.0", "-m", "release 1.1.0")
    subprocess.run(["git", "clone", "--quiet", remote, inst], check=True, capture_output=True, text=True)
    git(inst, "checkout", "--quiet", "--detach", "v1.0.0")
    with open(os.path.join(inst, "stats.db"), "w", encoding="utf-8") as f:
        f.write("precious")
    return remote, inst


def write_hook(inst, marker):
    quoted = marker.replace("'", "'\\''")
    script = "#!/bin/sh\ntouch '%s'\nexit 1\n" % quoted
    hookdir = os.path.join(inst, ".git", "hooks")
    os.makedirs(hookdir, exist_ok=True)
    for name in ("pre-checkout", "post-checkout"):
        path = os.path.join(hookdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(path, 0o755)


class ReleaseHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.server.agent = self.headers.get("User-Agent")
        if self.path != "/repos/SaltShaker87/token-annihilation-dashboard/releases/latest":
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        body = self.server.body
        self.send_response(self.server.code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class UpdateLogicTests(unittest.TestCase):
    def test_page_offers_one_update_and_does_not_poll(self):
        self.assertEqual(ta.PAGE.count("id=updBtn"), 1)
        self.assertIn(">Update</button>", ta.PAGE)
        chunk = ta.PAGE.split("function checkUpd", 1)[1]
        self.assertNotIn("setInterval", chunk)
        self.assertIn("checkUpd();", chunk)
        self.assertIn("confirm:true", chunk)
        self.assertIn("s.boot!==boot0", chunk.replace(" ", ""))

    def test_official_urls_and_restart_command_are_fixed(self):
        src = read_text(os.path.join(HERE, "token_annihilator.py"))
        self.assertIn('OFFICIAL_GIT = "https://github.com/SaltShaker87/token-annihilation-dashboard.git"', src)
        self.assertIn('RELEASES_URL = "https://api.github.com/repos/%s/releases/latest" % OFFICIAL_REPO', src)
        self.assertNotIn("x-access-token", src)
        self.assertEqual(ta.RESTART_ARGV, ["systemctl", "--user", "restart", "token-annihilator"])
        blob = " ".join(ta.RESTART_ARGV)
        for bad in ("ollama", "vllm", "llama", "11434", "11435", "8080", "8081", "8000", "8001"):
            self.assertNotIn(bad, blob)
        import inspect
        self.assertEqual(list(inspect.signature(ta.apply_update).parameters), [])

    def test_versions_and_tags(self):
        self.assertEqual(ta.ver_tuple("v1.2.3"), ta.ver_tuple("1.2.3"))
        self.assertLess(ta.ver_tuple("1.9.0"), ta.ver_tuple("1.10.0"))
        self.assertEqual(ta.ver_tuple("v1.2.3\n"), (1, 2, 3))
        for bad in ("1.2", "01.2.3", "v1.2.3-rc1", "main", "v1.2.3;reboot", "--version", ""):
            self.assertIsNone(ta.ver_tuple(bad), bad)
            self.assertFalse(ta.TAG_OK.match(bad), bad)
        self.assertTrue(ta.TAG_OK.match("v0.0.0"))
        self.assertFalse(ta.TAG_OK.match("1.2.3"))

    def test_origin_and_confirm(self):
        host = {"Origin": "http://127.0.0.1:11480", "Host": "127.0.0.1:11480"}
        self.assertTrue(ta.origin_ok(host))
        self.assertTrue(ta.origin_ok(dict(host, **{"Sec-Fetch-Site": "same-origin"})))
        self.assertFalse(ta.origin_ok({}))
        self.assertFalse(ta.origin_ok({"Host": "127.0.0.1:11480"}))
        self.assertFalse(ta.origin_ok({"Origin": "http://evil.example", "Host": "127.0.0.1:11480"}))
        self.assertFalse(ta.origin_ok({"Origin": "http://127.0.0.1:9", "Host": "127.0.0.1:11480"}))
        self.assertFalse(ta.origin_ok({"Origin": "http://127.0.0.1:11480.evil", "Host": "127.0.0.1:11480"}))
        self.assertFalse(ta.origin_ok(dict(host, **{"Sec-Fetch-Site": "cross-site"})))
        self.assertFalse(ta.origin_ok({"Origin": "http://127.0.0.1:11480\r\nX: y", "Host": "127.0.0.1:11480"}))
        seen = []
        old = ta.apply_update
        ta.apply_update = lambda: seen.append("go") or {"ok": True, "restart": "manual"}
        try:
            code, _ = ta.begin_update({"confirm": True, "tag": "main", "url": "https://evil.example/x.git", "ref": "refs/heads/main"}, host)
            self.assertEqual(code, 200)
            self.assertEqual(seen, ["go"])
            code, res = ta.begin_update({"confirm": True, "tag": "v9.9.9", "url": "https://evil.example/x.git"},
                                        {"Origin": "http://evil.example", "Host": "127.0.0.1:11480"})
            self.assertEqual(code, 403)
            self.assertFalse(res["ok"])
            code, _ = ta.begin_update({"confirm": "true", "tag": "v9.9.9"}, host)
            self.assertEqual(code, 400)
            self.assertEqual(seen, ["go"])
        finally:
            ta.apply_update = old

    def test_restart_plan_matches_only_this_process(self):
        self.assertEqual(ta.service_state("", 5, ""), "manual")
        self.assertEqual(ta.service_state("ActiveState=active\nMainPID=5\n", 5, None), "manual")
        self.assertEqual(ta.service_state("ActiveState=active\nMainPID=5\n", 5, "abc"), "systemd")
        self.assertEqual(ta.service_state("ActiveState=inactive\nMainPID=5\n", 5, "abc"), "manual")
        self.assertEqual(ta.service_state("ActiveState=active\nMainPID=9\n", 5, "abc"), "manual")
        self.assertEqual(ta.restart_plan(), "manual")

    def test_schedule_restart_is_only_the_user_unit(self):
        seen = []
        def fake_popen(argv, **kw):
            seen.append((list(argv), kw))
            return object()
        old_p, old_d = subprocess.Popen, ta.RESTART_DELAY
        subprocess.Popen = fake_popen
        ta.RESTART_DELAY = 0
        try:
            ta.schedule_restart().join(2)
        finally:
            subprocess.Popen = old_p
            ta.RESTART_DELAY = old_d
        self.assertEqual(len(seen), 1)
        argv, kw = seen[0]
        self.assertEqual(argv, ["systemctl", "--user", "restart", "token-annihilator"])
        self.assertTrue(kw.get("start_new_session"))
        self.assertIsNot(kw.get("shell"), True)

    def test_github_payload_cannot_choose_a_url(self):
        srv = ta.TS(("127.0.0.1", 0), ReleaseHandler)
        srv.code, srv.body, srv.agent = 200, b"", None
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        old = ta.RELEASES_URL
        try:
            port = srv.server_address[1]
            ta.RELEASES_URL = "http://127.0.0.1:%d/repos/SaltShaker87/token-annihilation-dashboard/releases/latest" % port
            srv.code, srv.body = 404, b'{"message":"Not Found","tarball_url":"https://evil.example/nope.tar.gz"}'
            self.assertIsNone(ta.fetch_latest())
            payload = {"tag_name": "v1.2.3", "tarball_url": "https://evil.example/evil.tar.gz",
                       "zipball_url": "https://evil.example/evil.zip", "target_commitish": "main",
                       "html_url": "https://evil.example/release"}
            srv.code, srv.body = 200, json.dumps(payload).encode()
            self.assertEqual(ta.fetch_latest(), {"tag": "v1.2.3", "version": "1.2.3"})
            self.assertEqual(srv.agent, "token-annihilation-dashboard")
            srv.body = json.dumps({"tag_name": "v1.2.3-rc1", "tarball_url": "https://evil.example/x.tar.gz"}).encode()
            with self.assertRaises(RuntimeError) as cm:
                ta.fetch_latest()
            self.assertNotIn("evil.example", str(cm.exception))
            srv.code = 403
            srv.body = b'{"message":"rate limit"}'
            with self.assertRaises(RuntimeError):
                ta.fetch_latest()
        finally:
            ta.RELEASES_URL = old
            srv.shutdown(); srv.server_close()


class UpdateRepoTests(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.real_head = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True)
        self.real_status = subprocess.check_output(["git", "-C", HERE, "status", "--porcelain"], text=True)
        self._git_env = {k: os.environ.get(k) for k in ("GIT_DIR", "GIT_WORK_TREE")}
        self.root = tempfile.mkdtemp(prefix="tad-upd-")
        self.addCleanup(self._cleanup)
        self._saved = {k: getattr(ta, k) for k in (
            "HOME", "OFFICIAL_GIT", "RELEASES_URL", "fetch_latest", "schedule_restart", "restart_plan", "apply_update", "git_cmd")}
        self.remote, self.inst = make_install(self.root)
        self.marker = os.path.join(self.root, "hook-touched")
        write_hook(self.inst, self.marker)
        self.restarts = []
        ta.HOME = self.inst
        ta.OFFICIAL_GIT = self.remote
        ta.fetch_latest = lambda: {"tag": "v1.1.0", "version": "1.1.0"}
        ta.schedule_restart = lambda: self.restarts.append(1)
        ta.restart_plan = lambda: "manual"

    def _cleanup(self):
        for k, v in self._saved.items():
            setattr(ta, k, v)
        try: os.chdir(self.cwd)
        except OSError: pass
        for k, v in self._git_env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        subprocess.run(["rm", "-rf", self.root], check=False)
        now = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True)
        st = subprocess.check_output(["git", "-C", HERE, "status", "--porcelain"], text=True)
        self.assertEqual(now, self.real_head)
        self.assertEqual(st, self.real_status)

    def _ver(self):
        return read_text(os.path.join(self.inst, "VERSION")).strip()

    def _record(self):
        calls, real = [], ta.git_cmd
        def wrap(args, timeout=60):
            calls.append(list(args))
            return real(args, timeout)
        ta.git_cmd = wrap
        return calls

    def test_no_release_and_current_release_do_not_checkout(self):
        calls = self._record()
        ta.fetch_latest = lambda: None
        res = ta.apply_update()
        self.assertFalse(res["ok"]); self.assertEqual(res["state"], "no_release"); self.assertEqual(self._ver(), "1.0.0")
        ta.fetch_latest = lambda: {"tag": "v1.0.0", "version": "1.0.0"}
        res = ta.apply_update()
        self.assertTrue(res["ok"]); self.assertEqual(res["state"], "up_to_date"); self.assertIsNone(res["restart"])
        self.assertEqual(self._ver(), "1.0.0")
        self.assertFalse(any(c and c[0] in ("fetch", "checkout", "reset", "update-ref") for c in calls))
        self.assertFalse(os.path.exists(self.marker))

    def test_dirty_tree_is_left_alone(self):
        calls = self._record()
        path = os.path.join(self.inst, "token_annihilator.py")
        with open(path, "a", encoding="utf-8") as f:
            f.write("# local tweak\n")
        before = git(self.inst, "rev-parse", "HEAD")
        res = ta.apply_update()
        self.assertFalse(res["ok"]); self.assertEqual(res["state"], "dirty")
        self.assertIn("token_annihilator.py", res["message"])
        self.assertEqual(git(self.inst, "rev-parse", "HEAD"), before)
        self.assertIn("# local tweak", read_text(path))
        self.assertFalse(any(c and c[0] in ("fetch", "checkout", "reset", "update-ref") for c in calls))
        self.assertEqual(self.restarts, [])

    def test_untracked_files_do_not_block_update(self):
        for n in ("run.log", "my_helper.py"):
            with open(os.path.join(self.inst, n), "w", encoding="utf-8") as f:
                f.write("keep me\n")
        self.assertEqual(ta.local_modifications(), [])
        self.assertEqual(ta.update_status().get("dirty"), [])
        for n in ("run.log", "my_helper.py"):
            self.assertEqual(read_text(os.path.join(self.inst, n)), "keep me\n")

    def test_local_commits_are_not_overwritten(self):
        with open(os.path.join(self.inst, "LOCAL"), "w", encoding="utf-8") as f:
            f.write("mine\n")
        git(self.inst, "add", "LOCAL"); git(self.inst, "commit", "-m", "local only")
        head = git(self.inst, "rev-parse", "HEAD")
        calls = self._record()
        res = ta.apply_update()
        self.assertFalse(res["ok"]); self.assertEqual(res["state"], "blocked"); self.assertIn("commits", res["message"])
        self.assertEqual(git(self.inst, "rev-parse", "HEAD"), head)
        self.assertEqual(self._ver(), "1.0.0")
        self.assertTrue(os.path.exists(os.path.join(self.inst, "LOCAL")))
        self.assertFalse(any(c and c[0] in ("checkout", "reset") for c in calls))
        self.assertEqual(self.restarts, [])

    def test_bad_release_is_rolled_back(self):
        commit_ver(self.remote, "9.9.9", "print('nope')\n", "wrong version file")
        git(self.remote, "tag", "v1.2.0")
        commit_ver(self.remote, "1.3.0", "def (\n", "syntax error")
        git(self.remote, "tag", "v1.3.0")
        old = git(self.inst, "rev-parse", "HEAD")
        ta.fetch_latest = lambda: {"tag": "v1.2.0", "version": "1.2.0"}
        res = ta.apply_update()
        self.assertFalse(res["ok"]); self.assertIn("put back", res["message"])
        self.assertEqual(git(self.inst, "rev-parse", "HEAD"), old)
        self.assertEqual(self._ver(), "1.0.0")
        self.assertEqual(read_text(os.path.join(self.inst, "token_annihilator.py")), "print('one')\n")
        self.assertEqual(git(self.inst, "rev-parse", "refs/backup/token-annihilator-prev"), old)
        ta.fetch_latest = lambda: {"tag": "v1.3.0", "version": "1.3.0"}
        res = ta.apply_update()
        self.assertFalse(res["ok"]); self.assertIn("syntax error", res["message"]); self.assertIn("put back", res["message"])
        self.assertEqual(git(self.inst, "rev-parse", "HEAD"), old)
        self.assertEqual(read_text(os.path.join(self.inst, "token_annihilator.py")), "print('one')\n")
        self.assertEqual(read_text(os.path.join(self.inst, "stats.db")), "precious")
        self.assertEqual(self.restarts, [])

    def test_endpoint_installs_only_the_official_tag(self):
        calls = self._record()
        old = git(self.inst, "rev-parse", "HEAD")
        srv = ta.TS(("127.0.0.1", 0), ta.Dash)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        origin = "http://127.0.0.1:%d" % port
        try:
            st, raw = req(port, "GET", "/api/update")
            self.assertEqual(st, 200)
            info = json.loads(raw)
            self.assertEqual(info["state"], "update_available")
            self.assertEqual(info["current"], "1.0.0")
            self.assertEqual(info["latest"], "1.1.0")
            self.assertTrue(info["boot"])
            def post(body, headers):
                raw_body = json.dumps(body).encode()
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                c.request("POST", "/api/update", body=raw_body, headers=dict({"Content-Type": "application/json"}, **headers))
                r = c.getresponse(); data = r.read(); c.close()
                return r.status, json.loads(data)
            self.assertEqual(post({"confirm": True, "tag": "main", "url": "https://evil.example/x.git"}, {})[0], 403)
            self.assertEqual(post({"confirm": True, "url": "https://evil.example/x.git"}, {"Origin": "http://evil.example"})[0], 403)
            self.assertEqual(post({"confirm": True}, {"Origin": origin, "Sec-Fetch-Site": "cross-site"})[0], 403)
            self.assertEqual(post({"confirm": "true", "tag": "v9.9.9", "url": "https://evil.example/x.git"}, {"Origin": origin})[0], 400)
            self.assertEqual(self._ver(), "1.0.0")
            code, res = post({"confirm": True, "tag": "v9.9.9", "url": "https://evil.example/x.git", "ref": "main"},
                             {"Origin": origin, "Sec-Fetch-Site": "same-origin"})
            self.assertEqual(code, 200, res)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["restart"], "manual")
            self.assertEqual(res["version"], "1.1.0")
            self.assertIn("yourself", res["message"])
            self.assertEqual(self._ver(), "1.1.0")
            self.assertEqual(read_text(os.path.join(self.inst, "token_annihilator.py")), "print('two')\n")
            self.assertEqual(read_text(os.path.join(self.inst, "stats.db")), "precious")
            self.assertEqual(git(self.inst, "rev-parse", "HEAD"), git(self.inst, "rev-parse", "v1.1.0^{commit}"))
            self.assertEqual(git(self.inst, "rev-parse", "refs/backup/token-annihilator-prev"), old)
            self.assertFalse(os.path.exists(self.marker))
            self.assertEqual(self.restarts, [])
            self.assertIn(["fetch", "--quiet", self.remote, "+refs/tags/v1.1.0:refs/tags/v1.1.0"], calls)
            self.assertIn(["checkout", "--detach", "--quiet", "refs/tags/v1.1.0"], calls)
            self.assertFalse(any(c and c[0] == "reset" for c in calls))
            blob = " ".join(" ".join(c) for c in calls)
            self.assertNotIn("evil.example", blob)
            for bad in ("11434", "11435", "8080", "8081", "8000", "8001"):
                self.assertNotIn(bad, blob)
            self.assertEqual(os.getcwd(), self.cwd)
            again = json.loads(req(port, "GET", "/api/update")[1])
            self.assertEqual(again["state"], "up_to_date")
            self.assertEqual(again["current"], "1.1.0")
        finally:
            srv.shutdown(); srv.server_close()

    def test_git_dir_env_cannot_redirect_the_update(self):
        other = tempfile.mkdtemp(prefix="tad-other-")
        try:
            git(other, "init", "-b", "main")
            with open(os.path.join(other, "keep"), "w", encoding="utf-8") as f:
                f.write("safe\n")
            git(other, "add", "keep"); git(other, "commit", "-m", "keep")
            other_head = git(other, "rev-parse", "HEAD")
            os.environ["GIT_DIR"] = os.path.join(other, ".git")
            os.environ["GIT_WORK_TREE"] = other
            res = ta.apply_update()
            self.assertTrue(res["ok"], res)
            self.assertEqual(self._ver(), "1.1.0")
            env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
            head = subprocess.run(["git", "-C", other, "rev-parse", "HEAD"], capture_output=True, text=True, env=env)
            self.assertEqual(head.stdout.strip(), other_head)
            self.assertFalse(os.path.exists(os.path.join(other, "token_annihilator.py")))
            self.assertEqual(read_text(os.path.join(other, "keep")), "safe\n")
        finally:
            os.environ.pop("GIT_DIR", None)
            os.environ.pop("GIT_WORK_TREE", None)
            subprocess.run(["rm", "-rf", other], check=False)


if __name__ == "__main__":
    unittest.main()
