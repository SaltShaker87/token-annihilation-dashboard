#!/usr/bin/env python3
"""Stdlib checks for the Ollama, vLLM and llama.cpp token counting. Run: python3 test_annihilator.py"""
import http.client, http.server, json, os, sqlite3, subprocess, sys, tempfile, threading, time, unittest
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
            ta.RUNTIMES[:] = [{"src": "proxy", "listen": 1, "up": ("127.0.0.1", self.up["vllm"]), "cls": ta.Proxy}]
            ta.check_runtimes(); self.assertEqual(ta.Proxy.SRC, "vllm")
            ta.RUNTIMES[0]["up"] = ("127.0.0.1", self.up["ollama"])
            ta.check_runtimes(); self.assertEqual(ta.Proxy.SRC, "proxy")
        finally: ta.RUNTIMES[:] = saved; ta.Proxy.SRC = "proxy"

    def test_streamed_vllm_gets_speed(self):
        while self.servers["vllm"].queue["/v1/chat/completions"][:-1]: self.servers["vllm"].queue["/v1/chat/completions"].pop(0)   # if run alone
        st, body = req(self.px["vllm"], "POST", "/v1/chat/completions", {"model": "slowpoke", "stream": True})
        self.assertEqual(st, 200); self.assertIn(b"[DONE]", body)
        deadline = time.time() + 3; r = None
        while time.time() < deadline and not r:
            r = sqlite3.connect(self.db).execute("SELECT input_tokens, output_tokens, eval_s FROM requests WHERE model='slowpoke'").fetchone(); time.sleep(0.05)
        self.assertEqual(r[:2], (2, 5)); self.assertTrue(0.1 < r[2] < 2, r)


if __name__ == "__main__":
    unittest.main()
