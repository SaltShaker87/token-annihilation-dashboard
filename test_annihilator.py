#!/usr/bin/env python3
"""Stdlib checks for the Ollama, vLLM, and llama.cpp token proxies."""
import http.client, json, os, sqlite3, subprocess, sys, tempfile, threading, time, unittest
from socketserver import ThreadingMixIn
import http.server

os.environ.setdefault("DASH_BIND", "127.0.0.1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import token_annihilator as ta


def parse(body, req=b"{}"):
    if isinstance(body, str):
        body = body.encode()
    return ta.parse_usage("/test", req, body)


class ParseTests(unittest.TestCase):
    def test_ollama_ndjson_and_pretty_json(self):
        body = '\n'.join([
            '{"model":"llama3","done":false,"message":{"content":"Hi"}}',
            '{"model":"llama3","done":true,"prompt_eval_count":10,"eval_count":20,"eval_duration":2000000000}',
        ])
        self.assertEqual(parse(body), ("llama3", 10, 20, 2.0))
        pretty = '{\n  "model": "m",\n  "prompt_eval_count": 1,\n  "eval_count": 2\n}\n'
        self.assertEqual(parse(pretty)[:3], ("m", 1, 2))

    def test_openai_sse_ignores_null_usage_and_done(self):
        body = (
            'data: {"model":"streamy","usage":null}\n\n'
            'data: {"model":"streamy","usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n'
            'data: [DONE]\n'
        )
        self.assertEqual(parse(body, b'{"model":"from-req"}'), ("streamy", 3, 4, None))

    def test_stream_totals_are_not_summed(self):
        body = (
            'data: {"content":"a","stop":false,"tokens_predicted":1,"tokens_evaluated":4}\n\n'
            'data: {"content":"","stop":true,"tokens_predicted":5,"tokens_evaluated":4,'
            '"timings":{"predicted_ms":100,"predicted_n":5,"prompt_n":4}}\n\n'
        )
        self.assertEqual(parse(body), (None, 4, 5, 0.1))

    def test_anthropic_splits_input_and_output(self):
        body = (
            'event: message_start\n'
            'data: {"type":"message_start","message":{"model":"anth","usage":{"input_tokens":25,"output_tokens":1}}}\n\n'
            'event: message_delta\n'
            'data: {"type":"message_delta","usage":{"output_tokens":15}}\n\n'
            'event: message_stop\n'
            'data: {"type":"message_stop"}\n'
        )
        self.assertEqual(parse(body), ("anth", 25, 15, None))

    def test_responses_api_nested_usage(self):
        body = 'data: {"type":"response.completed","response":{"model":"resp","usage":{"input_tokens":2,"output_tokens":9,"total_tokens":11}}}\n\n'
        self.assertEqual(parse(body)[:3], ("resp", 2, 9))

    def test_vllm_metrics_generation_time(self):
        body = '{"model":"meta-llama","usage":{"prompt_tokens":5,"completion_tokens":7},"metrics":{"generation_time_ms":500}}'
        self.assertEqual(parse(body), ("meta-llama", 5, 7, 0.5))

    def test_llamacpp_native_and_batch_and_timings(self):
        one = '{"content":"hi","tokens_evaluated":8,"tokens_predicted":6,"model":"gguf","timings":{"predicted_ms":250,"predicted_n":6,"prompt_n":8}}'
        self.assertEqual(parse(one), ("gguf", 8, 6, 0.25))
        batch = '[{"tokens_evaluated":3,"tokens_predicted":4},{"tokens_evaluated":5,"tokens_predicted":6}]'
        self.assertEqual(parse(batch), (None, 8, 10, None))
        timings = '{"timings":{"prompt_n":2,"cache_n":3,"predicted_n":4,"predicted_ms":40}}'
        self.assertEqual(parse(timings), (None, 5, 4, 0.04))
        unset_cache = '{"timings":{"prompt_n":2,"cache_n":-1,"predicted_n":4,"predicted_ms":40}}'
        self.assertEqual(parse(unset_cache), (None, 2, 4, 0.04))

    def test_embeddings_use_the_usage_object_not_the_vectors(self):
        body = '{"model":"e","data":[{"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":6,"total_tokens":6}}'
        self.assertEqual(parse(body), ("e", 6, None, None))

    def test_request_model_and_rerank(self):
        self.assertEqual(parse('{"id":"x"}', b'{"model":"asked"}'), ("asked", None, None, None))
        self.assertEqual(parse('{"model":"rerank","usage":{"prompt_tokens":12,"total_tokens":12}}'), ("rerank", 12, None, None))

    def test_keep_tail_preserves_head_and_end(self):
        tail = bytearray(b"HEAD-START" + b"x" * 5_000_000 + b"TAIL-END")
        ta._keep_tail(tail)
        self.assertTrue(tail.startswith(b"HEAD-START"))
        self.assertTrue(tail.endswith(b"TAIL-END"))
        self.assertLess(len(tail), 2_000_000)
        small = bytearray(b"tiny")
        ta._keep_tail(small)
        self.assertEqual(bytes(small), b"tiny")


class MigrationTests(unittest.TestCase):
    def test_old_rows_become_ollama(self):
        path = tempfile.mktemp(suffix=".db")
        c = sqlite3.connect(path)
        c.execute("""CREATE TABLE requests(id INTEGER PRIMARY KEY, ts REAL, model TEXT, endpoint TEXT,
          input_tokens INTEGER, output_tokens INTEGER, duration_s REAL, eval_s REAL, status INTEGER, source TEXT, dedupe TEXT UNIQUE)""")
        c.execute("INSERT INTO requests(ts, model, endpoint, source) VALUES(1,'old','/api/chat','proxy')")
        c.commit(); c.close()
        old = ta.DB
        try:
            ta.DB = path
            ta.db().close()
            c = sqlite3.connect(path)
            cols = [r[1] for r in c.execute("PRAGMA table_info(requests)")]
            self.assertIn("backend", cols)
            self.assertEqual(c.execute("SELECT backend, model FROM requests").fetchone(), ("ollama", "old"))
            c.close()
        finally:
            ta.DB = old
            os.remove(path)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _reply(self, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._reply(b'{"status":"ok"}')
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        self.server.seen.append((self.path, body))
        key = self.path.split("?")[0].rstrip("/") or "/"
        self._reply(self.server.replies.get(key, '{"ok":true}'))


class Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def post(port, path, payload=None, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
    r = conn.getresponse()
    data = r.read()
    status = r.status
    conn.close()
    return status, data


def get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    r = conn.getresponse()
    data = r.read()
    status = r.status
    conn.close()
    return status, data


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="annihilator-")
        cls.db = os.path.join(cls.tmp, "stats.db")
        cls.up_ports = {"ollama": 19134, "vllm": 19100, "llamacpp": 19180}
        cls.proxy_ports = {"ollama": 18135, "vllm": 18136, "llamacpp": 18137}
        cls.dash = 18180
        replies = {
            cls.up_ports["ollama"]: {
                "/api/chat": '\n'.join([
                    '{"model":"llama3","done":false,"message":{"content":"Hi"}}',
                    '{"model":"llama3","done":true,"prompt_eval_count":10,"eval_count":20,"eval_duration":2000000000}',
                ]),
                "/api/tags": '{"models":[]}',
            },
            cls.up_ports["vllm"]: {
                "/v1/chat/completions": "",  # filled per request below via a mutable box
                "/v1/responses": 'data: {"type":"response.completed","response":{"model":"resp","usage":{"input_tokens":2,"output_tokens":9}}}\n\n',
                "/v1/rerank": '{"model":"rerank","usage":{"prompt_tokens":12,"total_tokens":12}}',
                "/health": '{"status":"ok"}',
            },
            cls.up_ports["llamacpp"]: {
                "/completion": '{"content":"hi","tokens_evaluated":8,"tokens_predicted":6,"model":"gguf","timings":{"predicted_ms":250,"predicted_n":6,"prompt_n":8}}',
                "/v1/messages": (
                    'event: message_start\n'
                    'data: {"type":"message_start","message":{"model":"anth","usage":{"input_tokens":25,"output_tokens":1}}}\n\n'
                    'event: message_delta\n'
                    'data: {"type":"message_delta","usage":{"output_tokens":15}}\n\n'
                ),
                "/health": '{"status":"ok"}',
            },
        }
        # Two different bodies share /v1/chat/completions. Serve them in order.
        cls.vllm_chats = [
            '{"model":"meta-llama","usage":{"prompt_tokens":5,"completion_tokens":7},"metrics":{"generation_time_ms":500}}',
            'data: {"model":"streamy","usage":null}\n\ndata: {"model":"streamy","usage":{"prompt_tokens":3,"completion_tokens":4}}\n\ndata: [DONE]\n',
            '{"id":"blank"}',
        ]
        cls.llama_completions = [
            replies[cls.up_ports["llamacpp"]]["/completion"],
            'data: {"content":"a","tokens_predicted":1,"tokens_evaluated":4,"stop":false}\n\ndata: {"content":"","stop":true,"tokens_predicted":5,"tokens_evaluated":4,"timings":{"predicted_ms":100,"predicted_n":5,"prompt_n":4}}\n\n',
            '[{"tokens_evaluated":3,"tokens_predicted":4,"model":"batch"},{"tokens_evaluated":5,"tokens_predicted":6}]',
        ]
        cls.servers = []
        for port, routes in replies.items():
            httpd = Server(("127.0.0.1", port), Handler)
            httpd.replies = dict(routes)
            httpd.seen = []
            if port == cls.up_ports["vllm"]:
                chats = cls.vllm_chats
                def do_post_vllm(self, chats=chats):
                    n = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(n)
                    self.server.seen.append((self.path, body))
                    key = self.path.split("?")[0].rstrip("/") or "/"
                    if key == "/v1/chat/completions":
                        payload = chats.pop(0)
                    else:
                        payload = self.server.replies[key]
                    self._reply(payload)
                httpd.RequestHandlerClass = type("VllmHandler", (Handler,), {"do_POST": do_post_vllm})
            if port == cls.up_ports["llamacpp"]:
                comps = cls.llama_completions
                def do_post_llama(self, comps=comps):
                    n = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(n)
                    self.server.seen.append((self.path, body))
                    key = self.path.split("?")[0].rstrip("/") or "/"
                    if key == "/completion":
                        payload = comps.pop(0)
                    else:
                        payload = self.server.replies[key]
                    self._reply(payload)
                httpd.RequestHandlerClass = type("LlamaHandler", (Handler,), {"do_POST": do_post_llama})
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            cls.servers.append(httpd)
        env = os.environ.copy()
        env.update({
            "STATS_DB": cls.db,
            "DASH_BIND": "127.0.0.1",
            "DASH_PORT": str(cls.dash),
            "PROXY_BIND": "127.0.0.1",
            "PROXY_PORT": str(cls.proxy_ports["ollama"]),
            "VLLM_PROXY_PORT": str(cls.proxy_ports["vllm"]),
            "LLAMACPP_PROXY_PORT": str(cls.proxy_ports["llamacpp"]),
            "UPSTREAM_HOST": "127.0.0.1",
            "UPSTREAM_PORT": str(cls.up_ports["ollama"]),
            "VLLM_UPSTREAM_HOST": "127.0.0.1",
            "VLLM_UPSTREAM_PORT": str(cls.up_ports["vllm"]),
            "LLAMACPP_UPSTREAM_HOST": "127.0.0.1",
            "LLAMACPP_UPSTREAM_PORT": str(cls.up_ports["llamacpp"]),
            "ENGINES": "ollama,vllm,llamacpp",
        })
        cls.proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "token_annihilator.py")],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 8
        cls.boot = ""
        while time.time() < deadline:
            line = cls.proc.stdout.readline()
            cls.boot += line
            if "dashboard http://" in line:
                break
        else:
            raise RuntimeError("proxy did not start: " + cls.boot)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try: cls.proc.wait(timeout=3)
        except subprocess.TimeoutExpired: cls.proc.kill()
        try: cls.proc.stdout.close()
        except Exception: pass
        for s in cls.servers:
            s.shutdown(); s.server_close()

    def test_counts_each_engine_and_renders_dashboard(self):
        ollama_body = b'{"model":"llama3","messages":[{"role":"user","content":"hi"}]}'
        status, echoed = post(self.proxy_ports["ollama"], "/api/chat", raw=ollama_body)
        self.assertEqual(status, 200)
        self.assertIn(b"prompt_eval_count", echoed)
        status, _ = post(self.proxy_ports["vllm"], "/v1/chat/completions", {"model": "meta-llama", "messages": []})
        self.assertEqual(status, 200)
        status, streamed = post(self.proxy_ports["vllm"], "/v1/chat/completions", {"model": "streamy", "stream": True})
        self.assertIn(b"[DONE]", streamed)
        status, _ = post(self.proxy_ports["vllm"], "/v1/chat/completions", {"model": "a<b>", "messages": []})
        self.assertEqual(status, 200)
        status, _ = post(self.proxy_ports["vllm"], "/v1/responses", {"model": "resp", "input": "hi"})
        self.assertEqual(status, 200)
        status, _ = post(self.proxy_ports["vllm"], "/v1/rerank", {"model": "rerank", "query": "q", "documents": ["a"]})
        self.assertEqual(status, 200)
        status, _ = post(self.proxy_ports["llamacpp"], "/completion", {"prompt": "hi"})
        self.assertEqual(status, 200)
        status, _ = post(self.proxy_ports["llamacpp"], "/completion", {"prompt": "hi", "stream": True})
        self.assertEqual(status, 200)
        status, _ = post(self.proxy_ports["llamacpp"], "/completion", {"prompt": ["a", "b"]})
        self.assertEqual(status, 200)
        status, _ = post(self.proxy_ports["llamacpp"], "/v1/messages", {"model": "anth", "messages": [], "max_tokens": 8})
        self.assertEqual(status, 200)
        status, tags = get(self.proxy_ports["ollama"], "/api/tags")
        self.assertEqual(status, 200)
        self.assertIn(b"ok", tags)
        post(self.proxy_ports["ollama"], "/api/show", {"model": "nope"})

        deadline = time.time() + 3
        rows = []
        while time.time() < deadline:
            c = sqlite3.connect(self.db)
            rows = c.execute("SELECT backend, model, endpoint, input_tokens, output_tokens, eval_s FROM requests WHERE source='proxy' ORDER BY id").fetchall()
            c.close()
            if len(rows) >= 10:
                break
            time.sleep(0.05)
        self.assertEqual(rows, [
            ("ollama", "llama3", "/api/chat", 10, 20, 2.0),
            ("vllm", "meta-llama", "/v1/chat/completions", 5, 7, 0.5),
            ("vllm", "streamy", "/v1/chat/completions", 3, 4, None),
            ("vllm", "a<b>", "/v1/chat/completions", None, None, None),
            ("vllm", "resp", "/v1/responses", 2, 9, None),
            ("vllm", "rerank", "/v1/rerank", 12, None, None),
            ("llamacpp", "gguf", "/completion", 8, 6, 0.25),
            ("llamacpp", None, "/completion", 4, 5, 0.1),
            ("llamacpp", "batch", "/completion", 8, 10, None),
            ("llamacpp", "anth", "/v1/messages", 25, 15, None),
        ])

        status, page = get(self.dash, "/")
        self.assertEqual(status, 200)
        text = page.decode()
        self.assertIn('src="/mascot.gif"', text)
        status, gif = get(self.dash, "/mascot.gif")
        self.assertEqual(status, 200)
        self.assertTrue(gif.startswith(b"GIF89a"))
        self.assertIn("vLLM", text)
        self.assertIn("llama.cpp", text)
        self.assertIn("Ollama", text)
        self.assertIn("llama3", text)
        self.assertIn("meta-llama", text)
        self.assertIn("a&lt;b&gt;", text)
        self.assertNotIn("<td>a<b></td>", text)
        self.assertIn("13.3", text)  # output tokens/sec across engines that reported a generation time
        self.assertIn("The server at", text)
        self.assertIn("1 request came through with no token receipt", text)
        self.assertIn("$0.0018", text)

        status, only = get(self.dash, "/?backend=vllm")
        self.assertEqual(status, 200)
        vtext = only.decode()
        self.assertIn("Showing vLLM.", vtext)
        self.assertNotIn("llama3", vtext)
        self.assertIn(">42<", vtext)
        self.assertIn("14.0 tok/s", vtext)

        status, junk = get(self.dash, "/?backend=%3Cscript%3E")
        text = junk.decode()
        self.assertNotIn("backend=<script>", text)
        self.assertNotIn("%3Cscript%3E", text)
        self.assertIn("Showing every engine.", text)

    def test_startup_lists_three_proxies(self):
        self.assertIn("ollama proxy 127.0.0.1:18135 -> 127.0.0.1:19134", self.boot)
        self.assertIn("vllm proxy 127.0.0.1:18136 -> 127.0.0.1:19100", self.boot)
        self.assertIn("llamacpp proxy 127.0.0.1:18137 -> 127.0.0.1:19180", self.boot)


class OneEngineTests(unittest.TestCase):
    def test_engines_flag_skips_the_others(self):
        env = os.environ.copy()
        env.update({
            "STATS_DB": os.path.join(tempfile.mkdtemp(), "s.db"),
            "DASH_BIND": "127.0.0.1", "DASH_PORT": "18181",
            "PROXY_BIND": "127.0.0.1", "PROXY_PORT": "18141",
            "VLLM_PROXY_PORT": "18142", "LLAMACPP_PROXY_PORT": "18143",
            "ENGINES": "ollama",
        })
        proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "token_annihilator.py"), "--engines", "ollama"],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            boot = ""
            deadline = time.time() + 8
            while time.time() < deadline:
                line = proc.stdout.readline()
                boot += line
                if "dashboard http://" in line:
                    break
            self.assertIn("ollama proxy", boot)
            self.assertNotIn("vllm proxy", boot)
            self.assertNotIn("llamacpp proxy", boot)
            status, page = get(18181, "/")
            self.assertEqual(status, 200)
            text = page.decode()
            self.assertIn("Ollama", text)
            self.assertNotIn("backend=vllm", text)
            self.assertNotIn("vLLM", text)
        finally:
            proc.terminate()
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired: proc.kill()
            try: proc.stdout.close()
            except Exception: pass


if __name__ == "__main__":
    unittest.main()
