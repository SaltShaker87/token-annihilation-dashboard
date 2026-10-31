# Token Annihilation Dashboard

<p align="center">
  <img src="logo.png" alt="A little chip breathing out dollar signs" width="180">
</p>

Your local GPU is out there obliterating tokens that a cloud provider would have happily billed you for. This keeps score, with maximum drama.

One stdlib-only Python file that counts every token your local models chew through and serves the gloriously over-the-top **DGX Token Annihilator Dashboard**. Built on an NVIDIA DGX Spark, works anywhere Python 3.8+ runs.

Backends supported, each with its own source label:

| Backend | How it is counted | Label |
|---|---|---|
| **Ollama** | Transparent proxy (default `:11435` -> `:11434`) | Ollama |
| **vLLM** | OpenAI-compatible counting proxy (`OPENAI_PROXIES`) | vLLM |
| **llama.cpp** (`llama-server`) | Read-only tail of the server log (`LLAMA_LOG`) and/or the OpenAI-compatible proxy | llama.cpp |
| **NVIDIA NIM / NeMo, TensorRT-LLM, Triton (OpenAI front end), SGLang, LM Studio** | OpenAI-compatible counting proxy (`OPENAI_PROXIES`) | NVIDIA NIM, NeMo, TensorRT-LLM, ... |
| **ONNX Runtime** (Foundry Local, onnxruntime-genai OpenAI servers) | OpenAI-compatible counting proxy (`OPENAI_PROXIES`) | ONNX Runtime |

Anything that speaks `/v1/chat/completions`, `/v1/completions`, `/v1/responses` or `/v1/embeddings` and returns a `usage` field is counted exactly. Run as many runtimes at once as you like; each one gets its own label, leaderboard chip and status chip.

**Your inference servers are never modified.** The proxies just forward requests and read the token counts on the way back, and the llama.cpp log tail never connects to the server at all. Everything fails open: if logging breaks, your request still succeeds. If the dashboard dies, point your client back at the real port and carry on.

## Features

- **Live odometer** of tokens obliterated (all time, never resets), live tokens/sec gauge and "Now annihilating" card over Server-Sent Events (throttled to ~4 updates/s, ~0.2% CPU idle).
- **Time filter**: 24 hours (rolling) / 7 days / 30 days / 60 days / 90 days / All time. Filters the in/out/request/speed totals, cloud savings, model leaderboard, per-source chips and daily ledger, and dims calendar days outside the window. Your choice is remembered in the browser. The odometer and trophies always stay all-time.
- **Model leaderboard** (apex predators) with per-source chips: tokens, requests, average and best tok/s, biggest single output.
- **Trophy case**: 27 personal records (Speed Demon, Night Owl, Week of Wrath, Iron Streak, Reigning Champion...).
- **Full-screen milestone celebrations** with a muted looping video, confetti and dramatic copy: 1M and 10M tokens, every 100M, a much bigger Total Eradication show at every 1B, and every $100 saved. "🎉 test the party" in the footer previews it.
- **Current killing spree**: tokens, victims (requests), savings and average speed since you last ended a spree. Past sprees rest in "The mass graves". Nothing is ever deleted.
- **📸 Share card**: previews a 1200x630 PNG of your carnage, then download it, copy it, or use your phone's share sheet (copy and share need https or localhost).
- **Runtime status chips** next to the time filter: which proxies are up, and a warning when a runtime you have used stops answering.
- **Odometer that keeps up**: adds a reel whenever the count gains a digit, shrinks the reels to fit the card (phones too), shows a plain-English caption ("≈ 50.8 million tokens"), and on very narrow screens rolls in millions/billions with a unit badge.
- **GitHub-style calendar** of tokens per day for the last year, in NVIDIA green.
- **Cloud invoices eradicated** vs Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra, Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 3.1 Pro, GPT-6.1 Terra and Grok 4.7, from `prices.json` (list prices checked 2026-10-28, sources inside; edit freely).

## Install

```bash
git clone https://github.com/SaltShaker87/token-annihilation-dashboard ~/token-annihilation-dashboard
cd ~/token-annihilation-dashboard
python3 token_annihilator.py            # try it in the foreground
```

Open `http://<your-tailscale-ip>:11480` (it binds to your Tailscale IP if `tailscale` is installed, else `127.0.0.1`).

Run it as a systemd user service (no sudo). Edit the `Environment=` lines in the unit file first if you use vLLM or llama.cpp:

```bash
mkdir -p ~/.config/systemd/user
cp token-annihilator.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now token-annihilator
loginctl enable-linger "$USER"   # keep running after logout
```

## Backend setup

### Ollama

On by default. Point clients at port **11435** instead of 11434:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama run <model>
```

OpenAI-style clients: base URL `http://127.0.0.1:11435/v1`. Not using Ollama? Set `OLLAMA_PROXY=0`.

To make every new shell use the proxy, add this to `~/.bashrc`:

```bash
export OLLAMA_HOST=127.0.0.1:11435
```

### vLLM

Automatic: as soon as vLLM answers on port 8000, a counting proxy opens on **8001** in front of it (checked every 30 s, so vLLM can start after the dashboard). Point your clients at `http://127.0.0.1:8001/v1`. Different ports or host (say vLLM runs across two Sparks with the API on the head node):

```bash
OPENAI_PROXIES="vllm:8001=spark-1:8000" python3 token_annihilator.py
```

The proxy reads the `usage` field from every response. For streaming requests it automatically adds `"stream_options": {"include_usage": true}` (only when the client did not set `stream_options` itself) so the final chunk carries the token counts. Disable that with `INJECT_USAGE=0`. vLLM reports no timings, so for streamed requests the speed (tok/s) is measured from the first to the last streamed chunk.

### llama.cpp

Two options, use either or both:

1. **Log tail (zero contact with the server).** Save llama-server's output to a file and point `LLAMA_LOG` at it:

   ```bash
   llama-server -m model.gguf --port 8080 2>&1 | tee ~/llama_server.log
   LLAMA_LOG=~/llama_server.log python3 token_annihilator.py
   ```

   The file is polled every `LLAMA_POLL_S` seconds (default 15), and the model name comes from the GGUF file name. To import a log that already exists, run `python3 token_annihilator.py --llama-backfill` once.

2. **Counting proxy**, opened automatically on **8081** once llama-server answers on 8080 (not when `LLAMA_LOG` is set), then use port 8081.

Don't count the same server both ways at once, or its tokens are counted twice.

Proxy pattern in practice: llama-server stays on **8080** (its start line and any health checks keep using 8080), and only the clients that send prompts move to **8081**, for example `--base-url http://127.0.0.1:8081/v1`. Streaming passes straight through (about one token per chunk), and the proxy adds only a few percent to total request time. If you want the proxy fixed instead of auto-detected, list it yourself (optional; this replaces the automatic vLLM proxy too, so add `vllm:8001=127.0.0.1:8000` to the list if you also run vLLM):

```bash
OPENAI_PROXIES="llamacpp:8081=127.0.0.1:8080" python3 token_annihilator.py
```

The proxy ports are API passthroughs, not the dashboard. Opening `/` on 8081 shows llama-server's own web UI (curl without `Accept-Encoding: gzip` gets llama-server's "gzip is not supported"), and `/api/stats` there is a 404 from llama-server. The dashboard and `/api/stats` live on port 11480.

### NVIDIA NIM / NeMo, ONNX Runtime and other OpenAI-compatible servers

Give each one a name and a port in `OPENAI_PROXIES`. Known names get a pretty label (`nim`, `nemo`, `trtllm`, `triton`, `onnx`, `sglang`, `lmstudio`, `openai`); any other name is shown as written.

```bash
OPENAI_PROXIES="vllm:8001=127.0.0.1:8000,nim:8002=127.0.0.1:9000,onnx:8003=127.0.0.1:5273" python3 token_annihilator.py
```

Raw Triton KServe (`/v2/models/.../generate`) and in-process ONNX Runtime return no token counts, so they can't be counted. Put Triton's OpenAI front end or an OpenAI-compatible ONNX server in front.

### Several runtimes at once

The Ollama proxy (11435) starts by default. The vLLM (8001) and llama.cpp (8081) proxies open on their own once a real vLLM or llama-server answers on 8000 / 8080, so running a big vLLM model and a small Ollama model side by side needs no config, and an unrelated app on 8000 or 8080 is never proxied or exposed. Setting `OPENAI_PROXIES` replaces those automatic proxies with exactly what you list, started right away with no detection (`OPENAI_PROXIES=` with nothing after it turns them all off). The 11435 proxy labels itself by whatever actually answers upstream (if you point `UPSTREAM_PORT` at vLLM it says vLLM, not Ollama); force it with `PROXY_SOURCE`.

**Only traffic that goes through a proxy is counted.** Requests sent straight to the real servers (Ollama on 11434, llama-server on 8080, vLLM on 8000) bypass the dashboard and are not counted, unless you use the llama.cpp log tail.

Upgrading from the earlier multi-engine build? Its `ENGINES`, `VLLM_*` and `LLAMACPP_*` settings still work, and rows it saved are relabelled from Ollama to their real runtime the first time this version starts.

## Configuration

| Env var | Default | What |
|---|---|---|
| `DASH_BIND` / `DASH_PORT` | Tailscale IP or `127.0.0.1` / `11480` | Dashboard address |
| `PROXY_BIND` / `PROXY_PORT` | `0.0.0.0` / `11435` | Ollama proxy listen address |
| `UPSTREAM_HOST` / `UPSTREAM_PORT` | `127.0.0.1` / `11434` | Where Ollama lives |
| `OLLAMA_PROXY` | `1` | `0` disables the Ollama proxy |
| `OPENAI_PROXIES` | unset: `vllm:8001=127.0.0.1:8000,llamacpp:8081=127.0.0.1:8080`, each opened only once that runtime is detected | `source:listen_port=upstream_host:port`, comma-separated, started right away. Set empty for none |
| `PROXY_SOURCE` | detected | Force the label of the 11435 proxy (`ollama`, `vllm`, ...) |
| `ENGINES` | `ollama,vllm,llamacpp` | Older-build switch; used only when `OPENAI_PROXIES` is unset |
| `INJECT_USAGE` | `1` | Ask streaming `/v1` requests to include usage |
| `LLAMA_LOG` | empty (off) | llama-server log file to tail |
| `LLAMA_POLL_S` | `15` | Log poll interval, seconds |
| `LLAMA_MODEL` | `llama.cpp` | Model name if the log has no load line |
| `LLAMA_STARTS` | empty | Optional comma-separated epoch start times for exact `--llama-backfill` timestamps |
| `STATS_DB` | `./stats.db` | SQLite file |

## Backfill Ollama history (optional)

```bash
python3 token_annihilator.py --backfill $(date +%s)
```

Imports past request lines from `journalctl -u ollama`. Those lines have no token counts, so they are stored but never mixed into totals.

## Tests

```bash
python3 test_annihilator.py
```

Stdlib only. Spins up fake Ollama, vLLM and llama.cpp servers on local ports and checks the counting end to end.

## Uninstall

```bash
systemctl --user disable --now token-annihilator
rm ~/.config/systemd/user/token-annihilator.service
rm -rf ~/token-annihilation-dashboard
```

Your inference servers were never touched, so there is nothing to undo there.

## License

MIT
