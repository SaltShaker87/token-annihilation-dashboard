# Token Annihilation Dashboard

Your local GPU is out there obliterating tokens that a cloud provider would have happily billed you for. This keeps score, with maximum drama.

One stdlib-only Python file that counts every token your local models chew through and serves the gloriously over-the-top **DGX Token Annihilator Dashboard**. Built on an NVIDIA DGX Spark, works anywhere Python 3.8+ runs.

Backends supported, each with its own source label:

| Backend | How it is counted | Label |
|---|---|---|
| **Ollama** | Transparent proxy (default `:11435` -> `:11434`) | Ollama |
| **vLLM** | OpenAI-compatible counting proxy (`OPENAI_PROXIES`) | vLLM |
| **llama.cpp** (`llama-server`) | Read-only tail of the server log (`LLAMA_LOG`) and/or the OpenAI-compatible proxy | llama.cpp |

**Your inference servers are never modified.** The proxies just forward requests and read the token counts on the way back, and the llama.cpp log tail never connects to the server at all. Everything fails open: if logging breaks, your request still succeeds. If the dashboard dies, point your client back at the real port and carry on.

## Features

- **Live odometer** of tokens obliterated (all time, never resets), live tokens/sec gauge and "Now annihilating" card over Server-Sent Events (throttled to ~4 updates/s, ~0.2% CPU idle).
- **Time filter**: 7 days / 30 days / 60 days / 90 days / All time. Filters the in/out/request/speed totals, cloud savings, model leaderboard, per-source chips and daily ledger, and dims calendar days outside the window. Your choice is remembered in the browser. The odometer and trophies always stay all-time.
- **Model leaderboard** (apex predators) with per-source chips: tokens, requests, average and best tok/s, biggest single output.
- **Trophy case**: 27 personal records (Speed Demon, Night Owl, Week of Wrath, Iron Streak, Reigning Champion...).
- **Full-screen milestone celebrations** with a muted looping video, confetti and dramatic copy: 1M and 10M tokens, every 100M, a much bigger Total Eradication show at every 1B, and every $100 saved. "🎉 test the party" in the footer previews it.
- **Trip meter** with a reset button and a "Glovebox" of past trips. Nothing is ever deleted.
- **📸 Share card**: one click downloads a 1200x630 PNG of your carnage.
- **GitHub-style calendar** of tokens per day for the last year, in NVIDIA green.
- **Cloud invoices eradicated** vs Claude Opus 5.5, Claude Sonnet 5.5, GPT-6.1 Sol, GPT-6.1 Terra and Grok 4.7, from `prices.json` (list prices checked 2026-10-28, sources inside; edit freely).

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

### vLLM

Start vLLM as usual (say on port 8000) and add a counting proxy in front of it:

```bash
OPENAI_PROXIES="vllm:8001=127.0.0.1:8000" python3 token_annihilator.py
```

Then point your clients at `http://127.0.0.1:8001/v1`. The proxy reads the `usage` field from every response. For streaming requests it automatically adds `"stream_options": {"include_usage": true}` (only when the client did not set `stream_options` itself) so the final chunk carries the token counts. Disable that with `INJECT_USAGE=0`.

### llama.cpp

Two options, use either or both:

1. **Log tail (zero contact with the server).** Save llama-server's output to a file and point `LLAMA_LOG` at it:

   ```bash
   llama-server -m model.gguf --port 8080 2>&1 | tee ~/llama_server.log
   LLAMA_LOG=~/llama_server.log python3 token_annihilator.py
   ```

   The file is polled every `LLAMA_POLL_S` seconds (default 15), and the model name comes from the GGUF file name. To import a log that already exists, run `python3 token_annihilator.py --llama-backfill` once.

2. **Counting proxy**, same as vLLM: `OPENAI_PROXIES="llamacpp:8081=127.0.0.1:8080"`, then use port 8081.

Don't count the same server both ways at once, or its tokens are counted twice. Several proxies can be combined: `OPENAI_PROXIES="vllm:8001=127.0.0.1:8000,llamacpp:8081=127.0.0.1:8080"`.

## Configuration

| Env var | Default | What |
|---|---|---|
| `DASH_BIND` / `DASH_PORT` | Tailscale IP or `127.0.0.1` / `11480` | Dashboard address |
| `PROXY_BIND` / `PROXY_PORT` | `0.0.0.0` / `11435` | Ollama proxy listen address |
| `UPSTREAM_HOST` / `UPSTREAM_PORT` | `127.0.0.1` / `11434` | Where Ollama lives |
| `OLLAMA_PROXY` | `1` | `0` disables the Ollama proxy |
| `OPENAI_PROXIES` | empty | `source:listen_port=upstream_host:port`, comma-separated |
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

## Uninstall

```bash
systemctl --user disable --now token-annihilator
rm ~/.config/systemd/user/token-annihilator.service
rm -rf ~/token-annihilation-dashboard
```

Your inference servers were never touched, so there is nothing to undo there.

## License

MIT
