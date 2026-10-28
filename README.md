# Token Annihilation Dashboard

Your local GPU is out there annihilating tokens that a cloud provider would have happily billed you for. This keeps score.

A single-file, stdlib-only Python proxy that sits in front of [Ollama](https://ollama.com), [vLLM](https://docs.vllm.ai/), or [llama.cpp](https://github.com/ggml-org/llama.cpp)'s server, counts every token that passes through, and serves a gloriously over-the-top dashboard (the "DGX Token Annihilator Dashboard") showing how many cloud dollars you did not set on fire. Built for an NVIDIA DGX Spark. One process can watch all three engines at once.

**The local servers are never modified.** No config changes, no restarts, no patches. Each proxy just forwards requests and reads the token receipt on the way back. If a proxy ever dies, point that client back at the server's own port and carry on.

## Features

- **Three transparent proxies**, one per engine. Streaming responses pass through chunk by chunk, never buffered. Fails open: if logging breaks, your request still succeeds.
- Counts tokens for the native routes and the OpenAI-compatible ones, stored in a tiny SQLite file. Ollama's `prompt_eval_count` / `eval_count`, vLLM and llama.cpp OpenAI `usage`, the Responses and Anthropic `input_tokens` / `output_tokens` shapes, and llama.cpp's `tokens_evaluated` / `tokens_predicted` (plus `timings`) are all read. A batch reply is summed. A stream keeps the latest total, so partial chunks are not added twice.
- **Dashboard on port 11480**: tokens annihilated (input/output), requests, output tokens/sec, daily devastation log, hungriest models, and a card for each engine. Click an engine to see only its score.
- **GitHub-style contribution calendar** of tokens per day for the last year, in NVIDIA green, with hover tooltips.
- **Cloud bills obliterated**: estimated savings at list price vs Claude Opus 5.5, Claude Sonnet 5.5, GPT-6.1 Sol, GPT-6.1 Terra and Grok 4.7.
- **`prices.json`**: USD per 1M input/output tokens, last checked **2026-10-28** against each provider's official pricing page (sources inside). Edit it whenever prices change; the page rereads it on every load.
- Optional backfill of past request counts from the Ollama journal (no token counts exist there, so these are stored but never mixed into totals). vLLM and llama.cpp logs do not carry token counts, so only traffic through their proxies is scored.

## Install

Requires Python 3.8+ and nothing else.

```bash
git clone https://github.com/SaltShaker87/token-annihilation-dashboard ~/token-annihilation-dashboard
cd ~/token-annihilation-dashboard
python3 token_annihilator.py            # try it in the foreground
```

Run it as a systemd user service (no sudo):

```bash
mkdir -p ~/.config/systemd/user
cp token-annihilator.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now token-annihilator
loginctl enable-linger "$USER"   # keep running after logout (may need admin on some systems)
```

### Configuration

| Setting | Env var | Flag | Default |
|---|---|---|---|
| Proxy bind (all engines) | `PROXY_BIND` | `--proxy-bind` | `0.0.0.0` |
| Ollama proxy port | `PROXY_PORT` | `--proxy-port` | `11435` |
| vLLM proxy port | `VLLM_PROXY_PORT` | `--vllm-proxy-port` | `8001` |
| llama.cpp proxy port | `LLAMACPP_PROXY_PORT` | `--llamacpp-proxy-port` | `8081` |
| Which engines to listen for | `ENGINES` | `--engines` | `ollama,vllm,llamacpp` |
| Dashboard bind | `DASH_BIND` | `--dash-bind` | output of `tailscale ip -4`, else `127.0.0.1` |
| Dashboard port | `DASH_PORT` | `--dash-port` | `11480` |
| Ollama host/port | `UPSTREAM_HOST` / `UPSTREAM_PORT` | | `127.0.0.1` / `11434` |
| vLLM host/port | `VLLM_UPSTREAM_HOST` / `VLLM_UPSTREAM_PORT` | | `127.0.0.1` / `8000` |
| llama.cpp host/port | `LLAMACPP_UPSTREAM_HOST` / `LLAMACPP_UPSTREAM_PORT` | | `127.0.0.1` / `8080` |

`ENGINES` is a comma-separated list. `llama.cpp` and `llama` are accepted as aliases of `llamacpp`. An engine whose port is already taken is skipped; the others still start.

Then open `http://<your-tailscale-ip>:11480`.

## Point clients at it

Only requests that go through a proxy get counted. Leave each server on its own port and aim clients at the proxy one step up:

| Engine | Server port | Point clients at |
|---|---|---|
| Ollama | 11434 | 11435 |
| vLLM | 8000 | 8001 |
| llama.cpp | 8080 | 8081 |

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama run <model>
```

OpenAI-compatible clients, set the base URL to the matching proxy:

- Ollama: `http://127.0.0.1:11435/v1`
- vLLM: `http://127.0.0.1:8001/v1`
- llama.cpp: `http://127.0.0.1:8081/v1`

llama.cpp's own routes (`/completion`, `/infill`, `/embedding`, `/rerank`, and the Anthropic-shaped `/v1/messages`) are counted on the llama.cpp proxy too.

**Streaming with an OpenAI-style API?** The stream has to carry a usage total on its last chunk. Send `"stream_options": {"include_usage": true}`, otherwise the annihilation goes unrecorded. vLLM can also be started with `--enable-force-include-usage` so every reply includes that receipt. Native Ollama `/api/*` routes and llama.cpp's final `/completion` event always report counts. Output tokens per second show up when the reply includes a generation time (Ollama's `eval_duration`, llama.cpp's `timings`, or vLLM's `metrics.generation_time_ms` from `--enable-per-request-metrics`).

## Backfill (optional)

```bash
python3 token_annihilator.py --backfill $(date +%s)
```

Imports past request lines from `journalctl -u ollama` (timestamps, endpoints, durations only) and files them under Ollama.

## Uninstall

```bash
systemctl --user disable --now token-annihilator
rm ~/.config/systemd/user/token-annihilator.service
rm -rf ~/token-annihilation-dashboard
```

Ollama, vLLM, and llama.cpp were never touched, so there is nothing to undo there.

## License

MIT
