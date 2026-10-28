# Token Annihilation Dashboard

Your local GPU is out there annihilating tokens that a cloud provider would have happily billed you for. This keeps score.

A single-file, stdlib-only Python proxy that sits in front of [Ollama](https://ollama.com), counts every token that passes through, and serves a gloriously over-the-top dashboard (the "DGX Token Annihilator Dashboard") showing how many cloud dollars you did not set on fire. Built for an NVIDIA DGX Spark, works anywhere Ollama runs.

**Ollama is never modified.** No config changes, no restarts, no patches. The proxy just forwards requests to Ollama and reads the token counts on the way back. If it ever dies, point your client back at port 11434 and carry on.

## Features

- **Transparent proxy on port 11435** forwarding everything to Ollama on `127.0.0.1:11434`. Streaming responses pass through chunk by chunk, never buffered. Fails open: if logging breaks, your request still succeeds.
- Counts tokens for `/api/generate`, `/api/chat`, `/api/embed`, `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`, stored in a tiny SQLite file.
- **Dashboard on port 11480**: tokens annihilated (input/output), requests, output tokens/sec, daily devastation log, hungriest models.
- **GitHub-style contribution calendar** of tokens per day for the last year, in NVIDIA green, with hover tooltips.
- **Cloud bills obliterated**: estimated savings at list price vs Claude Opus 5.5, Claude Sonnet 5.5, GPT-6.1 Sol, GPT-6.1 Terra and Grok 4.7.
- **`prices.json`**: USD per 1M input/output tokens, last checked **2026-10-28** against each provider's official pricing page (sources inside). Edit it whenever prices change; the page rereads it on every load.
- Optional backfill of past request counts from the Ollama journal (no token counts exist there, so these are stored but never mixed into totals).

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
| Proxy bind | `PROXY_BIND` | `--proxy-bind` | `0.0.0.0` |
| Proxy port | `PROXY_PORT` | `--proxy-port` | `11435` |
| Dashboard bind | `DASH_BIND` | `--dash-bind` | output of `tailscale ip -4`, else `127.0.0.1` |
| Dashboard port | `DASH_PORT` | `--dash-port` | `11480` |
| Ollama host/port | `UPSTREAM_HOST` / `UPSTREAM_PORT` | | `127.0.0.1` / `11434` |

Then open `http://<your-tailscale-ip>:11480`.

## Point clients at it

Only requests that go through the proxy get counted. Use port 11435 instead of 11434:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama run <model>
```

OpenAI-compatible clients: set the base URL to `http://127.0.0.1:11435/v1`.

**Streaming with the OpenAI API?** Send `"stream_options": {"include_usage": true}`, otherwise the stream carries no usage numbers and the annihilation goes unrecorded. Native `/api/*` endpoints always report counts.

## Backfill (optional)

```bash
python3 token_annihilator.py --backfill $(date +%s)
```

Imports past request lines from `journalctl -u ollama` (timestamps, endpoints, durations only).

## Uninstall

```bash
systemctl --user disable --now token-annihilator
rm ~/.config/systemd/user/token-annihilator.service
rm -rf ~/token-annihilation-dashboard
```

Ollama was never touched, so there is nothing to undo there.

## License

MIT
