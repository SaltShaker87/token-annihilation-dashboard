# Token Annihilation Dashboard

<p align="center">
  <img src="static/icon.webp" alt="An angry chip robot inhaling a stream of tokens" width="200">
</p>

Your local GPU is out there obliterating tokens that a cloud provider would have happily billed you for. This keeps score, with maximum drama.

One stdlib-only Python file that counts every token your local models chew through and serves the gloriously over-the-top **DGX Token Annihilator Dashboard**. Built on an NVIDIA DGX Spark, works anywhere Python 3.8+ runs.

Backends supported, each with its own source label:

| Backend | How it is counted | Label |
|---|---|---|
| **Ollama** | Transparent proxy (default `:11435` -> `:11434`) | Ollama |
| **vLLM** | Counting proxy. Default `:8001` -> `:8000`. Opens once vLLM answers. Change either port with `VLLM_PROXY_PORT` and `VLLM_UPSTREAM_PORT` | vLLM |
| **SGLang** | Counting proxy, same idea as vLLM. Default `:30001` -> `:30000`. Opens once SGLang answers. Change either port with `SGLANG_PROXY_PORT` and `SGLANG_UPSTREAM_PORT` | SGLang |
| **TensorFold** (MiaAI-Lab's Zig engine, OpenAI-compatible) | Same counting proxy as vLLM. Detected from `owned_by: tensorfold` in `/v1/models` or a `Server: TensorFold` header | TensorFold |
| **llama.cpp** (`llama-server`) | Read-only read of llama-server's log. Default file: `~/pixpilot/run/server.log`. The 8081 proxy is optional and stays off while a log path is set | llama.cpp |
| **NVIDIA NIM / NeMo, TensorRT-LLM, Triton (OpenAI front end), LM Studio** | OpenAI-compatible counting proxy (`OPENAI_PROXIES`) | NVIDIA NIM, NeMo, TensorRT-LLM, ... |
| **ONNX Runtime** (Foundry Local, onnxruntime-genai OpenAI servers) | OpenAI-compatible counting proxy (`OPENAI_PROXIES`) | ONNX Runtime |

Anything that speaks `/v1/chat/completions`, `/v1/completions`, `/v1/responses` or `/v1/embeddings` and returns a `usage` field is counted exactly. Run as many runtimes at once as you like; each one gets its own label, leaderboard chip and status chip.

**Your inference servers are never modified.** The proxies just forward requests and read the token counts on the way back, and the llama.cpp log tail never connects to the server at all. Everything fails open: if logging breaks, your request still succeeds. If the dashboard dies, point your client back at the real port and carry on.

## Features

- **Live odometer** of tokens obliterated (all time on the default view, or the window you pick in the time filter), a tokens/sec gauge averaged over the last 60 seconds with a sparkline, and a "Now annihilating" card, all over Server-Sent Events (throttled to ~4 updates/s, ~0.2% CPU idle).
- **Header mascot**: the angry chip robot at the top left plays a short inhale loop while a request is in flight (including a non-streamed request that is still waiting) and sits on the idle still otherwise. A brief gap between requests does not restart the loop. If the live feed drops, or the browser asks for reduced motion, it stays on the still.
- **Time filter**: 24 hours (rolling) / 7 days / 30 days / 60 days / 90 days / All time. Filters the odometer, the in/out/request/speed totals, cloud savings, model leaderboard, per-source chips and daily ledger, and dims calendar days outside the window. Your choice is remembered in the browser. Trophies and milestones always stay all-time.
- **Extinction events survived** (milestone badges): collapsed by default to the most recent few in one row (`MS_PREVIEW`), with a "View all" toggle that expands and collapses the full list.
- **Settings window** (⚙ button, bottom-right corner): change the milestone tiers, one-off milestones and badge preview count without editing files or env vars. Entries accept K/M/B/T shorthand (`5B`), are validated on the server, apply instantly and are remembered in `stats.db`. "Reset to defaults" forgets the saved settings.
- **Model leaderboard** (apex predators) with per-source chips: tokens, requests, average and best tok/s, biggest single output.
- **Trophy case**: 27 personal records (Speed Demon, Night Owl, Week of Wrath, Iron Streak, Reigning Champion...).
- **Full-screen milestone celebrations** with a muted looping video, confetti and dramatic copy. They thin out as your totals grow: dollars saved every $100 to $1,000, every $500 to $10,000, then every $10,000; tokens at 1M, 10M, 100M and 500M, then whole billions only (every 1B to 10B, every 5B to 100B, every 50B after that). Opening the page never replays a backlog: at most one celebration, for the most recent milestone you missed, and the rest are marked seen. The schedule is configurable (`MS_*` below). "🎉 test the party" in the footer previews it.
- **Current killing spree**: tokens, victims (requests), savings and average speed since you last pressed "☠ End this spree" (you can name it). Past sprees rest in "The mass graves". Nothing is ever deleted.
- **📸 Share card**: previews a 1200x630 PNG of your carnage, then download it, copy it, or use your phone's share sheet (copy and share need https or localhost).
- **Runtime status chips** next to the time filter: which proxies are up, and a warning when a runtime you have used stops answering.
- **Odometer that keeps up**: one reel per digit (added or removed as the count changes), reels shrink to fit the card (phones too), a plain-English caption ("≈ 50.8 million tokens"), and scientific notation (d.ddd × 10^e) from 14 digits up or when the reels would get too small to read.
- **GitHub-style calendar** of tokens per day for the last year, in NVIDIA green.
- **Cloud invoices eradicated** vs Claude Opus 5.5, Claude Fable 5.1, GPT-6 Astra, Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 3.1 Pro, GPT-6.1 Terra and Grok 4.7, from `prices.json` (USD per 1M tokens, list prices checked 2026-10-28, sources inside; edit freely, it is re-read on every page load). The first entry is the headline number and drives the $100 milestones.

## Install

```bash
git clone https://github.com/SaltShaker87/token-annihilation-dashboard ~/token-annihilation-dashboard
cd ~/token-annihilation-dashboard
python3 token_annihilator.py            # try it in the foreground
```

Open `http://<your-tailscale-ip>:11480` (it binds to your Tailscale IP if `tailscale` is installed, else `127.0.0.1`).

Run it as a systemd user service (no sudo). Edit the `Environment=` lines in the unit file first if vLLM or SGLang is not on the usual port, or if llama-server's log is not `~/pixpilot/run/server.log`:

```bash
mkdir -p ~/.config/systemd/user
cp token-annihilator.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now token-annihilator
loginctl enable-linger "$USER"   # keep running after logout
```

## Updates

The bottom of the dashboard has an **Update** button and a small hint: **up to date**, **update available**, or **no release found**. Opening the page checks once. Nothing keeps asking GitHub in the background.

**No release found** means GitHub returned no published release (for example on a fork with none). The button then tells you the installed version and does not change any files.

When a newer release exists, the button shows the version you have next to the latest one and asks you to confirm. Only after that does it:

- stop, and name the files, if the install folder has uncommitted edits (it will not overwrite them)
- stop if you have local commits that the release does not already contain
- download that release's git tag from the official repo only (`github.com/SaltShaker87/token-annihilation-dashboard`). A version, link, or branch sent by the browser is ignored. There is no setting to point this at another URL
- keep the previous checkout under `refs/backup/token-annihilator-prev` and put it back if the new files do not match the release
- restart itself, and only after the new files match the release. If systemd is running it and that service is set to come back (the included unit has `Restart=always`; the service can have any name), the process exits and systemd starts it again. If you started it in a terminal, it reloads itself in place. The page notices, reconnects, and shows the new version. Only if that fails does it ask you to start it again yourself

Your stats database stays. Ollama, llama.cpp, vLLM, SGLang, their ports, and any other service are not restarted or reconfigured.

To undo an update that will not start:

```bash
cd ~/token-annihilation-dashboard
git checkout --detach refs/backup/token-annihilator-prev
systemctl --user restart token-annihilator
```

### Versioning and cutting a release

The version is the `VERSION` file in the repo root: three numbers and nothing else, such as `1.2.0`. The git tag for a release is `v` plus the same number (`v1.2.0`), and the `VERSION` file on that tag has to say the same thing. The running dashboard reads `VERSION`. It finds the latest release with the public GitHub API, which needs no token. Drafts and pre-releases are ignored, and a tag on its own is not enough: the API only sees a published GitHub Release.

```bash
# on main: set VERSION to the new number, commit, tag, and push
git add VERSION && git commit -m "v1.2.0"
git tag v1.2.0
git push origin main --tags
```

On GitHub: Releases → Draft a new release → choose the tag `v1.2.0` → Publish release. Open the dashboard and the hint changes to **update available**.

## Backend setup

### Counting addresses

The usual way needs no port numbers from you. Leave the dashboard running. Every few seconds it looks at which ports on this machine are open, and asks each one (with a single read of `/v1/models`) whether a model server is there. For each one it finds, it opens a **counting address** on the next free port. vLLM, SGLang, TensorFold, llama.cpp's server, and anything else that answers like them get their own label.

The page says this in yellow, because it is the whole trick: **a program is counted only when it uses the counting address.** Traffic sent straight to the model server is invisible. The counting addresses, with a Copy button, are in Settings (the gear).

The model server is never restarted, never sent a prompt, and never has its own port taken over.

**Chat, completions, and Anthropic-style messages.** The counting address counts `/v1/chat/completions`, `/v1/completions`, and `/v1/messages`, streaming or not, from the `usage` numbers in the answer. On an OpenAI-style stream it asks for those numbers (`stream_options.include_usage`) when the program did not already ask. An Anthropic-style `/v1/messages` stream is left as the program sent it; the counts are read from the final event. A Maya server is recognized from its model list (a card with `architecture` and `status`, and no `owned_by`) and labelled **Maya**. Any other list is still counted, under **OpenAI-compatible** or the name the server gives.

**Traffic that skips the counting address.** Some servers publish a running token total (SGLang's `/metrics` when it was started with `--enable-metrics`, vLLM's `/metrics`, or a Prometheus page at `/metrics/prometheus`). The dashboard reads that total every half minute. The first read is only a starting point. Later growth is counted, except the tokens the counting address already wrote down, so a request is not counted twice. Tokens from before the dashboard was watching are not added. A server with no counter, and a stream that never sends `usage`, is counted only when the program uses the counting address.

Ollama is unchanged: it is still counted through port **11435**. llama.cpp is still counted from its log when that log is set, so the same requests are not counted twice.

**Ports that are left alone.** These are never checked and never given a counting address:

- Ollama's own port **11434**, the Ollama counting port **11435**, and the dashboard port (**11480**, or whatever you set)
- every counting port the dashboard itself opened
- anything you add

On a machine that also runs some other app, add that app's ports. For example an app on **3000**, and something already using **8001**:

```bash
PORT_IGNORE=3000,8001 python3 token_annihilator.py
```

Or type `3000, 8001` into **Ports to leave alone** in Settings and press Save. A port you add there is never probed and never proxied. The same window can turn the search off entirely (uncheck "Look for model servers on this machine"), add a server the search did not notice, or pick the counting port yourself.

If the counting port next to a server is already taken, the dashboard prints a warning, uses the next free port, and keeps running. Nothing else stops.

**The old port variables still work**, if you would rather name them yourself. With the search on, they are used as well as the search. With the search off (`AUTO_DETECT=0`), the dashboard goes back to the fixed ports: vLLM **8001** in front of **8000**, SGLang **30001** in front of **30000**.

```bash
AUTO_DETECT=0 python3 token_annihilator.py
VLLM_UPSTREAM_PORT=8010 VLLM_PROXY_PORT=8011 python3 token_annihilator.py
SGLANG_UPSTREAM_PORT=30000 SGLANG_PROXY_PORT=30011 python3 token_annihilator.py
OPENAI_PROXIES="vllm:8011=127.0.0.1:8010,sglang:30011=127.0.0.1:30000" python3 token_annihilator.py
```

`vllm bench serve` still has to use the counting port, and it adds `/v1/completions` itself:

```bash
vllm bench serve --host 127.0.0.1 --port COUNTING_PORT --model YOUR_MODEL
```

In an OpenAI-style program the base URL is `http://127.0.0.1:COUNTING_PORT/v1`.

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

vLLM stays where it is. The usual port is **8000**. The dashboard does not change vLLM. With the search on, the counting address is the next free port (8001 when 8001 is free) and it is shown in Settings. With the search off, it opens **8001**. Either way it writes down tokens that pass through that address.

A request sent straight to 8000 is invisible here. Point the program at 8001.

In an OpenAI-style program the base URL is `http://127.0.0.1:8001/v1` (the program adds `/chat/completions` or `/completions` itself).

`vllm bench serve` talks to `/v1/completions`. Point it at the counting port, not at vLLM's own port. It adds `/v1/completions` itself, so do not put `/v1` on the host:

```bash
vllm bench serve --host 127.0.0.1 --port 8001 --model YOUR_MODEL
```

Streaming and ordinary (non-streaming) chat and completions both count. The number comes from the `usage` field (`prompt_tokens` and `completion_tokens`). For a stream, the dashboard asks for that field by adding `"stream_options": {"include_usage": true}` when the program did not set `stream_options` itself. `vllm bench serve` already asks. Turn the extra ask off with `INJECT_USAGE=0`. vLLM reports no timings, so for streamed requests the speed (tok/s) is measured from the first to the last streamed chunk.

Different machine or ports (vLLM is on 8010, and 8001 is already taken by something else):

```bash
VLLM_UPSTREAM_HOST=127.0.0.1 VLLM_UPSTREAM_PORT=8010 VLLM_PROXY_PORT=8011 python3 token_annihilator.py
vllm bench serve --host 127.0.0.1 --port 8011 --model YOUR_MODEL
```

`VLLM_UPSTREAM_HOST` is the machine vLLM runs on. `VLLM_UPSTREAM_PORT` is the port vLLM is already using. `VLLM_PROXY_PORT` is the port your programs should use.

Leave the variables unset to keep 8001 and 8000. The counting port opens on its own once vLLM is actually answering (checked every 30 s, so vLLM can start later). A machine that is not running vLLM never takes port 8001.

If the counting port is already taken, the dashboard prints a warning like `vLLM proxy skipped: port 8001 is already in use` and keeps going. The page and every other counter stay up. A yellow chip on the page says the port is busy. Pick a free `VLLM_PROXY_PORT`.

Swapping vLLM and TensorFold on the same port is fine: the proxy is relabelled to whichever one answers (re-checked every 30 s), so each new request is logged under the right name. Old rows keep the label they were logged with. Other engines are never relabelled.

### SGLang

Same idea as vLLM. SGLang usually listens on **30000**. With the search on, the counting address is the next free port (30001 when 30001 is free). With the search off, it opens **30001** once SGLang is answering. The tokens are labelled **SGLang**.

Point programs at 30001, not 30000. A request sent straight to 30000 is not counted.

OpenAI-style base URL: `http://127.0.0.1:30001/v1`

A completions benchmark, including `vllm bench serve` aimed at SGLang, uses the counting port:

```bash
vllm bench serve --host 127.0.0.1 --port 30001 --model YOUR_MODEL
```

Streaming and ordinary `/v1/chat/completions` and `/v1/completions` both count, from the same `usage` field. Streaming requests get the same `include_usage` ask as vLLM.

SGLang on another port, or 30001 already taken:

```bash
SGLANG_UPSTREAM_HOST=127.0.0.1 SGLANG_UPSTREAM_PORT=30000 SGLANG_PROXY_PORT=30011 python3 token_annihilator.py
```

Then the base URL is `http://127.0.0.1:30011/v1`.

Leave the variables unset to keep 30001 and 30000. The counting port waits until SGLang answers, so an unused 30001 is left alone. If 30001 is busy when SGLang is detected, that one proxy is skipped with a warning (`Set SGLANG_PROXY_PORT to a free port`) and everything else keeps running.

If the dashboard never notices SGLang, name the ports yourself. This list replaces the automatic vLLM and SGLang ports, so add vLLM back if you use both:

```bash
OPENAI_PROXIES="vllm:8001=127.0.0.1:8000,sglang:30001=127.0.0.1:30000" python3 token_annihilator.py
```

### llama.cpp

Counted from llama-server's own log. Nothing sits in front of the server, and the server does not have to be running.

By default the dashboard reads `~/pixpilot/run/server.log` (Locally's harness writes llama-server's output there). A missing file or a missing folder is fine: the dashboard keeps running and starts counting when the file appears. It only reads the file. It does not write it, lock it, or talk to port 8080.

The model name is the GGUF file name from the log's `loading model '...'` line. Prompt tokens are the `prompt eval time` count (tokens the server actually processed). Answer tokens are the `eval time` count. The `stop processing: n_tokens` figure is the slot's context size, not the size of that request, so it is not used as the prompt.

The file is checked every `LLAMA_POLL_S` seconds once caught up (default 15). A long log is read in pieces (`LLAMA_READ_MAX` bytes, default 2 MB) so catching up does not stall the page. Reading the same log again does not count twice, including after a restart. If the file is replaced or rotated (`server.log` renamed to `server.log.1`), the unread tail of the old file is finished and the new file is read from the start. A compressed `.gz` rotation is not read.

Different log path, or turn the tail off:

```bash
LLAMA_LOG=~/llama_server.log python3 token_annihilator.py
LLAMA_LOG= python3 token_annihilator.py          # tail off
```

To import a log in one shot: `python3 token_annihilator.py --llama-backfill`.

**Counting proxy (optional).** With the log tail on, the automatic proxy on **8081** stays off so the same requests are not counted twice. Set `LLAMA_LOG` empty (and do not set `OPENAI_PROXIES`) to get the old behaviour: a proxy opens on **8081** once llama-server answers on 8080, and clients must use 8081. Don't point clients at 8081 while a log tail of that same server is also on.

Proxy pattern in practice: llama-server stays on **8080** (its start line and any health checks keep using 8080), and only the clients that send prompts move to **8081**, for example `--base-url http://127.0.0.1:8081/v1`. Streaming passes straight through (about one token per chunk), and the proxy adds only a few percent to total request time. If you want the proxy fixed instead of auto-detected, list it yourself (optional; this replaces the automatic vLLM and SGLang proxies too, so add those to the list if you also run them):

```bash
OPENAI_PROXIES="llamacpp:8081=127.0.0.1:8080" python3 token_annihilator.py
```

The proxy ports are API passthroughs, not the dashboard. Opening `/` on 8081 shows llama-server's own web UI (curl without `Accept-Encoding: gzip` gets llama-server's "gzip is not supported"), and `/api/stats` there is a 404 from llama-server. The dashboard and `/api/stats` live on port 11480.

### NVIDIA NIM / NeMo, ONNX Runtime and other OpenAI-compatible servers

Give each one a name and a port in `OPENAI_PROXIES`. Known names get a pretty label (`vllm`, `sglang`, `nim`, `nemo`, `trtllm`, `triton`, `onnx`, `lmstudio`, `openai`); any other name is shown as written.

```bash
OPENAI_PROXIES="vllm:8001=127.0.0.1:8000,sglang:30001=127.0.0.1:30000,nim:8002=127.0.0.1:9000,onnx:8003=127.0.0.1:5273" python3 token_annihilator.py
```

Raw Triton KServe (`/v2/models/.../generate`) and in-process ONNX Runtime return no token counts, so they can't be counted. Put Triton's OpenAI front end or an OpenAI-compatible ONNX server in front.

### Several runtimes at once

The Ollama proxy (11435) starts by default. With the search on (the normal setting), a vLLM, SGLang, TensorFold or other OpenAI-compatible server is given a counting port as soon as it is noticed, on the next free port beside it. Turn the search off with `AUTO_DETECT=0` and the older fixed ports come back: vLLM **8001** once vLLM answers on **8000**, SGLang **30001** once SGLang answers on **30000**, and the llama.cpp proxy **8081** only when the log tail is off. An unrelated app is never proxied. Setting `OPENAI_PROXIES` adds exactly the proxies you list (`OPENAI_PROXIES=` with nothing after it adds none). The 11435 proxy labels itself by whatever actually answers there (vLLM, SGLang, llama.cpp, or Ollama). Force it with `PROXY_SOURCE`.

**Ollama, vLLM and SGLang are counted only when the request goes through their counting port.** llama-server on 8080 is counted from its log, with no proxy. Requests sent straight to Ollama on 11434, vLLM on 8000, or SGLang on 30000 still bypass the dashboard.

A counting port that is already taken does not stop the dashboard. That one proxy is skipped, a warning is printed, and the others start as usual.

Upgrading from the earlier multi-engine build? Its `ENGINES`, `VLLM_*` and `LLAMACPP_*` settings still work, and rows it saved are relabelled from Ollama to their real runtime the first time this version starts. SGLang is on by default now. An `ENGINES` list you set yourself is used as written, so add `sglang` to that list if you want the automatic SGLang port.

## Configuration

| Env var | Default | What |
|---|---|---|
| `DASH_BIND` / `DASH_PORT` | Tailscale IP or `127.0.0.1` / `11480` | Dashboard address |
| `PROXY_BIND` / `PROXY_PORT` | `0.0.0.0` / `11435` | Ollama proxy listen address |
| `UPSTREAM_HOST` / `UPSTREAM_PORT` | `127.0.0.1` / `11434` | Where Ollama lives |
| `OLLAMA_PROXY` | `1` | `0` disables the Ollama proxy |
| `AUTO_DETECT` | `1` | Look for model servers on this machine and open a counting address beside each one. `0` turns that off and brings back the fixed vLLM / SGLang ports |
| `PORT_IGNORE` | empty (Ollama `11434`, Ollama counter `11435`, and the dashboard port are always skipped) | Extra ports to never check and never proxy, comma-separated. Example: `3000,8001` |
| `SCAN_EVERY` | `5` | Seconds between looks. The look is a read of which ports are open, then one short `/v1/models` read on anything new |
| `OPENAI_PROXIES` | unset while auto-detect is on: none, the search opens them. With `AUTO_DETECT=0`: vLLM `8001` -> `127.0.0.1:8000`, SGLang `30001` -> `127.0.0.1:30000`, and llama.cpp `8081` -> `127.0.0.1:8080` only when `LLAMA_LOG` is empty | `name:listen_port=host:upstream_port`, comma-separated, started right away. Set empty for none. A busy listen port is skipped |
| `VLLM_PROXY_PORT` | `8001` | Port your programs use for vLLM. Used when `OPENAI_PROXIES` is unset |
| `VLLM_UPSTREAM_HOST` / `VLLM_UPSTREAM_PORT` | `127.0.0.1` / `8000` | Where vLLM itself is listening |
| `SGLANG_PROXY_PORT` | `30001` | Port your programs use for SGLang. Used when `OPENAI_PROXIES` is unset |
| `SGLANG_UPSTREAM_HOST` / `SGLANG_UPSTREAM_PORT` | `127.0.0.1` / `30000` | Where SGLang itself is listening |
| `PROXY_SOURCE` | detected | Force the label of the 11435 proxy (`ollama`, `vllm`, `sglang`, ...) |
| `ENGINES` | `ollama,vllm,sglang,llamacpp` | Older-build switch; used only when `OPENAI_PROXIES` is unset. Setting this, or any `VLLM_*` / `SGLANG_*` / `LLAMACPP_*` port, opens the automatic counting ports right away instead of waiting for the server |
| `INJECT_USAGE` | `1` | Ask streaming `/v1` chat, completions and other `/v1` requests (except embeddings and `/v1/responses`) to include usage |
| `LLAMA_LOG` | `~/pixpilot/run/server.log` | llama-server log to read. Empty turns the tail off and brings back the automatic 8081 proxy. A missing file is fine |
| `LLAMA_POLL_S` | `15` | Seconds between reads once caught up |
| `LLAMA_READ_MAX` | `2000000` | Max new log bytes read per pass, so a big file cannot stall the page |
| `LLAMA_MODEL` | `llama.cpp` | Model name until the log says which GGUF was loaded |
| `LLAMA_STARTS` | empty | Optional comma-separated epoch start times for exact `--llama-backfill` timestamps |
| `STATS_DB` | `./stats.db` | SQLite file |
| `MS_USD_TIERS` | `1000:100,10000:500,*:10000` | Dollar-saved milestones as `upper_bound:step` pairs: every `step` up to `upper_bound`, then the next pair. `*` means no upper bound and goes last |
| `MS_USD_FIXED` | empty | Extra one-off dollar milestones, comma-separated |
| `MS_TOK_TIERS` | `10000000000:1000000000,100000000000:5000000000,*:50000000000` | Token milestones in the same format (every 1B to 10B, every 5B to 100B, then every 50B) |
| `MS_TOK_FIXED` | `1000000,10000000,100000000,500000000` | One-off token milestones below the first tier |
| `MS_PREVIEW` | `6` | Badges shown in the collapsed "Extinction events survived" row |

Precedence, per setting: **saved from the Settings window** (gear button), then the `MS_*` env vars above, then the built-in defaults (`MS_DEFAULTS` near the top of `token_annihilator.py`). Reset to defaults in the window clears the saved values, so env vars (if any) apply again. Under the hood: `GET /api/settings` returns the effective, baseline and default settings plus where each came from, and `POST /api/settings` (same-origin only) takes `{"settings": {...}}` or `{"reset": true}`; invalid input is answered with `400` and per-field messages.

Milestones that fall off a changed schedule are hidden, never deleted, and a malformed `MS_*` value is ignored with a warning. When a new schedule includes milestones your totals already passed, they are filed as seen (dated when the total really crossed them), so changing settings never triggers a pop-up flood.

## Backfill Ollama history (optional)

```bash
python3 token_annihilator.py --backfill $(date +%s)
```

Imports past request lines from `journalctl -u ollama`. Those lines have no token counts, so they are stored but never mixed into totals.

## Tests

```bash
python3 test_annihilator.py
```

Stdlib only. Spins up fake Ollama, vLLM, SGLang and llama.cpp servers on local ports and checks the counting end to end, including `/v1/completions` traffic like `vllm bench serve`, a counting port that is already taken, and a search that never touches ports on the leave-alone list.

## Uninstall

```bash
systemctl --user disable --now token-annihilator
rm ~/.config/systemd/user/token-annihilator.service
rm -rf ~/token-annihilation-dashboard
```

Your inference servers were never touched, so there is nothing to undo there.

## License

MIT
