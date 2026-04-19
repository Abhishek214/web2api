# web2api

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Turn any LLM web interface into an OpenAI-compatible API — no API key needed.**

web2api uses [Playwright](https://playwright.dev/) to control a real Chrome browser session (one you're already logged in to) and exposes the result as a local REST API that speaks both the **Anthropic** (`/v1/messages`) and **OpenAI** (`/v1/chat/completions`) protocols.

**Two modes of operation:**
- **Claude Code mode** — Full tool support, message transformations, and Claude Code optimizations
- **Normal mode** — Simple pass-through for general API use with any OpenAI-compatible client

### Supported providers

| Provider | `TARGET_PROVIDER` value | Default URL |
|---|---|---|
| Google Gemini | `gemini` | https://gemini.google.com |
| ChatGPT | `chatgpt` | https://chatgpt.com |
| Claude.ai | `claude` | https://claude.ai |

> **Heads-up on Claude.ai:** Using Claude web as a backend for Claude Code creates a loop (Claude Code → web2api → Claude.ai → Claude). It works, but responses may differ from the API model. If you already have an Anthropic API key, Claude Code will use it directly and won't need this proxy for Claude.

---

## How it works

```
Your tool (Claude Code / curl / SDK)
         │  POST /v1/messages  or  /v1/chat/completions
         ▼
   FastAPI proxy  (main.py)
         │  worker.query(prompt)
         ▼
 PlaywrightWorker  (worker.py)
         │  fills textarea → clicks Send → polls response
         ▼
   Chrome browser  (already logged in to the LLM site)
```

- **One browser instance** is shared, protected by an asyncio lock — concurrent requests queue safely.
- **Session persistence** — connect to your already-authenticated Chrome, so no login automation needed.
- **Both API formats** — Anthropic `/v1/messages` (used by Claude Code) and OpenAI `/v1/chat/completions` (used by most other tools).

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or later |
| Google Chrome | Any recent version |
| OS | macOS, Linux, Windows |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Abhishek214/web2api.git
cd web2api
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the proxy

Copy the example env file and customize it:

```bash
cp env.example .env
```

Edit `.env`:

```env
# ── Provider selection ─────────────────────────────────────────────────────────
# Which LLM web interface to automate.
# Choices: gemini | chatgpt | claude
TARGET_PROVIDER=gemini

# ── API Mode ───────────────────────────────────────────────────────────────────
# Choose how the API processes requests:
#   claude  - Claude Code optimized mode (default). Handles tool calls, message
#             transformations, and deduplication for use with Claude Code.
#   normal  - Standard LLM API mode. No special processing, just passes
#             messages through to the web LLM and returns raw responses.
API_MODE=claude
```

**API Modes explained:**

| Mode | Use case | Description |
|------|----------|-------------|
| `claude` (default) | Claude Code, coding agents | Full Claude Code optimizations including tool call parsing, message transformations, deduplication, and retry logic |
| `normal` | General API use, other SDKs | Simple pass-through mode. Messages are formatted as conversations and raw responses are returned without special processing |

If you're using Claude Code, keep `API_MODE=claude`. For other use cases (OpenAI SDK, custom scripts, etc.), set `API_MODE=normal`.

### 4. Start Chrome with remote debugging enabled

This is the most important step. web2api connects to a Chrome instance **you are already logged in to**, so it inherits your session — no password automation required.

**macOS**
```bash
# Gemini / ChatGPT / Claude — same command, just navigate to the right site
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-web2api
```

**Linux**
```bash
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-web2api
```

**Windows (PowerShell)**
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:TEMP\chrome-web2api"
```

> Using `--user-data-dir` keeps this session separate from your normal Chrome profile so your regular browsing is unaffected.

### 5. Log in to the LLM web interface

With that Chrome window open, navigate to your provider and log in manually:

| Provider | URL |
|---|---|
| Gemini | https://gemini.google.com |
| ChatGPT | https://chatgpt.com |
| Claude.ai | https://claude.ai |

Make sure you can see the chat input box before proceeding.

### 6. Start the proxy server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
[worker] Connected to Chrome on port 9222
[worker] Ready — provider=gemini  url=https://gemini.google.com
INFO:     Application startup complete.
```

Check the health endpoint to confirm:
```bash
curl http://localhost:8000/health
# {"status":"ok","provider":"gemini","worker_ready":true}
```

---

## Using with Claude Code

Claude Code uses the Anthropic `/v1/messages` endpoint. Point it at the proxy:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=not-needed \
claude
```

Or add to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.):

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
```

Then just run `claude` normally.

---

## Using with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",   # required by SDK but ignored by the proxy
)

resp = client.chat.completions.create(
    model="gemini-web",     # or chatgpt-web / claude-web depending on TARGET_PROVIDER
    messages=[{"role": "user", "content": "Explain transformers in ML"}],
)
print(resp.choices[0].message.content)
```

---

## Using with curl

**Anthropic format:**
```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gemini-web",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**OpenAI format:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-web",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Anthropic-compatible endpoint (Claude Code, Anthropic SDK) |
| `/v1/chat/completions` | POST | OpenAI-compatible endpoint (most other tools) |
| `/v1/models` | GET | Returns available model list |
| `/health` | GET | Health check — confirms worker is ready |
| `/` | GET | Status page |

---

## Fixing selectors (when the UI changes)

LLM web interfaces update their HTML frequently. If the proxy stops working, selectors need updating.

Open the LLM site in your Chrome debug session, press **F12** to open DevTools, and use the element inspector:

| Element | How to find it |
|---|---|
| Prompt input | Click the text box → Inspect → copy the selector |
| Send button | Hover over it → Inspect |
| Response container | Inspect the last assistant message bubble |
| Stop button | Click send on a long prompt → quickly inspect the stop button |
| New chat button | Inspect the "New chat" or pencil icon |

Then update the relevant entry in the `PROVIDERS` dict in `worker.py`:

```python
PROVIDERS = {
    "gemini": {
        "selectors": {
            "input":       "rich-textarea .ql-editor",   # ← update here
            "send_button": "button.send-button",
            ...
        }
    },
    ...
}
```

### Current selectors

**Gemini**
```
input:       rich-textarea .ql-editor
send_button: button.send-button
response:    model-response:last-of-type .markdown
stop_button: button[aria-label='Stop response']
new_chat:    a[href='/'][aria-label='New chat']
```

**ChatGPT**
```
input:       #prompt-textarea
send_button: button[data-testid='send-button']
response:    div[data-message-author-role='assistant']:last-of-type .markdown
stop_button: button[data-testid='stop-button']
new_chat:    a[href='/']
```

**Claude.ai**
```
input:       div[contenteditable='true'].ProseMirror
send_button: button[aria-label='Send Message']
response:    .font-claude-message:last-of-type
stop_button: button[aria-label='Stop']
new_chat:    a[href='/new']
```

---

## Tips

- **Headed mode** — set `headless=False` in `PlaywrightWorker()` while debugging so you can watch the browser.
- **Rate limits** — if the web app throttles rapid requests, increase `RESPONSE_POLL_INTERVAL` or add `asyncio.sleep` between retries in `worker.py`.
- **Multiple providers at once** — run two instances of the proxy on different ports (e.g. 8000 for Gemini, 8001 for ChatGPT), each with its own `TARGET_PROVIDER` in their respective `.env` files.
- **Selector stability** — prefer `data-testid` attributes over class names; they tend to survive UI redesigns.
- **Streaming** — the proxy re-streams the complete response word-by-word to emulate token streaming. True token-level streaming is not possible via web scraping.

---

## Limitations

- **One request at a time** per proxy instance (requests are queued behind an asyncio lock).
- **No multimodal support** — images and file uploads in the request are ignored.
- **Selector fragility** — web UIs change; expect occasional selector updates.
- **Terms of Service** — automating web interfaces may violate the provider's ToS. Use responsibly and for personal/dev use only.

---

## Contributing

Contributions are welcome! Here are some areas where help is especially appreciated:

- **Updated selectors**: Web UIs change frequently — PRs with updated selectors are always welcome
- **New providers**: Support for Perplexity, Mistral, or other LLM web interfaces
- **Better parsing**: Improved response extraction for complex formats
- **Bug fixes**: Edge cases in tool call parsing or message handling

Please ensure your code:
- Passes existing functionality tests
- Follows the existing code style
- Includes appropriate comments for complex logic
- Updates documentation if behavior changes

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Disclaimer

This project automates web interfaces that may have Terms of Service restricting automated access. Use responsibly and at your own risk. This tool is intended for personal development and educational purposes only.