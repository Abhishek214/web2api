# LLM Web App → OpenAI API Proxy

Wraps any browser-accessible LLM web app with an OpenAI-compatible REST API,
using Playwright for automation and FastAPI to serve the endpoint.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure
cp .env.example .env
# Edit .env — set TARGET_URL and optionally LOGIN_EMAIL / LOGIN_PASSWORD

# 3. Configure selectors in worker.py → SELECTORS dict
#    (see "How to find selectors" below)

# 4. Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## How to Find Selectors

Open the LLM web app in Chrome/Firefox, open DevTools (F12):

| Element              | How to find                                      |
|----------------------|--------------------------------------------------|
| Prompt textarea      | Click the input box → inspect → copy selector   |
| Send button          | Hover the button → inspect → copy selector      |
| Response container   | Inspect the last assistant message bubble       |
| New chat button      | Inspect the "New chat" or reset button          |

Then update `SELECTORS` in `worker.py`.

---

## API Endpoints

### `POST /v1/chat/completions`

Drop-in replacement for OpenAI's endpoint.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "web-llm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Streaming** — set `"stream": true` for SSE token-by-token output.

### `GET /v1/models`

Returns a fake model list (compatible with OpenAI SDK model listing).

### `GET /health`

Returns `{ "status": "ok", "worker_ready": true }`.

---

## Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",        # required by SDK but ignored by proxy
)

resp = client.chat.completions.create(
    model="web-llm",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
print(resp.choices[0].message.content)
```

---

## Architecture

```
Client (OpenAI SDK / curl)
        │  POST /v1/chat/completions
        ▼
   FastAPI (main.py)
        │  worker.query(prompt)
        ▼
 PlaywrightWorker (worker.py)
        │  fills textarea → clicks send → polls response
        ▼
   LLM Web App (Chromium, headless)
```

- **One browser instance** is shared, protected by an asyncio lock → safe for concurrent API requests (they queue).
- **Session persistence** — after the first login, `auth_state.json` is saved so subsequent restarts skip the login flow.
- **Streaming** — the proxy re-streams the complete response word-by-word with a small delay, mimicking token streaming.

---

## Tips

- **Multiple workers** — instantiate a pool in `main.py` if you need parallel throughput.
- **Headed mode** — set `headless=False` in `PlaywrightWorker()` during debugging to watch the browser.
- **Selector stability** — prefer `data-testid` attributes over class names; they change less often.
- **Rate limits** — add a `asyncio.sleep` between requests if the web app throttles rapid submissions.
