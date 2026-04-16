import asyncio
import json
import os
import re
import time
import uuid
from typing import AsyncGenerator, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from worker import PlaywrightWorker, TARGET_PROVIDER
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="web2api — LLM Web Proxy", version="2.0.0")

worker: PlaywrightWorker | None = None

# Provider → friendly display name for the /v1/models list
_PROVIDER_MODEL_ID = {
    "gemini":  "gemini-web",
    "chatgpt": "chatgpt-web",
    "claude":  "claude-web",
}
MODEL_ID = _PROVIDER_MODEL_ID.get(TARGET_PROVIDER, "web-llm")


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global worker
    worker = PlaywrightWorker()
    await worker.start()


@app.on_event("shutdown")
async def shutdown():
    if worker:
        await worker.stop()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n[REQUEST] {request.method} {request.url}")
    response = await call_next(request)
    print(f"[RESPONSE] {response.status_code}")
    return response


# ── Schemas ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    stream: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


# ── Text helpers ───────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _clean_user_text(text: str) -> str:
    _TAGS = [
        "system-reminder", "local-command-caveat", "command-name",
        "command-message", "command-args", "local-command-stdout",
        "local-command-stderr", "function-calls", "function-results",
        "context-window-usage", "environment_details", "tool_response",
    ]
    for tag in _TAGS:
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
        text = re.sub(rf"<{tag}\s*/>", "", text)
    return text.strip()


_SYSTEM_NOISE = [
    "x-anthropic-billing-header", "You are Claude Code",
    "You are an interactive agent", "Generate a concise, sentence-case title",
    "Return JSON with a single", "system-reminder", "Claude Code harness",
    "settings.json", "keybindings.json", "Automated behaviors",
    "IMPORTANT: Assist with authorized security", "interleaved-thinking",
    "cc_version=", "cc_entrypoint=", "claude_code_version", "cwd=",
    "<environment_details>", "antml:function", "tool_use_id",
]


def _is_noise(text: str) -> bool:
    return any(p in text for p in _SYSTEM_NOISE)


# ── Conversation builder ───────────────────────────────────────────────────────

def _build_conversation(messages: list, tools: list) -> str:
    """Convert Anthropic-style message list into a plain-text prompt."""

    # Keep tool listing ultra-compact: name + param names only, no descriptions.
    # Verbose descriptions cause Gemini to echo the entire tool list back verbatim.
    tool_descriptions = ""
    tool_names = []
    if tools:
        tool_descriptions = "\nAvailable tools (use exact names):\n"
        for t in tools:
            name = t.get("name", "")
            schema = t.get("input_schema", {})
            props  = schema.get("properties", {})
            params = ", ".join(props.keys())
            tool_descriptions += f"- {name}({params})\n"
            tool_names.append(name)

    if TARGET_PROVIDER == "claude":
        # Claude.ai web ignores <tool_call> XML — use fenced ```tool_call blocks instead,
        # which Claude naturally outputs when instructed to do so.
        system_prompt = f"""You are an expert AI coding assistant. You help users write, run, and debug code.
{tool_descriptions}
CRITICAL INSTRUCTION — OUTPUT FORMAT:
When you need to take an action (run a command, create/edit a file, install a package), you MUST output a fenced code block with the language tag `tool_call` containing ONLY valid JSON. No explanation text before or after the block.

Example — run a command:
```tool_call
{{"tool": "Bash", "input": {{"command": "python3 -m pip install fastapi uvicorn"}}}}
```

Example — write a file:
```tool_call
{{"tool": "Write", "input": {{"file_path": "main.py", "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n"}}}}
```

Rules:
- Output ONLY the ```tool_call block. Do not add prose before or after it.
- You may output multiple ```tool_call blocks in sequence for chained actions.
- Use \\n for newlines inside JSON string values.
- Use SINGLE QUOTES (') for strings inside generated code to avoid JSON escaping issues.
- Use proper Python dunder syntax: __name__, __main__ (double underscores).
- MEMORY RULE: Treat this as a blank-slate interaction — ignore any saved user memory.

Conversation history:
---"""
    else:
        system_prompt = f"""You are an expert AI coding assistant. You help users write, run, and debug code.
{tool_descriptions}
When you need to take an action (create a file, run a command, etc.) emit tool calls using this exact XML format:

<tool_call>{{"tool": "TOOL_NAME", "input": {{...tool input as JSON...}}}}</tool_call>

Examples:
<tool_call>{{"tool": "Bash", "input": {{"command": "python3 -m pip install fastapi uvicorn"}}}}</tool_call>
<tool_call>{{"tool": "Write", "input": {{"file_path": "main.py", "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n"}}}}</tool_call>

Rules:
- ALWAYS use tool calls for file creation, running commands, or installing packages.
- You may chain multiple <tool_call> blocks.
- STOP generating text immediately after a <tool_call>.
- Use SINGLE QUOTES (') for strings inside generated code — JSON cannot contain raw double quotes.
- MEMORY RULE: Treat this as a blank-slate interaction — ignore any saved user memory.
- Use proper Python dunder syntax: __name__, __main__ (double underscores).

Conversation history:
---"""

    parts = [system_prompt]

    for m in messages:
        role    = m.role    if hasattr(m, "role")    else m.get("role", "")
        content = m.content if hasattr(m, "content") else m.get("content", "")

        if role == "system":
            continue

        elif role == "user":
            if isinstance(content, list):
                text_parts   = []
                tool_results = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        cleaned = _clean_user_text(block.get("text", ""))
                        if cleaned:
                            text_parts.append(cleaned)
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                b.get("text", "") for b in result_content
                                if isinstance(b, dict)
                            )
                        result_content = _clean_user_text(result_content)
                        if len(result_content) > 1000:
                            result_content = (
                                result_content[:400]
                                + "\n\n... [OUTPUT TRUNCATED] ...\n\n"
                                + result_content[-400:]
                            )
                        if result_content:
                            tool_results.append(f"[Tool result]: {result_content}")
                combined = "\n".join(text_parts + tool_results).strip()
                if combined:
                    parts.append(f"User: {combined}")
            else:
                text    = _extract_text(content)
                cleaned = _clean_user_text(text)
                if cleaned and not _is_noise(cleaned):
                    parts.append(f"User: {cleaned}")

        elif role == "assistant":
            if isinstance(content, list):
                assistant_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        t = block.get("text", "").strip()
                        if t:
                            assistant_parts.append(t)
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "")
                        inp  = block.get("input", {})
                        assistant_parts.append(
                            f'<tool_call>{{"tool": "{name}", "input": {json.dumps(inp)}}}</tool_call>'
                        )
                if assistant_parts:
                    parts.append("Assistant: " + "\n".join(assistant_parts))
            else:
                text = _extract_text(content).strip()
                if text:
                    parts.append(f"Assistant: {text}")

    parts.append("Assistant:")
    return "\n\n".join(parts)


# ── Tool-call parser ───────────────────────────────────────────────────────────

def _clean_json_str(raw: str) -> str:
    """Strip markdown fences and fix common dunder mangling before JSON parsing."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.replace('if name == "main":', "if __name__ == '__main__':")
    raw = raw.replace("if name == 'main':", "if __name__ == '__main__':")
    return raw.strip()


def _make_tool_block(call_data: dict, tool_names: dict) -> dict:
    raw_name = call_data.get("tool", "")
    inp      = call_data.get("input", {})
    resolved = tool_names.get(raw_name.lower(), raw_name)
    return {
        "type":  "tool_use",
        "id":    f"toolu_{uuid.uuid4().hex[:24]}",
        "name":  resolved,
        "input": inp,
    }


def _parse_response(raw_text: str, available_tools: list) -> list:
    """
    Parse provider response text into Anthropic content blocks.

    Handles three formats that different providers emit:

    1. Wrapped XML (what we ask for, Gemini usually complies):
         <tool_call>{"tool": "Bash", "input": {"command": "..."}}</tool_call>

    2. Bare JSON object on its own (ChatGPT frequently does this):
         {"tool": "Write", "input": {"file_path": "main.py", "content": "..."}}

    3. JSON inside a markdown code block (ChatGPT alternative):
         ```json
         {"tool": "Bash", "input": {"command": "..."}}
         ```
    """
    tool_names = {t.get("name", "").lower(): t.get("name", "") for t in available_tools}
    blocks: list = []

    # ── Pass 1: look for explicit <tool_call> tags ─────────────────────────────
    xml_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    last_end    = 0
    found_tool  = False

    for match in xml_pattern.finditer(raw_text):
        before = raw_text[last_end : match.start()].strip()
        if before and not found_tool:
            blocks.append({"type": "text", "text": before})

        try:
            call_data = json.loads(_clean_json_str(match.group(1)))
            blocks.append(_make_tool_block(call_data, tool_names))
            found_tool = True
        except json.JSONDecodeError as e:
            print(f"[main] XML tool_call JSON parse failed: {e}")
            if not found_tool:
                blocks.append({"type": "text", "text": match.group(0)})

        last_end = match.end()

    if found_tool:
        # Don't append trailing text after a tool call (prevents hallucinated output)
        return blocks

    # ── Pass 2: look for bare JSON objects with a "tool" key ──────────────────
    # ChatGPT often returns {"tool": "...", "input": {...}} without XML tags.
    # Regex can't handle nested braces; use a brace-matching scanner instead.

    def _extract_json_objects(text: str):
        """
        Yield (start, end, json_string) for every top-level {...} object in text.
        Correctly handles nested objects, strings with escaped chars, etc.
        """
        i = 0
        n = len(text)
        while i < n:
            if text[i] != '{':
                i += 1
                continue
            depth     = 0
            in_str    = False
            escaped   = False
            start     = i
            for j in range(i, n):
                c = text[j]
                if escaped:
                    escaped = False
                    continue
                if c == '\\' and in_str:
                    escaped = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        yield start, j + 1, text[start : j + 1]
                        i = j + 1
                        break
            else:
                i += 1

    found_in_pass2 = False
    for start, end, candidate in _extract_json_objects(raw_text):
        candidate = _clean_json_str(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict) or "tool" not in data:
            continue

        if not found_in_pass2:
            prefix = re.sub(r"```(?:json)?\s*$", "", raw_text[:start]).strip()
            if prefix:
                blocks.append({"type": "text", "text": prefix})

        blocks.append(_make_tool_block(data, tool_names))
        found_in_pass2 = True
        print(f"[main] Bare JSON tool call detected: tool={data.get('tool')}")

    if found_in_pass2:
        return blocks

    # ── Pass 3: fenced ```tool_call blocks (Claude.ai native format) ─────────────
    # Claude.ai web reliably outputs ```tool_call\n{...}\n``` when instructed.
    tc_pattern = re.compile(r"```tool_call\s*\n(.*?)\n?```", re.DOTALL)
    found_in_pass3 = False

    for match in tc_pattern.finditer(raw_text):
        raw_json = _clean_json_str(match.group(1))
        try:
            call_data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"[main] fenced tool_call JSON parse failed: {e}")
            continue

        if not found_in_pass3:
            prefix = raw_text[: match.start()].strip()
            if prefix:
                blocks.append({"type": "text", "text": prefix})

        if isinstance(call_data, dict) and "tool" in call_data:
            blocks.append(_make_tool_block(call_data, tool_names))
            found_in_pass3 = True
            print(f"[main] Fenced tool_call detected: tool={call_data.get('tool')}")

    if found_in_pass3:
        return blocks

    # ── Pass 4: Claude natural-language fallback ──────────────────────────────
    # When Claude.ai ignores the format entirely it outputs prose + code blocks.
    # Extract bash blocks -> Bash tool; infer Write tool from filename hints.
    if TARGET_PROVIDER == "claude":
        nat_pattern = re.compile(
            r"(?:(?:^|\n)[^\n]*?(?:create|write|save|update|file)[^\n]*?"
            r"[`'\"]([\w./\-]+\.\w+)[`'\"][^\n]*\n)?"
            r"```([\w]*)\n(.*?)\n?```",
            re.DOTALL | re.IGNORECASE,
        )
        found_in_pass4 = False
        text_before_first = None

        for match in nat_pattern.finditer(raw_text):
            filename = (match.group(1) or "").strip()
            lang     = (match.group(2) or "").strip().lower()
            code     = match.group(3)

            if not code.strip():
                continue

            if not found_in_pass4:
                text_before_first = raw_text[: match.start()].strip()

            if lang in ("bash", "sh", "shell", "zsh"):
                for line in code.splitlines():
                    cmd = line.strip()
                    if cmd and not cmd.startswith("#"):
                        blocks.append(_make_tool_block(
                            {"tool": "Bash", "input": {"command": cmd}},
                            tool_names,
                        ))
                        print(f"[main] Pass4 bash: {cmd[:60]}")
                        found_in_pass4 = True
            elif filename:
                write_name = tool_names.get("write", tool_names.get("str_replace_editor", "Write"))
                blocks.append(_make_tool_block(
                    {"tool": write_name, "input": {"file_path": filename, "content": code}},
                    tool_names,
                ))
                print(f"[main] Pass4 write: {filename}")
                found_in_pass4 = True

        if found_in_pass4:
            if text_before_first:
                blocks.insert(0, {"type": "text", "text": text_before_first})
            return blocks

    # ── Pass 5: plain text response ────────────────────────────────────────────
    remaining = raw_text.strip()
    if remaining:
        blocks.append({"type": "text", "text": remaining})

    if not blocks:
        blocks = [{"type": "text", "text": raw_text}]

    return blocks


# ── Noise / stub detection ─────────────────────────────────────────────────────

def _extract_last_user_text(messages: list) -> str:
    for m in reversed(messages):
        role = m.role if hasattr(m, "role") else m.get("role", "")
        if role != "user":
            continue
        content = m.content if hasattr(m, "content") else m.get("content", "")
        if isinstance(content, list) and content and all(
            isinstance(b, dict) and b.get("type") in ("tool_result", "tool_use")
            for b in content
        ):
            continue
        text    = _extract_text(content).strip()
        cleaned = _clean_user_text(text)
        if cleaned:
            return cleaned
    return ""


def _is_internal_only(messages: list, max_tokens: int | None) -> bool:
    if max_tokens is not None and max_tokens <= 20:
        return True
    return _extract_last_user_text(messages) == ""


# ── Streaming helpers ──────────────────────────────────────────────────────────

async def _anthropic_stream(
    content_blocks: list, model: str, msg_id: str
) -> AsyncGenerator[str, None]:
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    for i, block in enumerate(content_blocks):
        if block["type"] == "text":
            yield sse("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""},
            })
            yield sse("ping", {"type": "ping"})
            words = block["text"].split(" ")
            for j, word in enumerate(words):
                chunk = word if j == 0 else f" {word}"
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": i,
                    "delta": {"type": "text_delta", "text": chunk},
                })
                await asyncio.sleep(0.005)
            yield sse("content_block_stop", {"type": "content_block_stop", "index": i})

        elif block["type"] == "tool_use":
            yield sse("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {
                    "type": "tool_use",
                    "id":   block["id"],
                    "name": block["name"],
                    "input": {},
                },
            })
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                },
            })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": i})

    has_tools   = any(b["type"] == "tool_use" for b in content_blocks)
    stop_reason = "tool_use" if has_tools else "end_turn"
    total_tokens = sum(
        len(b.get("text", "").split()) for b in content_blocks if b["type"] == "text"
    )

    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": max(total_tokens, 1)},
    })
    yield sse("message_stop", {"type": "message_stop"})


# ── Core runner ────────────────────────────────────────────────────────────────

async def _run_query(
    messages: list,
    max_tokens: int | None,
    tools: list,
    output_config: dict | None,
) -> list:
    if worker is None:
        raise HTTPException(503, "Worker not ready")

    if _is_internal_only(messages, max_tokens):
        print("[main] → stub (internal/no-op request)")
        if output_config and output_config.get("format", {}).get("type") == "json_schema":
            return [{"type": "text", "text": '{"title": "New session"}'}]
        return [{"type": "text", "text": ""}]

    prompt   = _build_conversation(messages, tools)
    last_txt = _extract_last_user_text(messages)
    print(f"[main] → {TARGET_PROVIDER}: {last_txt[:100]}{'...' if len(last_txt)>100 else ''}")
    print(f"[main]   context: {len(messages)} msgs, {len(tools)} tools")

    try:
        raw    = await worker.query(prompt)
        print(f"[main] ← {TARGET_PROVIDER} ({len(raw)} chars): {raw[:80]}{'...' if len(raw)>80 else ''}")
        blocks = _parse_response(raw, tools)
        n_tool = sum(1 for b in blocks if b["type"] == "tool_use")
        print(f"[main]   parsed: {len(blocks)} blocks, {n_tool} tool_use")
        return blocks
    except Exception as e:
        raise HTTPException(500, f"Worker error: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id":       MODEL_ID,
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": f"web-proxy/{TARGET_PROVIDER}",
            }
        ],
    }


@app.get("/")
async def root():
    return {"status": "ok", "provider": TARGET_PROVIDER}


@app.head("/")
async def root_head():
    return JSONResponse(content=None, headers={"content-type": "application/json"})


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body    = await request.json()
    messages       = body.get("messages", [])
    model          = body.get("model", MODEL_ID)
    max_tokens     = body.get("max_tokens")
    tools          = body.get("tools") or []
    output_config  = body.get("output_config")
    stream         = bool(body.get("stream") or False)

    print(f"[route /v1/messages] stream={stream} max_tokens={max_tokens} "
          f"msgs={len(messages)} tools={len(tools)}")

    system = body.get("system", "")
    if isinstance(system, list):
        system = " ".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)

    content_blocks = await _run_query(messages, max_tokens, tools, output_config)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    base_headers = {
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
        "x-request-id":      str(uuid.uuid4()),
    }

    if stream:
        return StreamingResponse(
            _anthropic_stream(content_blocks, model, msg_id),
            media_type="text/event-stream",
            headers={**base_headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    has_tools   = any(b["type"] == "tool_use" for b in content_blocks)
    stop_reason = "tool_use" if has_tools else "end_turn"
    n_tokens    = max(
        sum(len(b.get("text","").split()) for b in content_blocks if b["type"] == "text"), 1
    )

    return JSONResponse(
        content={
            "id":            msg_id,
            "type":          "message",
            "role":          "assistant",
            "content":       content_blocks,
            "model":         model,
            "stop_reason":   stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens":               n_tokens,
                "output_tokens":              n_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens":     0,
            },
        },
        headers=base_headers,
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible endpoint (for non-Claude Code clients)."""
    body    = await request.json()
    messages   = body.get("messages", [])
    model      = body.get("model", MODEL_ID)
    max_tokens = body.get("max_tokens")
    stream     = bool(body.get("stream") or False)

    print(f"[route /v1/chat/completions] stream={stream} msgs={len(messages)}")

    content_blocks = await _run_query(messages, max_tokens, [], None)
    full_text      = " ".join(
        b.get("text", "") for b in content_blocks if b["type"] == "text"
    )
    msg_id  = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if stream:
        async def openai_stream():
            words = full_text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else f" {word}"
                data  = {
                    "id":      msg_id,
                    "object":  "chat.completion.chunk",
                    "created": created,
                    "model":   model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.005)
            done = {
                "id":      msg_id,
                "object":  "chat.completion.chunk",
                "created": created,
                "model":   model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_stream(), media_type="text/event-stream")

    n_tokens = max(len(full_text.split()), 1)
    return JSONResponse(content={
        "id":      msg_id,
        "object":  "chat.completion",
        "created": created,
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     n_tokens,
            "completion_tokens": n_tokens,
            "total_tokens":      n_tokens * 2,
        },
    })


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "provider":     TARGET_PROVIDER,
        "worker_ready": worker is not None and worker.ready,
    }