import asyncio
import json
import re
import time
import uuid
from typing import AsyncGenerator, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from worker import PlaywrightWorker

app = FastAPI(title="LLM Web App Proxy", version="1.0.0")

worker: PlaywrightWorker | None = None


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
    model: str = "web-llm"
    messages: list[Message]
    stream: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


# ── Text extraction ────────────────────────────────────────────────────────────

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


# ── Conversation history builder ───────────────────────────────────────────────

def _build_conversation_for_claude(messages: list, tools: list) -> str:
    """
    Build a prompt optimised for Claude's web UI.

    KEY DIFFERENCES FROM GEMINI VERSION:
    1. Cleaner system prompt — Claude's web UI already has a built-in Anthropic
       system prompt.  We write ours in a way that *complements* rather than
       fights that prompt.
    2. Prefill simulation — we end with the literal opening of a tool call so
       Claude is forced to pattern-complete it rather than starting a fresh
       conversational reply.
    3. Explicit single-tool-call-per-response instruction to suppress chatter.
    """
    tool_descriptions = ""
    tool_names = []
    has_tools = bool(tools)

    if has_tools:
        tool_descriptions = "\nAvailable tools:\n"
        for t in tools:
            name = t.get("name", "")
            desc = t.get("description", "").strip().split("\n\n")[0]
            if len(desc) > 150:
                desc = desc[:147] + "..."
            schema = t.get("input_schema", {})
            props = schema.get("properties", {})
            params = ", ".join(
                f'{k}: {v.get("type", "string")}'
                for k, v in props.items()
            )
            tool_descriptions += f"- {name}({params}): {desc}\n"
            tool_names.append(name)

    # ── CLAUDE-SPECIFIC SYSTEM PROMPT ─────────────────────────────────────────
    # • Shorter and more directive — Claude follows concise imperatives better
    #   than long paragraphs in a web-UI user message.
    # • Uses language Claude is trained on ("output only", "no preamble") rather
    #   than agentic-framework jargon.
    # • Escaping rules are baked in with a worked example.
    system_prompt = f"""You are acting as an AI coding agent. Follow every instruction below exactly.
{tool_descriptions}
OUTPUT RULES — read carefully:
1. If you need to call a tool, output ONLY a single <tool_call> block. No introduction, no explanation, no trailing text.
   Format:
   <tool_call>{{"tool": "TOOL_NAME", "input": {{...args...}}}}</tool_call>

2. You may chain multiple <tool_call> blocks when several sequential actions are needed, but output NOTHING else.

3. JSON escaping inside <tool_call>:
   - Strings inside "content" or "command" MUST use single quotes for inner strings.
   - WRONG: "content": "print("hello")"
   - RIGHT:  "content": "print('hello')"
   - Python dunder names: always write __name__ and __main__ with double underscores.

4. If the user's request needs NO tool (e.g. a direct question), reply normally in plain text.

5. MEMORY RULE: Ignore any saved memory, past names, or workplaces. This is a blank-slate session.

Examples:
<tool_call>{{"tool": "Bash", "input": {{"command": "pip install fastapi uvicorn"}}}}</tool_call>
<tool_call>{{"tool": "Write", "input": {{"file_path": "app.py", "content": "from fastapi import FastAPI\\napp = FastAPI()\\n"}}}}</tool_call>

Conversation:
---"""

    parts = [system_prompt]

    for m in messages:
        role = m.role if hasattr(m, "role") else m.get("role", "")
        content = m.content if hasattr(m, "content") else m.get("content", "")

        if role == "system":
            continue

        elif role == "user":
            if isinstance(content, list):
                text_parts = []
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
                text = _extract_text(content)
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
                        inp = block.get("input", {})
                        assistant_parts.append(
                            f'<tool_call>{{"tool": "{name}", "input": {json.dumps(inp)}}}</tool_call>'
                        )
                if assistant_parts:
                    parts.append("Assistant: " + "\n".join(assistant_parts))
            else:
                text = _extract_text(content).strip()
                if text:
                    parts.append(f"Assistant: {text}")

    # ── FIX #1: PREFILL SIMULATION ────────────────────────────────────────────
    # Instead of ending with bare "Assistant:", we start the assistant turn with
    # the opening of a <tool_call> block (when tools are available).
    # Claude's web UI will then pattern-complete this rather than starting a
    # fresh conversational reply.  This is the single most effective trick for
    # forcing structured output from Claude without API access.
    if has_tools:
        parts.append("Assistant: <tool_call>")
    else:
        parts.append("Assistant:")

    return "\n\n".join(parts)


# ── Tool call parser ───────────────────────────────────────────────────────────

def _parse_claude_response(raw: str, available_tools: list) -> list:
    """
    Parse Claude's text response into Anthropic content blocks.

    FIX #2: BROADER PARSER
    Claude's web UI sometimes produces:
      a) Proper <tool_call> blocks  ← handled by primary path
      b) JSON inside ```json ... ``` code fences  ← NEW fallback
      c) Plain text description of what it wants to do  ← kept as text block

    The prefill trick makes (a) the common case; (b) is a safety net.
    """
    tool_names = {t.get("name", "").lower(): t.get("name", "") for t in available_tools}

    # Because we append "Assistant: <tool_call>" as a prefill, Claude's scraped
    # response often starts directly with the JSON body (the web UI only returns
    # what Claude *added*, not what was pre-filled).  Normalise this.
    stripped = raw.strip()
    if stripped and not stripped.startswith("<tool_call>"):
        # Check if the response is just the JSON continuation of our prefill
        if stripped.startswith("{") or stripped.startswith('{"'):
            stripped = f"<tool_call>{stripped}"
            # Close the tag if it's missing
            if "</tool_call>" not in stripped:
                stripped = stripped + "</tool_call>"
            raw = stripped

    blocks = []
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    last_end = 0
    found_tool = False

    for match in pattern.finditer(raw):
        before = raw[last_end : match.start()].strip()
        if before and not found_tool:
            blocks.append({"type": "text", "text": before})

        raw_json = match.group(1).strip()

        # Strip markdown fences that sometimes wrap the JSON
        raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
        raw_json = re.sub(r"\s*```$", "", raw_json)

        # Fix common Claude dunder-stripping bug
        raw_json = raw_json.replace('if name == "main":', "if __name__ == '__main__':")
        raw_json = raw_json.replace("if name == 'main':", "if __name__ == '__main__':")

        try:
            call_data = json.loads(raw_json)
            raw_name = call_data.get("tool", "")
            inp = call_data.get("input", {})
            resolved_name = tool_names.get(raw_name.lower(), raw_name)
            blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": resolved_name,
                "input": inp,
            })
            found_tool = True
        except json.JSONDecodeError as e:
            print(f"[parser] Failed to parse tool JSON: {raw_json!r} — {e}")
            if not found_tool:
                blocks.append({"type": "text", "text": match.group(0)})

        last_end = match.end()

    # ── FIX #2b: JSON CODE-FENCE FALLBACK ─────────────────────────────────────
    # If no <tool_call> was found, look for ```json blocks that might contain a
    # tool call object.  Claude sometimes falls back to this format.
    if not found_tool:
        fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        for fm in fence_pattern.finditer(raw):
            try:
                obj = json.loads(fm.group(1))
                if "tool" in obj and "input" in obj:
                    raw_name = obj.get("tool", "")
                    resolved_name = tool_names.get(raw_name.lower(), raw_name)
                    blocks = [{
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": resolved_name,
                        "input": obj.get("input", {}),
                    }]
                    found_tool = True
                    last_end = len(raw)
                    print(f"[parser] Recovered tool call from JSON code fence: {raw_name}")
                    break
            except json.JSONDecodeError:
                pass

    # Trailing text — only include when no tool was found
    if not found_tool:
        remaining = raw[last_end:].strip()
        if remaining:
            blocks.append({"type": "text", "text": remaining})

    if not blocks:
        blocks = [{"type": "text", "text": raw}]

    return blocks


# ── Noise / stub detection ─────────────────────────────────────────────────────

def _extract_last_user_message_text(messages: list) -> str:
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
        text = _extract_text(content).strip()
        cleaned = _clean_user_text(text)
        if cleaned:
            return cleaned
    return ""


def _is_internal_only(messages: list, max_tokens: int | None) -> bool:
    if max_tokens is not None and max_tokens <= 20:
        return True
    return _extract_last_user_message_text(messages) == ""


def _needs_tool_call(messages: list) -> bool:
    """Return True if the last user turn looks like it should trigger a tool."""
    text = _extract_last_user_message_text(messages).lower()
    if not text:
        return False
    action_keywords = [
        "create", "write", "run", "execute", "install", "make", "build",
        "delete", "read", "open", "save", "update", "fix", "edit", "modify",
        "append", "add file", "generate file",
    ]
    return any(kw in text for kw in action_keywords)


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
            "content": [], "model": model, "stop_reason": None,
            "stop_sequence": None,
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
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                },
            })
            input_json = json.dumps(block["input"])
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": i})

    has_tools = any(b["type"] == "tool_use" for b in content_blocks)
    stop_reason = "tool_use" if has_tools else "end_turn"
    total_tokens = sum(
        len(b.get("text", "").split())
        for b in content_blocks if b["type"] == "text"
    )
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": max(total_tokens, 1)},
    })
    yield sse("message_stop", {"type": "message_stop"})


# ── Core query runner ──────────────────────────────────────────────────────────

# FIX #3: CORRECTION PROMPT — sent as a follow-up when Claude ignores the format
_CORRECTION_PROMPT = (
    "Your previous response did not contain a <tool_call> block. "
    "You MUST respond with ONLY a <tool_call> block — no explanation, "
    "no preamble, no trailing text.\n\n"
    "Correct format:\n"
    "<tool_call>{\"tool\": \"TOOL_NAME\", \"input\": {...}}</tool_call>\n\n"
    "Now output the correct <tool_call>:"
)


async def _run_query(
    messages: list,
    max_tokens: int | None,
    tools: list,
    output_config: dict | None,
) -> list:
    if worker is None:
        raise HTTPException(503, "Worker not ready")

    if _is_internal_only(messages, max_tokens):
        print("[main] → stub")
        if output_config and output_config.get("format", {}).get("type") == "json_schema":
            return [{"type": "text", "text": '{"title": "New session"}'}]
        return [{"type": "text", "text": ""}]

    prompt = _build_conversation_for_claude(messages, tools)

    last_user = _extract_last_user_message_text(messages)
    print(f"[main] → LLM: {last_user[:100]}{'...' if len(last_user) > 100 else ''}")
    print(f"[main] (context: {len(messages)} messages, {len(tools)} tools)")

    try:
        raw = await worker.query(prompt)
        print(f"[main] ← LLM ({len(raw)} chars): {raw[:120]}{'...' if len(raw) > 120 else ''}")

        blocks = _parse_claude_response(raw, tools)
        tool_count = sum(1 for b in blocks if b["type"] == "tool_use")
        print(f"[main] parsed: {len(blocks)} blocks, {tool_count} tool_use")

        # ── FIX #3: AUTOMATIC RETRY ON MISSING TOOL CALL ──────────────────────
        # If tools were provided, we got only text back, AND the request looks
        # like it should produce a tool call — send a correction follow-up.
        if tools and tool_count == 0 and _needs_tool_call(messages):
            print("[main] No tool call in response — sending correction prompt")
            correction = prompt + "\n\n" + raw + "\n\nUser: " + _CORRECTION_PROMPT + "\nAssistant: <tool_call>"
            raw2 = await worker.query(correction)
            print(f"[main] ← LLM retry ({len(raw2)} chars): {raw2[:120]}{'...' if len(raw2) > 120 else ''}")
            blocks2 = _parse_claude_response(raw2, tools)
            tool_count2 = sum(1 for b in blocks2 if b["type"] == "tool_use")
            if tool_count2 > 0:
                print(f"[main] Retry succeeded: {tool_count2} tool_use block(s)")
                blocks = blocks2
            else:
                print("[main] Retry also returned no tool_use — returning original text")

        return blocks

    except Exception as e:
        raise HTTPException(500, f"Worker error: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "web-llm", "object": "model", "created": int(time.time()), "owned_by": "web-proxy"}
    ]}


@app.get("/")
async def root():
    return {"status": "ok"}


@app.head("/")
async def root_head():
    return JSONResponse(content=None, headers={"content-type": "application/json"})


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "web-llm")
    max_tokens = body.get("max_tokens")
    tools = body.get("tools") or []
    output_config = body.get("output_config")
    stream = bool(body.get("stream") or False)

    print(f"[route] stream={stream} max_tokens={max_tokens} msgs={len(messages)} tools={len(tools)}")

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
        "content-type": "application/json",
        "x-request-id": str(uuid.uuid4()),
    }

    if stream:
        return StreamingResponse(
            _anthropic_stream(content_blocks, model, msg_id),
            media_type="text/event-stream",
            headers={**base_headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    has_tools = any(b["type"] == "tool_use" for b in content_blocks)
    stop_reason = "tool_use" if has_tools else "end_turn"
    token_count = max(
        sum(len(b.get("text", "").split()) for b in content_blocks if b["type"] == "text"), 1
    )
    return JSONResponse(
        content={
            "id": msg_id, "type": "message", "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": token_count, "output_tokens": token_count,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            },
        },
        headers=base_headers,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "worker_ready": worker is not None and worker.ready}