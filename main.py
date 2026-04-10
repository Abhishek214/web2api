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

# ── Conversation history builder ───────────────────────────────────────────────

def _build_conversation_for_gemini(messages: list, tools: list) -> str:
    # 1. COMPRESS TOOLS: Drop the massive manuals and only send the schema
    tool_descriptions = ""
    tool_names = []
    if tools:
        tool_descriptions = "\nAvailable tools:\n"
        for t in tools:
            name = t.get("name", "")
            
            # Take only the first sentence/paragraph of the description
            desc = t.get("description", "").strip().split("\n\n")[0]
            if len(desc) > 150:
                desc = desc[:147] + "..."
                
            schema = t.get("input_schema", {})
            props = schema.get("properties", {})
            
            # Strip out parameter descriptions, just send the type
            params = ", ".join(
                f'{k}: {v.get("type", "string")}' 
                for k, v in props.items()
            )
            tool_descriptions += f"- {name}({params}): {desc}\n"
            tool_names.append(name)

    system_prompt = f"""You are an expert AI coding assistant (acting as Claude). You help users write, run, and debug code.

{tool_descriptions}
When you need to take an action (create a file, run a command, etc.) you MUST emit tool calls using this exact XML format — do NOT just describe what to do:

<tool_call>{{"tool": "TOOL_NAME", "input": {{...tool input as JSON...}}}}</tool_call>

Examples:
<tool_call>{{"tool": "Bash", "input": {{"command": "python3 -m pip install fastapi uvicorn"}}}}</tool_call>
<tool_call>{{"tool": "Write", "input": {{"file_path": "main.py", "content": "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get('/')\\ndef read_root():\\n    return {{'Hello': 'World'}}\\n"}}}}</tool_call>

Rules:
- ALWAYS use tool calls for file creation, running commands, or installing packages.
- You may chain multiple tool_call blocks in one response.
- STOP generating text immediately after a <tool_call>.
- CRITICAL JSON ESCAPING: Your tool calls are parsed by a strict JSON parser. You CANNOT use raw double quotes inside the "content" or "command" strings. You MUST use SINGLE QUOTES (') for all strings inside your generated code.
  - WRONG: "content": "print("hello")"
  - RIGHT: "content": "print('hello')"
- MEMORY RULE: Disregard all saved user memory, past contexts, names, or workplaces. Treat this as a completely isolated, blank-slate interaction.
- PYTHON DUNDER VARIABLES: You MUST use double underscores for `__name__` and `__main__`. Do NOT strip them to `name` or `main`.

Conversation history:
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
                        
                        # 2. CLEAN NOISE: Apply tag cleaner to tool results too
                        result_content = _clean_user_text(result_content)
                        
                        # 3. TRUNCATE TERMINAL LOGS: Keep head and tail to save thousands of tokens
                        if len(result_content) > 1000:
                            result_content = result_content[:400] + "\n\n... [OUTPUT TRUNCATED FOR LENGTH] ...\n\n" + result_content[-400:]
                            
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
                    parts.append(f"Assistant: " + "\n".join(assistant_parts))
            else:
                text = _extract_text(content).strip()
                if text:
                    parts.append(f"Assistant: {text}")

    parts.append("Assistant:")
    return "\n\n".join(parts)


# ── Tool call parser ───────────────────────────────────────────────────────────
def _parse_gemini_response(gemini_text: str, available_tools: list) -> list:
    """
    Parse Gemini's text response into a list of Anthropic content blocks.
    """
    tool_names = {t.get("name", "").lower(): t.get("name", "") for t in available_tools}

    blocks = []
    pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    
    last_end = 0
    found_tool = False

    for match in pattern.finditer(gemini_text):
        # Text before this tool call
        before = gemini_text[last_end:match.start()].strip()
        # Only add text if we haven't hit a tool call yet
        if before and not found_tool:
            blocks.append({"type": "text", "text": before})

        raw_json = match.group(1).strip()
        
        # 1. FIX THE MARKDOWN TRAP: Strip ```json and ``` before parsing
        raw_json = re.sub(r'\s*```$', '', raw_json)
        raw_json = raw_json.replace('if name == "main":', "if __name__ == '__main__':")
        raw_json = raw_json.replace("if name == 'main':", "if __name__ == '__main__':")
        try:
            call_data = json.loads(raw_json)
            raw_name = call_data.get("tool", "")
            inp = call_data.get("input", {})

            # Resolve tool name: match case-insensitively against real tool names
            resolved_name = tool_names.get(raw_name.lower(), raw_name)

            blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": resolved_name,
                "input": inp,
            })
            found_tool = True
        except json.JSONDecodeError as e:
            print(f"[worker] Failed to parse tool JSON: {raw_json} - {e}")
            # Malformed tool call — treat as text if we haven't found any valid tools
            if not found_tool:
                blocks.append({"type": "text", "text": match.group(0)})

        last_end = match.end()

    # 2. FIX THE HALLUCINATION: Only append remaining text if no tools were found.
    # If a tool was found, we discard trailing text so Claude Code is forced 
    # to execute the tool rather than reading Gemini's hallucinated result.
    if not found_tool:
        remaining = gemini_text[last_end:].strip()
        if remaining:
            blocks.append({"type": "text", "text": remaining})

    # If no blocks at all, return plain text block
    if not blocks:
        blocks = [{"type": "text", "text": gemini_text}]

    return blocks

# ── Noise / stub detection ─────────────────────────────────────────────────────

def _extract_last_user_message_text(messages: list) -> str:
    """Used only for stub detection — returns cleaned last user text."""
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


# ── Streaming helpers ──────────────────────────────────────────────────────────

async def _anthropic_stream(content_blocks: list, model: str, msg_id: str) -> AsyncGenerator[str, None]:
    """Stream Anthropic SSE events for a list of content blocks."""
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {
        "type": "message_start",
        "message": {"id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": model, "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}},
    })

    for i, block in enumerate(content_blocks):
        if block["type"] == "text":
            yield sse("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""}
            })
            yield sse("ping", {"type": "ping"})
            # Stream text word by word
            words = block["text"].split(" ")
            for j, word in enumerate(words):
                chunk = word if j == 0 else f" {word}"
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": i,
                    "delta": {"type": "text_delta", "text": chunk}
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
                    "input": {}
                }
            })
            # Stream the input JSON as a delta
            input_json = json.dumps(block["input"])
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": input_json}
            })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": i})

    # Determine stop reason: tool_use if any tool blocks, else end_turn
    has_tools = any(b["type"] == "tool_use" for b in content_blocks)
    stop_reason = "tool_use" if has_tools else "end_turn"

    total_tokens = sum(len(b.get("text", "").split()) for b in content_blocks if b["type"] == "text")

    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": max(total_tokens, 1)}
    })
    yield sse("message_stop", {"type": "message_stop"})


# ── Core query runner ──────────────────────────────────────────────────────────

async def _run_query(messages: list, max_tokens: int | None,
                     tools: list, output_config: dict | None) -> list:
    """
    Returns a list of Anthropic content blocks (text and/or tool_use).
    """
    if worker is None:
        raise HTTPException(503, "Worker not ready")

    if _is_internal_only(messages, max_tokens):
        print("[main] → stub")
        if output_config and output_config.get("format", {}).get("type") == "json_schema":
            return [{"type": "text", "text": '{"title": "New session"}'}]
        return [{"type": "text", "text": ""}]

    prompt = _build_conversation_for_gemini(messages, tools)
    # Log only the last user turn for brevity
    last_user = _extract_last_user_message_text(messages)
    print(f"[main] → Gemini: {last_user[:100]}{'...' if len(last_user) > 100 else ''}")
    print(f"[main]   (context: {len(messages)} messages, {len(tools)} tools)")

    try:
        raw = await worker.query(prompt)
        print(f"[main] ← Gemini ({len(raw)} chars): {raw[:80]}{'...' if len(raw) > 80 else ''}")
        blocks = _parse_gemini_response(raw, tools)
        tool_count = sum(1 for b in blocks if b["type"] == "tool_use")
        print(f"[main]   parsed: {len(blocks)} blocks, {tool_count} tool_use")
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

    messages     = body.get("messages", [])
    model        = body.get("model", "web-llm")
    max_tokens   = body.get("max_tokens")
    tools        = body.get("tools") or []
    output_config = body.get("output_config")
    stream       = bool(body.get("stream") or False)

    print(f"[route] stream={stream} max_tokens={max_tokens} msgs={len(messages)} tools={len(tools)}")

    # Prepend system prompt to messages list
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

    # Non-streaming
    has_tools = any(b["type"] == "tool_use" for b in content_blocks)
    stop_reason = "tool_use" if has_tools else "end_turn"
    token_count = max(sum(len(b.get("text","").split()) for b in content_blocks if b["type"]=="text"), 1)

    return JSONResponse(content={
        "id": msg_id, "type": "message", "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": token_count, "output_tokens": token_count,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        },
    }, headers=base_headers)


@app.get("/health")
async def health():
    return {"status": "ok", "worker_ready": worker is not None and worker.ready}