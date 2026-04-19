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
from dotenv import load_dotenv

from worker import PlaywrightWorker

# Load environment variables
load_dotenv(override=True)

# API mode: "claude" (default, for Claude Code) or "normal" (for general LLM API use)
API_MODE = os.getenv("API_MODE", "claude").strip().lower()

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


# ── Mode helpers ────────────────────────────────────────────────────────────────

def _is_claude_mode() -> bool:
    """Check if running in Claude Code optimized mode."""
    return API_MODE == "claude"


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
    if not _is_claude_mode():
        return False  # Don't filter noise in normal mode
    return any(p in text for p in _SYSTEM_NOISE)


# ── Conversation history builder ───────────────────────────────────────────────

def _build_simple_prompt(messages: list, tools: list) -> str:
    """
    Build a simple prompt for normal API mode.
    Just converts the message list to a conversation format without
    Claude Code specific transformations.
    """
    parts = []

    for m in messages:
        role = m.role if hasattr(m, "role") else m.get("role", "")
        content = m.content if hasattr(m, "content") else m.get("content", "")

        if role == "system":
            text = _extract_text(content)
            if text:
                parts.append(f"System: {text}")
        elif role == "user":
            text = _extract_text(content)
            if text:
                parts.append(f"User: {text}")
        elif role == "assistant":
            text = _extract_text(content).strip()
            if text:
                parts.append(f"Assistant: {text}")

    # Add tool descriptions if tools are provided
    if tools:
        tool_desc = "\n\nAvailable tools:\n"
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
            tool_desc += f"- {name}({params}): {desc}\n"
        tool_desc += "\nWhen you need to use a tool, output JSON in this format:\n"
        tool_desc += '{"tool": "TOOL_NAME", "input": {"param": "value"}}'
        parts.append(tool_desc)

    parts.append("Assistant:")
    return "\n\n".join(parts)


def _build_conversation_for_claude(messages: list, tools: list) -> str:
    """
    Build a conversation prompt optimized for Claude Code.
    This applies all the Claude Code specific transformations.
    """
    tool_descriptions = ""
    tool_names = []
    has_tools = bool(tools)
    alias_note = ""

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

        tool_aliases = {
            "Bash": "bash_tool",
            "Write": "create_file",
            "Read": "view",
        }
        for our_name, claude_name in tool_aliases.items():
            if any(t.get("name") == our_name for t in tools):
                alias_note += f"\nNote: '{our_name}' tool maps to Claude's '{claude_name}' tool internally."

    system_prompt = f"""You are acting as an AI coding agent.
{tool_descriptions}{alias_note}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:
1. When a tool is available to accomplish the user's request, you MUST call it.
2. Output ONLY the <tool_call> XML tag. NO prose, NO markdown, NO ```json fences EVER.
Format — reproduce this exactly:
<tool_call>{{"tool": "TOOL_NAME", "input": {{...args...}}}}</tool_call>

3. FORBIDDEN: markdown code blocks (```), "Here's...", "I'll...", "Sure...", "I understand...".
Any response that starts with text instead of <tool_call> is WRONG when a tool is needed.

4. JSON escaping inside string values:
- Escape inner double quotes: \\"
- Newlines: \\n Tabs: \\t
- Python dunders stay as-is: __name__, __main__

5. NEVER use `cat << 'EOF'` or any shell heredoc to create files.
Always use the Write tool: {{"tool": "Write", "input": {{"file_path": "...", "content": "..."}}}}

6. Python formatting rules inside string values:
- Decorators MUST stay on ONE line: @app.get("/") — NEVER split as @app.get\n("/")
- Use \n for newlines inside strings, never literal newlines.

7. If NO tool is needed (pure question), reply with plain text only.

8. MEMORY: Fresh session — no history from prior conversations.

Correct examples:
<tool_call>{{"tool": "Bash", "input": {{"command": "ls -la"}}}}</tool_call>
<tool_call>{{"tool": "Write", "input": {{"file_path": "hello.py", "content": "print('Hello World')\\nprint('done')"}}}}</tool_call>

User request:"""

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

                        # Transform known Claude Code errors into actionable hints.
                        if "has not been read yet" in result_content or \
                            "read it first before writing" in result_content.lower():
                            # Find the file_path from the tool_use_id match in this same content list
                            tool_use_id = block.get("tool_use_id", "")
                            file_path = "the file"
                            for sibling in content:
                                if (isinstance(sibling, dict) and
                                    sibling.get("type") == "tool_use" and
                                    sibling.get("id") == tool_use_id):
                                    file_path = sibling.get("input", {}).get("file_path", file_path)
                                    break
                            read_call = (
                                '<tool_call>{"tool": "Read", "input": {"file_path": "' +
                                file_path + '"}}</tool_call>'
                            )
                            result_content = (
                                f"[Tool error]: Write rejected — must Read '{file_path}' first. "
                                f"IMMEDIATELY call: {read_call} "
                                f"Then after the Read result arrives, call Write."
                            )

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

    if has_tools:
        parts.append("Assistant: <tool_call>")
    else:
        parts.append("Assistant:")

    return "\n\n".join(parts)


def _build_prompt(messages: list, tools: list) -> str:
    """
    Build the appropriate prompt based on API mode.
    - "claude" mode: Full Claude Code optimizations
    - "normal" mode: Simple conversation format
    """
    if _is_claude_mode():
        return _build_conversation_for_claude(messages, tools)
    else:
        return _build_simple_prompt(messages, tools)


# ── Tool call parser ───────────────────────────────────────────────────────────

def _unescape_string_values(obj):
    """
    Recursively convert double-escaped sequences in string values to real chars.
    Called AFTER json.loads so we only deal with literal \\n that Claude
    double-escaped (not JSON-level \\n which json.loads already converted).
    """
    if isinstance(obj, dict):
        return {k: _unescape_string_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unescape_string_values(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    return obj


# Valid non-whitespace characters that may follow a JSON string's closing
# double-quote. Anything else means the '"' was an unescaped inner quote.
_JSON_AFTER_STRING = frozenset(',}]":')


def _fix_json_string_values(raw_json: str) -> str:
    """
    Walk raw LLM JSON char-by-char and repair two common problems inside
    string values:

    1. Bare control characters (newline, tab, CR) -> proper JSON escape
       sequences.  Fixes: json.loads "Invalid control character at line N".

    2. Unescaped inner double-quotes, e.g.:
           "content": "app = FastAPI(title="My App", description="demo")"
       Two-layer heuristic:
       - Layer 1 (bracket depth): while inside a (, [, or { that was opened
         INSIDE the current string, any " must be an inner literal quote.
       - Layer 2 (lookahead): at depth 0, peek at the next non-whitespace
         char after the ".  Valid JSON terminators are always followed by
         one of  , } ] " :  -- anything else is an inner quote to escape.
       Fixes: json.loads "Expecting ',' delimiter".

    Already-valid JSON passes through unchanged.
    Already-escaped sequences (\\\\n, \\\\", etc.) are preserved verbatim.
    """
    result            = []
    in_string         = False
    escape_next       = False
    i                 = 0
    n                 = len(raw_json)
    str_paren_depth   = 0   # ( opened inside current string
    str_bracket_depth = 0   # [ opened inside current string
    str_brace_depth   = 0   # { opened inside current string

    while i < n:
        ch = raw_json[i]

        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue

        if ch == '\\':
            result.append(ch)
            escape_next = True
            i += 1
            continue

        if ch == '"':
            if not in_string:
                result.append(ch)
                in_string = True
                str_paren_depth = str_bracket_depth = str_brace_depth = 0
            else:
                any_depth = str_paren_depth + str_bracket_depth + str_brace_depth
                if any_depth > 0:
                    result.append('\\')
                    result.append('"')
                else:
                    j = i + 1
                    while j < n and raw_json[j] in ' \t\r\n':
                        j += 1
                    next_ch = raw_json[j] if j < n else ''
                    if next_ch in _JSON_AFTER_STRING or next_ch == '':
                        result.append(ch)
                        in_string = False
                    else:
                        result.append('\\')
                        result.append('"')
            i += 1
            continue

        if in_string:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
            if ch == '(':
                str_paren_depth += 1
            elif ch == ')':
                str_paren_depth = max(0, str_paren_depth - 1)
            elif ch == '[':
                str_bracket_depth += 1
            elif ch == ']':
                str_bracket_depth = max(0, str_bracket_depth - 1)
            elif ch == '{':
                str_brace_depth += 1
            elif ch == '}':
                str_brace_depth = max(0, str_brace_depth - 1)
        else:
            result.append(ch)

        i += 1

    return ''.join(result)


def _fix_unescaped_control_chars(raw_json: str) -> str:
    """Alias kept for compatibility -- delegates to _fix_json_string_values."""
    return _fix_json_string_values(raw_json)

def _fallback_extract_tool_call(raw_json: str) -> dict | None:
    """
    Regex-based last-resort fallback for when json.loads fails even after
    control-char repair (e.g. unescaped double-quotes inside string values).

    Handles ALL multi-line string fields — not just "content" — by doing
    positional extraction between consecutive field keys.  This correctly
    recovers old_string / new_string for Edit calls and command for Bash calls.
    """
    tool_m = re.search(r'"tool"\s*:\s*"([^"]+)"', raw_json)
    if not tool_m:
        return None
    tool = tool_m.group(1)

    input_dict: dict = {}

    # ── Simple single-line string fields ──────────────────────────────────────
    for field in ("file_path", "path", "description", "query"):
        m = re.search(rf'"{field}"\s*:\s*"([^"\n]*)"', raw_json)
        if m:
            val = m.group(1).replace('\\n', '\n').replace('\\t', '\t')
            input_dict[field] = val

    # ── Boolean / numeric fields ───────────────────────────────────────────────
    for field in ("replace_all",):
        m = re.search(rf'"{field}"\s*:\s*(true|false|\d+)', raw_json)
        if m:
            raw_val = m.group(1)
            if raw_val == "true":
                input_dict[field] = True
            elif raw_val == "false":
                input_dict[field] = False
            else:
                input_dict[field] = int(raw_val)

    # ── Multi-line string fields (positional extraction) ──────────────────────
    # These fields can contain literal newlines, unescaped quotes, etc.
    # We find each field's opening `"FIELD": "` marker, then read until the
    # next field's marker or the end of the JSON blob.
    #
    # Listed in typical schema order so the "up to next field" heuristic works.
    multiline_fields = ["command", "old_string", "new_string", "content"]

    field_positions: dict[str, int] = {}
    for field in multiline_fields:
        m = re.search(rf'"{field}"\s*:\s*"', raw_json)
        if m:
            field_positions[field] = m.end()   # index right after the opening "

    sorted_fields = sorted(field_positions.items(), key=lambda kv: kv[1])

    for idx, (field, start) in enumerate(sorted_fields):
        if idx + 1 < len(sorted_fields):
            next_field_name = sorted_fields[idx + 1][0]
            next_key_m = re.search(
                rf'"{re.escape(next_field_name)}"\s*:\s*"',
                raw_json[start:]
            )
            if next_key_m:
                raw_value = raw_json[start : start + next_key_m.start()]
            else:
                raw_value = raw_json[start:]
        else:
            # Last multi-line field — read to the closing `"}}` at the tail
            rest = raw_json[start:]
            end_m = re.search(r'"\s*\}\s*\}\s*$', rest)
            if end_m:
                raw_value = rest[: end_m.start()]
            else:
                last_q = rest.rfind('"')
                raw_value = rest[:last_q] if last_q != -1 else rest

        # Strip trailing quote+comma that was the field delimiter
        raw_value = re.sub(r'",?\s*$', '', raw_value)

        value = (
            raw_value
            .replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace('\\r', '\r')
            .replace('\\"', '"')
        )
        input_dict[field] = value
        print(f"[fallback] Extracted {field!r} ({len(value)} chars)")

    return {"tool": tool, "input": input_dict} if input_dict else None


def _validate_tool_input(tool_name: str, inp: dict, available_tools: list) -> list[str]:
    """
    Check that all required parameters for the tool are present and non-empty.
    Returns a list of missing field names (empty list = all good).
    """
    tool_schema = next(
        (t for t in available_tools if t.get("name", "").lower() == tool_name.lower()),
        None,
    )
    if not tool_schema:
        return []
    required = tool_schema.get("input_schema", {}).get("required", [])
    return [f for f in required if not inp.get(f)]


# ── Fix 3: Sanitize Python file content written via tool calls ────────────────

def _sanitize_python_content(content: str) -> str:
    """
    Fix common LLM formatting errors in Python source written via tool calls.

    ChatGPT sometimes splits decorator + route onto separate lines:
        @app.get
        ("/")
    instead of the valid:
        @app.get("/")

    This regex re-joins any decorator immediately followed by a dangling
    argument list on the next line.
    """
    content = re.sub(r'(@\w[\w.]*)\s*\n\s*(\()', r'\1\2', content)
    return content


def _strip_bash_command_garbage(command: str) -> str:
    """
    ChatGPT sometimes embeds extra Bash tool parameters (timeout, description,
    dangerouslyDisableSandbox) inside the command string value itself, like:

        cat << 'EOF' > f.py\\n...\\nEOF", "timeout": 200000, "description": "..."

    The fallback extractor picks up this trailing junk. Strip it cleanly.
    """
    # Case 1: heredoc — keep only up to and including the terminating EOF line
    heredoc_end = re.search(r'\nEOF\s*(?:",|$)', command)
    if heredoc_end:
        command = command[:heredoc_end.start() + 4]  # keep the \nEOF

    # Case 2: generic — strip trailing `", "some_key": ...` leakage
    command = re.sub(
        r'",\s*"(?:timeout|description|run_in_background|dangerouslyDisableSandbox)[^}]*$',
        '', command
    )
    return command.strip()


def _parse_simple_response(raw: str) -> list:
    """
    Parse a response in normal API mode.
    Just returns the raw text as a text content block.
    """
    # Strip any prompt echo that might be included
    raw = raw.strip()
    if not raw:
        return [{"type": "text", "text": ""}]
    return [{"type": "text", "text": raw}]


def _parse_claude_response(raw: str, available_tools: list) -> list:
    """
    Parse an LLM text response into Anthropic content blocks for Claude Code mode.

    Per <tool_call> block, three parse attempts are made in order:
    1. _fix_unescaped_control_chars → json.loads (fixes bare newlines in strings)
    2. json.loads on original raw_json (already-valid JSON)
    3. _fallback_extract_tool_call (regex positional, last resort)

    After parsing, required fields are validated. Blocks with missing required
    fields are dropped (logged as WARNING) rather than passed to Claude Code
    where they would cause InputValidationError loops.
    """
    tool_names = {t.get("name", "").lower(): t.get("name", "") for t in available_tools}

    stripped = raw.strip()
    if stripped and not stripped.startswith("<tool_call>"):
        if stripped.startswith("{") or stripped.startswith('{"'):
            stripped = f"<tool_call>{stripped}"
            if "</tool_call>" not in stripped:
                stripped = stripped + "</tool_call>"
        raw = stripped

    # Handle truncated tool_call (no closing tag)
    if "<tool_call>" in raw and "</tool_call>" not in raw:
        start_idx = raw.find("<tool_call>")
        if start_idx != -1:
            potential_json = raw[start_idx + len("<tool_call>"):].strip()
            if potential_json.startswith("{"):
                tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', potential_json)
                if tool_match:
                    print(f"[parser] Detected truncated tool_call with tool: {tool_match.group(1)}")
                    raw = f"<tool_call>{potential_json}</tool_call>"

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

        # Fix common dunder-stripping bug
        raw_json = raw_json.replace('if name == "main":', "if __name__ == '__main__':")
        raw_json = raw_json.replace("if name == 'main':", "if __name__ == '__main__':")

        call_data = None

        # ── Attempt 1: repair unescaped control chars, then json.loads ────────
        # Fixes ChatGPT's habit of writing literal newlines inside JSON strings.
        try:
            repaired = _fix_unescaped_control_chars(raw_json)
            call_data = json.loads(repaired)
            call_data["input"] = _unescape_string_values(call_data.get("input", {}))
            if repaired != raw_json:
                print("[parser] json.loads succeeded after control-char repair")
        except json.JSONDecodeError:
            pass

        # ── Attempt 2: raw json.loads (already valid JSON) ────────────────────
        if call_data is None:
            try:
                call_data = json.loads(raw_json)
                call_data["input"] = _unescape_string_values(call_data.get("input", {}))
            except json.JSONDecodeError as e:
                print(f"[parser] json.loads failed ({e}), trying fallback extractor")

                # ── Attempt 3: regex positional fallback ──────────────────────
                call_data = _fallback_extract_tool_call(raw_json)
                if call_data:
                    print(f"[parser] Fallback extractor succeeded for tool: {call_data.get('tool')}")
                else:
                    print(f"[parser] Fallback extractor also failed for: {raw_json[:120]!r}")

        if call_data:
            raw_name = call_data.get("tool", "")
            inp      = call_data.get("input", {})

            # Sanitize Python content for write-type tools
            if raw_name.lower() in ("write", "create_file") and "content" in inp:
                inp["content"] = _sanitize_python_content(inp["content"])
            # Fix broken decorators and strip leaked JSON params from Bash commands
            if raw_name.lower() == "bash" and "command" in inp:
                inp["command"] = _strip_bash_command_garbage(inp["command"])
                inp["command"] = _sanitize_python_content(inp["command"])

            # Validate required fields — drop silently broken tool calls
            missing = _validate_tool_input(raw_name, inp, available_tools)
            if missing:
                print(
                    f"[parser] WARNING: tool {raw_name!r} missing required fields "
                    f"{missing} — dropping block to prevent InputValidationError loop"
                )
                if not found_tool:
                    blocks.append({"type": "text", "text": match.group(0)})
                last_end = match.end()
                continue

            resolved = tool_names.get(raw_name.lower(), raw_name)
            blocks.append({
                "type": "tool_use",
                "id":   f"toolu_{uuid.uuid4().hex[:24]}",
                "name": resolved,
                "input": inp,
            })
            found_tool = True
        else:
            if not found_tool:
                blocks.append({"type": "text", "text": match.group(0)})

        last_end = match.end()

    # ── JSON code-fence fallback (no <tool_call> found at all) ────────────────
    if not found_tool:
        fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        for fm in fence_pattern.finditer(raw):
            try:
                repaired = _fix_unescaped_control_chars(fm.group(1))
                obj = json.loads(repaired)
                if "tool" in obj and "input" in obj:
                    raw_name     = obj.get("tool", "")
                    resolved     = tool_names.get(raw_name.lower(), raw_name)
                    obj["input"] = _unescape_string_values(obj.get("input", {}))

                    if raw_name.lower() in ("write", "create_file") and "content" in obj["input"]:
                        obj["input"]["content"] = _sanitize_python_content(obj["input"]["content"])
                    if raw_name.lower() == "bash" and "command" in obj["input"]:
                        obj["input"]["command"] = _strip_bash_command_garbage(obj["input"]["command"])
                        obj["input"]["command"] = _sanitize_python_content(obj["input"]["command"])

                    missing = _validate_tool_input(raw_name, obj["input"], available_tools)
                    if missing:
                        print(f"[parser] WARNING: fence fallback tool {raw_name!r} missing required fields: {missing}")
                        continue

                    blocks = [{
                        "type": "tool_use",
                        "id":   f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": resolved,
                        "input": obj.get("input", {}),
                    }]
                    found_tool = True
                    last_end   = len(raw)
                    print(f"[parser] Recovered tool call from JSON code fence: {raw_name}")
                    break
            except json.JSONDecodeError:
                pass

    # Trailing text — only when no tool was found
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


# ── Fix 1: Context-aware _needs_tool_call ─────────────────────────────────────

def _needs_tool_call(messages: list) -> bool:
    """
    Return True only if the original user request contains action keywords AND
    the most recent tool result does not signal successful completion.

    Without the completion check, every turn for "make a file" would keep
    triggering the correction prompt even after Write succeeded — causing the
    model to emit junk tool calls like TaskList in an infinite loop.
    """
    text = _extract_last_user_message_text(messages).lower()
    if not text:
        return False

    completion_signals = [
        "updated successfully",
        "created successfully",
        "written successfully",
        "deleted successfully",
        "no tasks found",
        "task complete",
        "file has been",
        "successfully written",
        "successfully created",
        "successfully updated",
        "successfully deleted",
    ]
    for m in reversed(messages):
        role    = m.role    if hasattr(m, "role")    else m.get("role", "")
        content = m.content if hasattr(m, "content") else m.get("content", "")
        if role != "user" or not isinstance(content, list):
            continue
        has_tool_result = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            has_tool_result = True
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_content = " ".join(
                    b.get("text", "") for b in result_content
                    if isinstance(b, dict)
                )
            result_lower = result_content.lower()
            if any(sig in result_lower for sig in completion_signals):
                print("[main] Completion signal found in last tool result — suppressing correction")
                return False
        if has_tool_result:
            break

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

_CORRECTION_PROMPT = "Output only the <tool_call> block, nothing else."


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

    prompt = _build_prompt(messages, tools)
    if _is_claude_mode():
        print(f"[DEBUG] Prompt sent to Claude:\n{prompt}\n---")

    last_user = _extract_last_user_message_text(messages)
    print(f"[main] → LLM: {last_user[:100]}{'...' if len(last_user) > 100 else ''}")
    print(f"[main] (context: {len(messages)} messages, {len(tools)} tools, mode={API_MODE})")

    try:
        raw = await worker.query(prompt)
        print(f"[main] ← LLM ({len(raw)} chars): {raw[:120]}{'...' if len(raw) > 120 else ''}")

        if raw.startswith(prompt):
            raw = raw[len(prompt):]
            print(f"[main] Stripped prompt, remaining length: {len(raw)}")
        else:
            max_prefix = min(len(prompt), len(raw))
            i = 0
            while i < max_prefix and prompt[i] == raw[i]:
                i += 1
            if i > 30:
                raw = raw[i:]
                print(f"[main] Stripped common prefix of length {i}, remaining length: {len(raw)}")

        # Parse response based on mode
        if _is_claude_mode():
            blocks = _parse_claude_response(raw, tools)
        else:
            blocks = _parse_simple_response(raw)

        tool_count = sum(1 for b in blocks if b["type"] == "tool_use")
        print(f"[main] parsed: {len(blocks)} blocks, {tool_count} tool_use")

        # Tool call correction only applies in Claude mode
        if _is_claude_mode() and tools and tool_count == 0 and _needs_tool_call(messages):
            print("[main] No tool call in response — sending correction prompt")
            correction = prompt + "\n\n" + raw + "\n\nUser: " + _CORRECTION_PROMPT + "\nAssistant: <tool_call>"
            raw2 = await worker.query(correction)
            print(f"[main] ← LLM retry ({len(raw2)} chars): {raw2[:120]}{'...' if len(raw2) > 120 else ''}")

            if raw2.startswith(prompt):
                raw2 = raw2[len(prompt):]
            else:
                max_prefix = min(len(prompt), len(raw2))
                i = 0
                while i < max_prefix and prompt[i] == raw2[i]:
                    i += 1
                if i > 30:
                    raw2 = raw2[i:]

            blocks2 = _parse_claude_response(raw2, tools)
            tool_count2 = sum(1 for b in blocks2 if b["type"] == "tool_use")
            if tool_count2 > 0:
                print(f"[main] Retry succeeded: {tool_count2} tool_use block(s)")
                blocks = blocks2
            else:
                # ── Second retry: re-send the original prompt fresh ────────────
                # The correction prompt failed — start a clean new chat and
                # re-submit the original prompt so the model isn't confused by
                # the prior bad exchange.
                print("[main] Correction prompt also returned no tool_use — re-sending original prompt fresh")
                raw3 = await worker.query(prompt)
                print(f"[main] ← LLM fresh retry ({len(raw3)} chars): {raw3[:120]}{'...' if len(raw3) > 120 else ''}")

                if raw3.startswith(prompt):
                    raw3 = raw3[len(prompt):]
                else:
                    max_prefix = min(len(prompt), len(raw3))
                    i = 0
                    while i < max_prefix and prompt[i] == raw3[i]:
                        i += 1
                    if i > 30:
                        raw3 = raw3[i:]

                blocks3 = _parse_claude_response(raw3, tools)
                tool_count3 = sum(1 for b in blocks3 if b["type"] == "tool_use")
                if tool_count3 > 0:
                    print(f"[main] Fresh retry succeeded: {tool_count3} tool_use block(s)")
                    blocks = blocks3
                else:
                    print("[main] Fresh retry also returned no tool_use — returning original text")

        # Tool deduplication only applies in Claude mode
        if _is_claude_mode():
            tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
            if tool_blocks:
                # ── Deduplicate against conversation history ───────────────────────
                history_fingerprints: set[str] = set()
                # Also track file-level fingerprints: tool_name + file_path only.
                # This catches cases where fallback extractor truncates content
                # differently each time, making full-input fingerprints differ
                # even though it's the same logical Write/Read operation.
                history_file_fingerprints: set[str] = set()
                for m in messages:
                    role = m.role if hasattr(m, "role") else m.get("role", "")
                    content = m.content if hasattr(m, "content") else m.get("content", "")
                    if role != "assistant" or not isinstance(content, list):
                        continue
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            fp = json.dumps(
                                {"name": blk.get("name"), "input": blk.get("input")},
                                sort_keys=True
                            )
                            history_fingerprints.add(fp)
                            # File-level: track Write attempts per file path
                            blk_inp = blk.get("input", {})
                            blk_name = blk.get("name", "").lower()
                            if blk_name in ("write", "create_file") and "file_path" in blk_inp:
                                history_file_fingerprints.add(
                                    f"write::{blk_inp['file_path']}"
                                )

                if history_fingerprints:
                    def _is_dup(b):
                        full_fp = json.dumps(
                            {"name": b.get("name"), "input": b.get("input")},
                            sort_keys=True
                        )
                        if full_fp in history_fingerprints:
                            return True
                        # File-level dedup for Write tool
                        b_name = b.get("name", "").lower()
                        b_inp = b.get("input", {})
                        if b_name in ("write", "create_file") and "file_path" in b_inp:
                            file_fp = f"write::{b_inp['file_path']}"
                            if file_fp in history_file_fingerprints:
                                print(f"[main] File-level dedup hit for {b_inp['file_path']!r}")
                                return True
                        return False

                    new_tool_blocks = [b for b in tool_blocks if not _is_dup(b)]
                    removed = len(tool_blocks) - len(new_tool_blocks)
                    if removed:
                        print(f"[main] Deduped {removed} echoed history tool_call(s), {len(new_tool_blocks)} new remain")

                    # Fix 2: Don't fall back to deduped blocks — return end_turn cleanly
                    tool_blocks = new_tool_blocks

                if tool_blocks:
                    print(f"[main] Filtering out {len(blocks) - len(tool_blocks)} text blocks, keeping {len(tool_blocks)} tool_use blocks")
                    return tool_blocks
                else:
                    # All blocks were duplicates. Instead of returning a text hint
                    # (which ChatGPT ignores and loops on), synthesize a Read tool
                    # call for the file that was being written. This gives Claude
                    # Code a concrete next step and breaks the loop.
                    written_files = []
                    for b in tool_blocks:
                        b_name = b.get("name", "").lower()
                        b_inp = b.get("input", {})
                        if b_name in ("write", "create_file") and "file_path" in b_inp:
                            written_files.append(b_inp["file_path"])
                    if written_files:
                        fp = written_files[0]
                        print(f"[main] All tool blocks duped — synthesizing Read for {fp!r}")
                        return [{
                            "type": "tool_use",
                            "id": f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": "Read",
                            "input": {"file_path": fp},
                        }]
                    else:
                        print("[main] All tool blocks were duplicates (no file path) — returning end_turn")
                        return [{"type": "text", "text": "Done."}]

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