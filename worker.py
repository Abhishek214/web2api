"""
Playwright worker — automates LLM web apps (Gemini, ChatGPT, Claude.ai).
"""

import asyncio
import json
import os
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from playwright_stealth import stealth_async
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

PROVIDERS: dict = {
    "gemini": {
        "url":          "https://gemini.google.com",
        "new_chat_url": "https://gemini.google.com/app",
        "domain":       "gemini.google.com",
        "selectors": {
            "input":       "rich-textarea .ql-editor",
            "send_button": "button.send-button",
            "response":    "model-response:last-of-type .markdown",
            "stop_button": "button[aria-label='Stop response']",
        },
    },
    "chatgpt": {
        "url":          "https://chatgpt.com",
        "new_chat_url": "https://chatgpt.com/",
        "domain":       "chatgpt.com",
        "selectors": {
            "input":       "#prompt-textarea",
            "send_button": "button[data-testid='send-button']",
            # Primary: testid-based selector is stable across all turns.
            # .markdown exists only sometimes (short/simple responses); the
            # outer conversation-turn container always exists.
            "response":    "[data-testid^='conversation-turn'] [data-message-author-role='assistant']",
            "response_fallbacks": [
                "div[data-message-author-role='assistant'] .markdown",
                "div[data-message-author-role='assistant'] .prose",
                "div[data-message-author-role='assistant']",
            ],
            "stop_button": "button[data-testid='stop-button']",
        },
    },
    "claude": {
        "url":          "https://claude.ai",
        "new_chat_url": "https://claude.ai/new",
        "domain":       "claude.ai",
        "selectors": {
            "input":       "div[contenteditable='true'].ProseMirror",
            "send_button": "button[aria-label='Send message']",
            # Selectors are tried in order; first match wins.
            # IMPORTANT: only list selectors that start at count=0 on a fresh
            # conversation page — generic classes like .prose or .whitespace-pre-wrap
            # already exist in UI chrome and cause false-positive matches.
            "response":    ".font-claude-message",
            "response_fallbacks": [
                '[data-testid="assistant-message"]',
                '[data-is-streaming]',
                ".claude-message",
            ],
            "stop_button": "button[aria-label='Stop']",
        },
    },
}

TARGET_PROVIDER = os.getenv("TARGET_PROVIDER", "gemini").strip().lower()

if TARGET_PROVIDER not in PROVIDERS:
    raise ValueError(
        f"Unknown TARGET_PROVIDER={TARGET_PROVIDER!r}. "
        f"Valid options: {list(PROVIDERS.keys())}"
    )

_CFG         = PROVIDERS[TARGET_PROVIDER]
TARGET_URL   = os.getenv("TARGET_URL", _CFG["url"])
SELECTORS    = _CFG["selectors"]
NEW_CHAT_URL = _CFG["new_chat_url"]

# The original primary response selector — never mutated.
_PRIMARY_RESPONSE_SEL = SELECTORS["response"]

RESPONSE_START_TIMEOUT = 20_000
RESPONSE_POLL_INTERVAL =    500
RESPONSE_MAX_WAIT      =    180
MAX_RETRIES            =      2


class PlaywrightWorker:
    def __init__(self, headless: bool = True):
        self.headless    = headless
        self.ready       = False
        self._playwright = None
        self._browser:  Browser        | None = None
        self._context:  BrowserContext | None = None
        self._page:     Page           | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        self._playwright = await async_playwright().start()
        debug_port = int(os.getenv("CHROME_DEBUG_PORT", "9222"))
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(
                f"http://localhost:{debug_port}"
            )
            print(f"[worker] Connected to Chrome on port {debug_port}")
            self._context = (
                self._browser.contexts[0]
                if self._browser.contexts
                else await self._browser.new_context()
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to Chrome on port {debug_port}.\n"
                f"Start Chrome with: --remote-debugging-port={debug_port}\n"
                f"Original error: {e}"
            )

        target_page = None
        for page in self._context.pages:
            if _CFG["domain"] in page.url:
                target_page = page
                break

        if target_page:
            self._page = target_page
            await self._page.bring_to_front()
            print(f"[worker] Reusing existing tab: {self._page.url}")
        else:
            self._page = await self._context.new_page()
            await self._page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)

        if HAS_STEALTH:
            await stealth_async(self._page)

        try:
            await self._page.wait_for_selector(
                SELECTORS["input"], state="visible", timeout=30_000
            )
        except Exception:
            print("[worker] WARNING: Input box not visible. Are you logged in?")

        self.ready = True
        print(f"[worker] Ready — provider={TARGET_PROVIDER}  url={TARGET_URL}")

    async def stop(self):
        if self._playwright:
            await self._playwright.stop()
        self.ready = False

    async def query(self, prompt: str) -> str:
        async with self._lock:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return await self._do_query(prompt)
                except Exception as e:
                    print(f"[worker] Attempt {attempt} failed: {e}")
                    if attempt < MAX_RETRIES:
                        print("[worker] Reloading and retrying…")
                        try:
                            await self._page.reload(
                                wait_until="domcontentloaded", timeout=30_000
                            )
                            await asyncio.sleep(2)
                        except Exception:
                            pass
                    else:
                        raise

    async def _reset_to_new_chat(self):
        """
        Navigate directly to the new-chat URL instead of clicking a button.
        Button clicks are unreliable due to SVG/overlay intercepts.
        Also resets the response selector back to the primary default so that
        fallback mutations from a previous query don't persist.
        """
        # ── KEY FIX: reset selector before every new conversation ─────────────
        # Without this, a fallback selector found in turn N (e.g. [data-is-streaming])
        # becomes the primary selector for turn N+1, causing misses on fresh pages.
        SELECTORS["response"] = _PRIMARY_RESPONSE_SEL

        page = self._page
        print(f"[worker] → new chat: {NEW_CHAT_URL}")
        await page.goto(NEW_CHAT_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(1.2)   # let React/Svelte settle after navigation
        await page.wait_for_selector(
            SELECTORS["input"], state="visible", timeout=15_000
        )
        print("[worker] Input ready")

    async def _fill_input(self, prompt: str):
        """
        Fill the input in a way that triggers React/ProseMirror synthetic events.
        Clipboard paste fires the native paste event which React and ProseMirror
        both handle correctly.
        """
        page     = self._page
        input_el = page.locator(SELECTORS["input"])

        await input_el.click()
        await asyncio.sleep(0.2)

        await page.evaluate("(t) => navigator.clipboard.writeText(t)", prompt)

        is_mac = "darwin" in os.uname().sysname.lower()
        modifier = "Meta" if is_mac else "Control"
        await page.keyboard.press(f"{modifier}+v")
        await asyncio.sleep(0.4)

        await input_el.dispatch_event("input")

    async def _click_send(self):
        """
        Click the send button; fall back to Enter key if it's still not clickable.
        """
        page     = self._page
        send_sel = SELECTORS["send_button"]

        try:
            await page.wait_for_function(
                """(sel) => {
                    const btn = document.querySelector(sel);
                    return btn && !btn.disabled &&
                           btn.getAttribute('aria-disabled') !== 'true';
                }""",
                arg=send_sel,
                timeout=8_000,
            )
            await page.evaluate(
                "(sel) => document.querySelector(sel).click()", send_sel
            )
            print("[worker] Send button clicked (JS)")
            return
        except Exception as e:
            print(f"[worker] Send button JS click failed ({e}), trying Enter…")

        await page.keyboard.press("Enter")
        print("[worker] Sent via Enter key")

    async def _resolve_response_locator(self):
        """
        Return the primary response locator and its current element count.
        Count == 0 is EXPECTED on a fresh page (pre-send baseline).
        """
        page    = self._page
        primary = SELECTORS["response"]
        loc     = page.locator(primary)
        count   = await loc.count()
        return loc, count

    async def _do_query(self, prompt: str) -> str:
        page = self._page
        await self._reset_to_new_chat()

        # Pre-paste snapshot — used only for CSS locator count baseline.
        # Do NOT use this for body-text extraction: after _fill_input, the
        # input box content is included in document.body.innerText, so
        # full_text[pre_paste_len:] would return the entire prompt + response.
        pre_paste_len = await page.evaluate("() => document.body.textContent.length")

        response_locator, count_before = await self._resolve_response_locator()

        await self._fill_input(prompt)

        # Post-paste snapshot — body now contains the prompt in the input box.
        post_paste_len = await page.evaluate("() => document.body.textContent.length")

        await self._click_send()

        # After sending, the input box is cleared by the UI.  Wait for that to
        # settle, then take a clean post-send snapshot.  The body-text fallback
        # must use THIS baseline so it only returns the LLM's actual reply.
        await asyncio.sleep(0.6)
        post_send_len = await page.evaluate("() => document.body.textContent.length")
        print(
            f"[worker] Sending prompt "
            f"(css_baseline={count_before}, "
            f"pre={pre_paste_len}, post_paste={post_paste_len}, post_send={post_send_len}, "
            f"sel={SELECTORS['response']!r})…"
        )

        # Use post_send_len for all body-text growth checks in Phase 1 & 2.
        body_baseline = post_send_len

        # ── Phase 1: wait for response to START appearing ─────────────────────
        response_started = False
        elapsed = 0.6  # already waited 0.6s above

        while elapsed < RESPONSE_START_TIMEOUT / 1000:
            await asyncio.sleep(0.5)
            elapsed += 0.5

            if response_locator is not None:
                cur = await response_locator.count()
                if cur > count_before:
                    print(f"[worker] CSS locator matched new block ({cur} elements) after {elapsed:.1f}s")
                    response_started = True
                    break
                if cur == 0 and elapsed > 3.0:
                    # Only probe fallbacks after giving the primary a fair chance.
                    # Only switch if the fallback already has elements — a count of 0
                    # means either "wrong selector" or "not rendered yet", and we
                    # can't tell which, so we skip it rather than locking in a dud.
                    for fb in SELECTORS.get("response_fallbacks", []):
                        fb_loc = page.locator(fb)
                        fb_count = await fb_loc.count()
                        if fb_count > 0:
                            print(f"[worker] Switching to fallback selector: {fb!r} (count={fb_count})")
                            SELECTORS["response"] = fb
                            response_locator = fb_loc
                            count_before = 0  # detect the *next* block appearing
                            break

            cur_len = await page.evaluate("() => document.body.textContent.length")
            if cur_len > body_baseline + 50:
                print(f"[worker] Body text grew {body_baseline}→{cur_len} after {elapsed:.1f}s — response started")
                response_started = True
                break

        if not response_started:
            tried = [SELECTORS["response"]] + SELECTORS.get("response_fallbacks", [])
            raise TimeoutError(
                f"No response from {TARGET_PROVIDER} after {elapsed:.0f}s. "
                f"Tried CSS selectors: {tried}."
            )

# ── Phase 2: wait for response to FINISH ──────────────────────────────
        if response_locator is not None:
            # Give the primary selector a short window to appear if Phase 1 exited early via body text growth
            for _ in range(6):
                if await response_locator.count() > count_before:
                    return (await self._wait_for_stable_response(response_locator.last, body_baseline)).strip()
                await asyncio.sleep(0.5)

        for fb in SELECTORS.get("response_fallbacks", []):
            if fb == SELECTORS.get("response"):
                continue
            fb_loc = self._page.locator(fb)
            fb_baseline = await fb_loc.count()
            for _ in range(6):
                await asyncio.sleep(0.5)
                if await fb_loc.count() > fb_baseline:
                    print(f"[worker] Phase-2 fallback matched: {fb!r}")
                    SELECTORS["response"] = fb
                    return (await self._wait_for_stable_response(fb_loc.last, body_baseline)).strip()

        print("[worker] WARNING: All CSS selectors failed — using JS page-text fallback")
        # Give streaming 2s to advance before entering the stability loop.
        # Without this, _wait_for_stable_text can exit after just 4 polls on
        # the first few chars of footer/cookie text that happened to appear
        # right when Phase 1 detected body growth.
        await asyncio.sleep(2.0)
        return (await self._wait_for_stable_text(body_baseline)).strip()

    async def _wait_for_stable_text(self, snapshot_len: int) -> str:
        """
        CSS-selector-free response extraction.
        Polls document.body.innerText until it stops growing.
        """
        page              = self._page
        prev_len          = snapshot_len
        stable_count      = 0
        elapsed           = 0.0
        last_progress_log = 0.0

        while elapsed < RESPONSE_MAX_WAIT:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL / 1000)
            elapsed += RESPONSE_POLL_INTERVAL / 1000

            stop_btn      = page.locator(SELECTORS["stop_button"])
            is_generating = await stop_btn.is_visible()

            # Use textContent length to avoid inner_text rendering artefacts
            cur_len = await page.evaluate("() => document.body.textContent.length")

            if cur_len == prev_len:
                stable_count += 1
            else:
                stable_count = 0
                prev_len     = cur_len

            if elapsed - last_progress_log >= 15.0:
                print(
                    f"[worker] body-text waiting… {elapsed:.0f}s, "
                    f"{cur_len} chars (+{cur_len - snapshot_len} over baseline), "
                    f"stable_count={stable_count}, generating={is_generating}"
                )
                last_progress_log = elapsed

            grown_enough = (cur_len - snapshot_len) >= 100
            if stable_count >= 4 and not is_generating and grown_enough:
                print(f"[worker] Body text stable after {elapsed:.1f}s ({cur_len} chars, +{cur_len - snapshot_len} over baseline)")
                full_text = await page.evaluate("() => document.body.textContent")
                return full_text[snapshot_len:].strip()
            elif stable_count >= 4 and not is_generating and not grown_enough:
                print(f"[worker] Body stable but only +{cur_len - snapshot_len} chars over baseline — waiting for real content")
                stable_count = 0

        print("[worker] Timeout in _wait_for_stable_text — returning partial")
        full_text = await page.evaluate("() => document.body.textContent")
        return full_text[snapshot_len:].strip()

    async def _wait_for_stable_response(self, locator, body_baseline: int = 0) -> str:
        """
        Wait until the response element stops changing, then return its text.

        Uses textContent (via JS evaluate) instead of inner_text() because:
        - inner_text() is a *rendered* API: it applies CSS layout and converts
          \n escape sequences inside JSON strings to actual newline chars (0x0A),
          which then breaks json.loads with "Invalid control character".
        - textContent returns the raw concatenated text of all DOM text nodes,
          preserving \n as the two literal characters backslash + n.

        Stability threshold is reduced to 2 when the stop button is gone, so
        we exit quickly after streaming ends rather than waiting for post-render
        syntax-highlighting micro-updates to settle.
        """
        page                   = self._page
        prev_text              = ""
        stable_count           = 0
        elapsed                = 0.0
        last_progress_log      = 0.0
        stop_gone_since        = None  # timestamp when stop button first disappeared
        incomplete_reset_count = 0     # how many times we've reset due to tool_call_incomplete
        # After this many resets the element text is clearly truncated (e.g.
        # ChatGPT rendered <tool_call> as an HTML element so </tool_call>
        # never appears in textContent). Fall back to body.textContent.
        MAX_INCOMPLETE_RESETS  = 3

        while elapsed < RESPONSE_MAX_WAIT:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL / 1000)
            elapsed += RESPONSE_POLL_INTERVAL / 1000

            stop_btn      = page.locator(SELECTORS["stop_button"])
            is_generating = await stop_btn.is_visible()

            # Track when the stop button disappears so we can apply a tighter
            # stability threshold — post-render DOM tweaks shouldn't stall us.
            if not is_generating:
                if stop_gone_since is None:
                    stop_gone_since = elapsed
            else:
                stop_gone_since = None

            try:
                # Use textContent to get raw text nodes, NOT inner_text() which
                # applies CSS rendering and mangles JSON escape sequences.
                current_text = await locator.evaluate("el => el.textContent")
                if not current_text:
                    current_text = prev_text
            except Exception:
                current_text = prev_text

            if current_text == prev_text:
                stable_count += 1
            else:
                stable_count = 0
                prev_text    = current_text

            # Progress heartbeat every 15s
            if elapsed - last_progress_log >= 15.0:
                print(
                    f"[worker] Still waiting… {elapsed:.0f}s elapsed, "
                    f"{len(current_text)} chars, stable_count={stable_count}, "
                    f"generating={is_generating}"
                )
                last_progress_log = elapsed

            # Never exit while a tool_call is visibly incomplete — the stop button
            # disappears slightly before ChatGPT finishes appending the closing tag.
            has_open  = "<tool_call>" in current_text
            has_close = "</tool_call>" in current_text
            tool_call_incomplete = has_open and not has_close

            # Exit conditions:
            # 1. Fast exit: 2 stable polls once stop button gone >1s AND response looks complete
            # 2. Normal:    4 stable polls (used when tool_call is still incomplete, giving
            #               ChatGPT more time to finish appending the closing tag)
            stop_gone_long_enough = (
                stop_gone_since is not None and
                (elapsed - stop_gone_since) >= 1.0
            )
            if tool_call_incomplete:
                needed_stable = 6   # give extra time for closing tag to appear
            elif stop_gone_long_enough:
                needed_stable = 2
            else:
                needed_stable = 4

            if stable_count >= needed_stable and not is_generating:
                if tool_call_incomplete:
                    incomplete_reset_count += 1
                    if incomplete_reset_count >= MAX_INCOMPLETE_RESETS:
                        # ChatGPT renders <tool_call> as a real HTML element so the
                        # tag strings are absent from every textContent read.
                        # Three-stage fallback:
                        #   1. body.textContent slice (works if tags appear as text)
                        #   2. element innerHTML (raw markup has literal tag strings)
                        #   3. reconstruct from JSON blob found in body text
                        print(
                            f"[worker] tool_call appears truncated in element after "
                            f"{incomplete_reset_count} resets — falling back to body.textContent"
                        )
                        import re as _re, html as _html
                        try:
                            full_body  = await page.evaluate("() => document.body.textContent")
                            body_slice = full_body[body_baseline:].strip()

                            # Stage 1: body text already has the closing tag
                            if "</tool_call>" in body_slice:
                                print(f"[worker] body fallback found </tool_call> ({len(body_slice)} chars)")
                                return body_slice

                            # Stage 2: try innerHTML — raw HTML preserves literal tag strings
                            print("[worker] body textContent missing </tool_call> — trying innerHTML")
                            try:
                                inner_html = await locator.evaluate("el => el.innerHTML")
                                if inner_html and "</tool_call>" in inner_html:
                                    clean = _re.sub(r'<[^>]+>', '', inner_html)
                                    clean = _html.unescape(clean).strip()
                                    if "<tool_call>" not in clean:
                                        clean = "<tool_call>" + clean
                                    if "</tool_call>" not in clean:
                                        clean = clean + "</tool_call>"
                                    print(f"[worker] innerHTML fallback succeeded ({len(clean)} chars)")
                                    return clean
                            except Exception as ie:
                                print(f"[worker] innerHTML fallback failed: {ie}")

                            # Stage 3: JSON blob is visible in body text — wrap it ourselves
                            json_match = _re.search(r'\{"tool"\s*:', body_slice)
                            if json_match:
                                json_blob = body_slice[json_match.start():]
                                # Trim trailing UI chrome after the closing }}
                                end_match = _re.search(r'\}\s*\}\s*$', json_blob)
                                if end_match:
                                    json_blob = json_blob[:end_match.end()]
                                reconstructed = f"<tool_call>{json_blob}</tool_call>"
                                print(f"[worker] Reconstructed tool_call from body JSON blob ({len(reconstructed)} chars)")
                                return reconstructed

                            print("[worker] All fallbacks exhausted — returning element text")
                        except Exception as e:
                            print(f"[worker] body fallback failed: {e}")
                        # Absolute last resort: truncated-tool_call parser in main.py
                        # will attempt JSON recovery from whatever we return.
                        return current_text
                    # Haven't hit the threshold yet — reset stable_count and keep waiting
                    print(
                        f"[worker] tool_call still incomplete after {elapsed:.1f}s, "
                        f"resetting stable counter (reset #{incomplete_reset_count})"
                    )
                    stable_count = 0
                else:
                    print(
                        f"[worker] Stable after {elapsed:.1f}s "
                        f"(needed={needed_stable}, stop_gone={stop_gone_long_enough})"
                    )
                    return current_text

        print("[worker] Timeout — returning partial response")
        return prev_text