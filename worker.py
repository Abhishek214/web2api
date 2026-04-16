"""
Playwright worker — automates LLM web apps (Gemini, ChatGPT, Claude.ai).
"""

import asyncio
import json
import os
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from dotenv import load_dotenv

load_dotenv()

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
            "response":    "div[data-message-author-role='assistant']:last-of-type .markdown",
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
            # Selectors are tried in order; first match wins and is cached.
            # IMPORTANT: only list selectors that start at count=0 on a fresh
            # conversation page — generic classes like .prose or .whitespace-pre-wrap
            # already exist in UI chrome and cause false-positive matches.
            "response":    ".font-claude-message",
            "response_fallbacks": [
                '[data-testid="assistant-message"]',
                ".font-claude-message",
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
        Button clicks are unreliable due to SVG/overlay intercepts (as seen in logs).
        Each provider has a dedicated URL that always opens a blank conversation.
        """
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

        fill() bypasses React's event system on contenteditable divs, leaving
        the send button disabled.  Clipboard paste fires the native paste event
        which React and ProseMirror both handle correctly — and it's instant
        regardless of prompt length.
        """
        page     = self._page
        input_el = page.locator(SELECTORS["input"])

        await input_el.click()
        await asyncio.sleep(0.2)

        # Write to clipboard via JS, then paste
        await page.evaluate("(t) => navigator.clipboard.writeText(t)", prompt)

        # Detect OS for correct modifier key
        is_mac = "darwin" in os.uname().sysname.lower()
        modifier = "Meta" if is_mac else "Control"
        await page.keyboard.press(f"{modifier}+v")
        await asyncio.sleep(0.4)

        # Belt-and-suspenders: also dispatch an input event
        await input_el.dispatch_event("input")

    async def _click_send(self):
        """
        Click the send button; fall back to Enter key if it's still not clickable.
        This handles providers where the button appears enabled in the DOM but
        is partially covered (ChatGPT sidebar SVG issue in logs).
        """
        page     = self._page
        send_sel = SELECTORS["send_button"]

        try:
            # Wait for button to be truly enabled (not aria-disabled)
            await page.wait_for_function(
                """(sel) => {
                    const btn = document.querySelector(sel);
                    return btn && !btn.disabled &&
                           btn.getAttribute('aria-disabled') !== 'true';
                }""",
                arg=send_sel,
                timeout=8_000,
            )
            # Use JS click to bypass any overlay/intercept issues
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

        Count == 0 is EXPECTED on a fresh page (pre-send baseline) — do NOT
        treat it as a broken selector here.  Fallback discovery happens in
        Phase 1 of _do_query, where count still being 0 *after* the response
        has started genuinely means the selector is broken.
        """
        page    = self._page
        primary = SELECTORS["response"]
        loc     = page.locator(primary)
        count   = await loc.count()
        return loc, count

    async def _do_query(self, prompt: str) -> str:
        page = self._page
        await self._reset_to_new_chat()

        # Snapshot page text length BEFORE sending — used by JS fallback
        snapshot_len = await page.evaluate("() => document.body.innerText.length")

        response_locator, count_before = await self._resolve_response_locator()
        print(f"[worker] Sending prompt (baseline={count_before}, text_len={snapshot_len}, sel={SELECTORS['response']!r})…")

        await self._fill_input(prompt)
        await self._click_send()

        # ── Phase 1: wait for response to START appearing ─────────────────────
        # Strategy A: CSS locator count increases
        # Strategy B (always active): page body text grows meaningfully
        response_started = False
        elapsed = 0.0

        while elapsed < RESPONSE_START_TIMEOUT / 1000:
            await asyncio.sleep(0.5)
            elapsed += 0.5

            # Check CSS locator (if we have one)
            if response_locator is not None:
                cur = await response_locator.count()
                if cur > count_before:
                    print(f"[worker] CSS locator matched new block ({cur} elements) after {elapsed:.1f}s")
                    response_started = True
                    break
                # Try fallbacks if primary still returns 0.
                # CRITICAL: set count_before to the fallback's REAL current count,
                # not 0 — otherwise UI-chrome elements trigger a false "response started".
                if cur == 0:
                    for fb in SELECTORS.get("response_fallbacks", []):
                        fb_loc = page.locator(fb)
                        fb_count = await fb_loc.count()
                        if fb_count >= 0:   # try every fallback, record real baseline
                            print(f"[worker] Trying fallback selector: {fb!r} (baseline={fb_count})")
                            SELECTORS["response"] = fb
                            response_locator = fb_loc
                            count_before = fb_count   # wait for count to GROW from here
                            break

            # Strategy B: body text grew — response is arriving regardless of selectors
            cur_len = await page.evaluate("() => document.body.innerText.length")
            if cur_len > snapshot_len + 50:
                print(f"[worker] Body text grew {snapshot_len}→{cur_len} after {elapsed:.1f}s — response started")
                response_started = True
                break

        if not response_started:
            tried = [SELECTORS["response"]] + SELECTORS.get("response_fallbacks", [])
            raise TimeoutError(
                f"No response from {TARGET_PROVIDER} after {elapsed:.0f}s. "
                f"Tried CSS selectors: {tried}."
            )

        # ── Phase 2: wait for response to FINISH, then return text ────────────
        if response_locator is not None and await response_locator.count() > count_before:
            return (await self._wait_for_stable_response(response_locator.last)).strip()

        # CSS count didn't grow past baseline — try remaining fallback selectors
        # now that we know a response is genuinely in progress.
        for fb in SELECTORS.get("response_fallbacks", []):
            if fb == SELECTORS.get("response"):
                continue   # already tried
            fb_loc = self._page.locator(fb)
            fb_baseline = await fb_loc.count()
            # Wait briefly for a new element to appear
            for _ in range(6):
                await asyncio.sleep(0.5)
                if await fb_loc.count() > fb_baseline:
                    print(f"[worker] Phase-2 fallback matched: {fb!r}")
                    SELECTORS["response"] = fb
                    return (await self._wait_for_stable_response(fb_loc.last)).strip()

        # Last resort: body-text slice
        print("[worker] WARNING: All CSS selectors failed — using JS page-text fallback")
        return (await self._wait_for_stable_text(snapshot_len)).strip()

    async def _wait_for_stable_text(self, snapshot_len: int) -> str:
        """
        CSS-selector-free response extraction.
        Polls document.body.innerText until it stops growing, then strips
        everything up to the snapshot length to return only the new text.
        Used when no CSS selector can locate the assistant response element.
        """
        page         = self._page
        prev_len     = snapshot_len
        stable_count = 0
        elapsed      = 0.0

        while elapsed < RESPONSE_MAX_WAIT:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL / 1000)
            elapsed += RESPONSE_POLL_INTERVAL / 1000

            stop_btn      = page.locator(SELECTORS["stop_button"])
            is_generating = await stop_btn.is_visible()

            cur_len = await page.evaluate("() => document.body.innerText.length")

            if cur_len == prev_len:
                stable_count += 1
            else:
                stable_count = 0
                prev_len     = cur_len

            if stable_count >= 4 and not is_generating:
                print(f"[worker] Body text stable after {elapsed:.1f}s ({cur_len} chars)")
                full_text = await page.evaluate("() => document.body.innerText")
                # Return only the portion that appeared after we sent the prompt
                return full_text[snapshot_len:].strip()

        print("[worker] Timeout in _wait_for_stable_text — returning partial")
        full_text = await page.evaluate("() => document.body.innerText")
        return full_text[snapshot_len:].strip()


    async def _wait_for_stable_response(self, locator) -> str:
        page         = self._page
        prev_text    = ""
        stable_count = 0
        elapsed      = 0.0

        while elapsed < RESPONSE_MAX_WAIT:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL / 1000)
            elapsed += RESPONSE_POLL_INTERVAL / 1000

            stop_btn      = page.locator(SELECTORS["stop_button"])
            is_generating = await stop_btn.is_visible()

            try:
                current_text = await locator.inner_text()
            except Exception:
                current_text = prev_text

            if current_text and current_text == prev_text:
                stable_count += 1
            else:
                stable_count = 0
                prev_text    = current_text

            if stable_count >= 4 and not is_generating:
                print(f"[worker] Stable after {elapsed:.1f}s")
                return current_text

        print("[worker] Timeout — returning partial response")
        return prev_text