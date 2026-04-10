"""
Playwright worker — automates Gemini web app.
"""

import asyncio
import os

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
try:
    from playwright_stealth import stealth_async
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

TARGET_URL = os.getenv("TARGET_URL", "https://gemini.google.com")

LOGIN_EMAIL    = os.getenv("LOGIN_EMAIL", "")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "")

SELECTORS = {
    "input":        "rich-textarea .ql-editor",
    "send_button":  "button.send-button",
    "response":     "model-response:last-of-type .markdown",
    "stop_button":  "button[aria-label='Stop response']",
    "new_chat":     "a[href='/'][aria-label='New chat'], button[aria-label='New chat']",
    "login_email":    "input[type='email']",
    "login_password": "input[type='password']",
    "login_submit":   "button[type='submit']",
}

RESPONSE_START_TIMEOUT = 15_000
RESPONSE_POLL_INTERVAL = 500
RESPONSE_MAX_WAIT      = 120
MAX_RETRIES            = 2


class PlaywrightWorker:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.ready = False
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
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
                f"Run Chrome with: --remote-debugging-port={debug_port}\n"
                f"Error: {e}"
            )

        gemini_page = None
        for page in self._context.pages:
            if "gemini.google.com" in page.url:
                gemini_page = page
                break

        if gemini_page:
            self._page = gemini_page
            await self._page.bring_to_front()
        else:
            self._page = await self._context.new_page()
            await self._page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)

        try:
            await self._page.wait_for_selector(SELECTORS["input"], state="visible", timeout=30_000)
        except Exception:
            print("[worker] Warning: input box not found")

        self.ready = True
        print("[worker] Ready")

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
                        print("[worker] Reloading and retrying...")
                        try:
                            await self._page.reload(wait_until="domcontentloaded", timeout=30_000)
                            await asyncio.sleep(2)
                        except Exception:
                            pass
                    else:
                        raise

    async def _reset_to_new_chat(self):
        page = self._page
        
        # Navigate to the base URL
        await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30_000)
        
        # Explicitly click the 'New chat' button to clear context
        try:
            # Give the UI a moment to load the sidebar
            await asyncio.sleep(1) 
            new_chat_btn = page.locator(SELECTORS["new_chat"]).first
            if await new_chat_btn.is_visible():
                await new_chat_btn.click()
                await asyncio.sleep(1)
        except Exception as e:
            print(f"[worker] Note: Could not click new chat button: {e}")

        # Wait for the input box to be visible and ready
        await page.wait_for_selector(SELECTORS["input"], state="visible", timeout=15_000)


    async def _do_query(self, prompt: str) -> str:
        page = self._page

        await self._reset_to_new_chat()

        input_el = page.locator(SELECTORS["input"])
        await input_el.wait_for(state="visible", timeout=10_000)
        
        # 1. Instantly fill the prompt
        await input_el.fill(prompt)

        # 2. NOW wait for the send button to become enabled (since it has text)
        try:
            await page.wait_for_function(
                """() => {
                    const btn = document.querySelector('button.send-button');
                    return btn && !btn.disabled && btn.getAttribute('aria-disabled') !== 'true';
                }""",
                timeout=5_000,
            )
        except Exception:
            print("[worker] Send button not enabled after typing, trying anyway...")

        response_locator = page.locator(SELECTORS["response"])
        count_before = await response_locator.count()

        send_btn = page.locator(SELECTORS["send_button"])
        await send_btn.click()
        print(f"[worker] Sent — baseline response count: {count_before}")

        new_response_locator = None
        elapsed = 0.0
        while elapsed < RESPONSE_START_TIMEOUT / 1000:
            await asyncio.sleep(0.5)
            elapsed += 0.5
            if await response_locator.count() > count_before:
                new_response_locator = response_locator.last
                print("[worker] New response block appeared")
                break

        if new_response_locator is None:
            raise TimeoutError("No response appeared — Gemini is stuck")

        return (await self._wait_for_stable_response(new_response_locator)).strip()


    async def _wait_for_stable_response(self, locator) -> str:
        page = self._page
        prev_text = ""
        stable_count = 0
        elapsed = 0.0

        while elapsed < RESPONSE_MAX_WAIT:
            await asyncio.sleep(RESPONSE_POLL_INTERVAL / 1000)
            elapsed += RESPONSE_POLL_INTERVAL / 1000
            
            # Check if Gemini's UI indicates it is still generating
            stop_btn = page.locator(SELECTORS["stop_button"])
            is_generating = await stop_btn.is_visible()

            try:
                current_text = await locator.inner_text()
            except Exception:
                current_text = prev_text

            if current_text and current_text == prev_text:
                stable_count += 1
            else:
                stable_count = 0
                
            prev_text = current_text

            # Done if text is stable for 2 seconds (4 ticks) AND the Stop button is gone
            if stable_count >= 4 and not is_generating:
                print(f"[worker] Stable and generation complete after {elapsed:.1f}s")
                return current_text

        print("[worker] Timed out — returning partial response")
        return prev_text

def _auth_state_exists() -> bool:
    return os.path.exists("auth_state.json")