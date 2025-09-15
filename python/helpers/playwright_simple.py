import asyncio
from typing import Optional
from pathlib import Path
from urllib.parse import urljoin
import base64
import os
import httpx

from playwright.async_api import async_playwright, Browser as PWBrowser, BrowserContext, Page

from python.helpers import files


class SimpleBrowser:
    def __init__(self, headless: bool = True):
        self._pw = None
        self.browser: Optional[PWBrowser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.headless = headless

    async def start(self):
        if self._pw is None:
            self._pw = await async_playwright().start()
        if self.browser is None:
            # Use CDP only if explicitly allowed
            allow_cdp = (os.environ.get("A0_ALLOW_CDP", "").lower() in ("1", "true", "yes"))
            cdp_url = os.environ.get("A0_CHROME_CDP_URL") or os.environ.get("CHROME_CDP_URL")
            if allow_cdp and cdp_url:
                self.browser = await self._pw.chromium.connect_over_cdp(cdp_url)
            else:
                executable = os.environ.get("A0_CHROME_EXECUTABLE") or os.environ.get("CHROME_PATH")
                launch_kwargs = {"headless": self.headless}
                if executable:
                    launch_kwargs["executable_path"] = executable
                # Launch isolated headless Chromium
                self.browser = await self._pw.chromium.launch(**launch_kwargs)
        if self.context is None:
            # When connecting over CDP, Chrome may already have a default context
            if self.browser.contexts:
                self.context = self.browser.contexts[0]
            else:
                self.context = await self.browser.new_context()
        if self.page is None:
            self.page = await self.context.new_page()
            await self.page.set_viewport_size({"width": 1280, "height": 1200})
            # optional init script if exists
            init_js_path = files.get_abs_path("lib/browser/init_override.js")
            if files.exists("lib/browser/init_override.js"):
                try:
                    await self.page.add_init_script(path=init_js_path)
                except Exception:
                    pass

    async def goto(self, url: str, wait: str = "domcontentloaded"):
        await self.start()
        await self.page.goto(url, wait_until=wait, timeout=20000)  # type: ignore

    async def click(self, selector: str):
        await self.start()
        await self.page.click(selector, timeout=8000)

    async def type(self, selector: str, text: str):
        await self.start()
        await self.page.fill(selector, text, timeout=8000)

    async def press(self, selector: Optional[str], key: str):
        await self.start()
        if selector:
            await self.page.press(selector, key, timeout=8000)
        else:
            await self.page.keyboard.press(key)

    async def wait_for_selector(self, selector: str, timeout_ms: int = 8000):
        await self.start()
        await self.page.wait_for_selector(selector, timeout=timeout_ms)

    async def screenshot(self, path: str, full_page: bool = False):
        await self.start()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        await self.page.screenshot(path=path, full_page=full_page)

    async def get_attr(self, selector: str, attr: str, index: int | None = None) -> str | None:
        await self.start()
        loc = self.page.locator(selector)
        if index is not None:
            loc = loc.nth(index)
        try:
            return await loc.get_attribute(attr)
        except Exception:
            return None

    async def save_image_by_selector(self, selector: str, path: str, index: int | None = None):
        await self.start()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        loc = self.page.locator(selector)
        if index is not None:
            loc = loc.nth(index)

        # Try src/currentSrc
        src = None
        try:
            src = await loc.get_attribute("src")
        except Exception:
            src = None
        if not src:
            try:
                src = await loc.evaluate("el => el.currentSrc || el.src || ''")
            except Exception:
                src = None
        if not src:
            # fallback: screenshot of element
            await loc.screenshot(path=path)
            return path

        # Data URI
        if src.startswith("data:"):
            # data:[<mediatype>][;base64],<data>
            b64_marker = ";base64,"
            idx = src.find(b64_marker)
            data_part = src[idx + len(b64_marker):] if idx != -1 else src.split(",",1)[-1]
            with open(path, "wb") as f:
                f.write(base64.b64decode(data_part))
            return path

        # Resolve relative/ protocol-relative
        page_url = self.page.url
        if src.startswith("//"):
            src = ("https:" if page_url.startswith("https") else "http:") + src
        elif not src.startswith("http"):
            src = urljoin(page_url, src)

        # Download via httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            resp = await client.get(src)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        return path

    async def close(self):
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
            self.context = None
            self.page = None
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None
