import uuid
from typing import Any
from python.helpers.tool import Tool, Response
from python.helpers import files
from python.helpers.dirty_json import DirtyJson
from python.helpers.playwright_simple import SimpleBrowser


class BrowserDo(Tool):
    async def execute(self, actions: Any = None, reset: str = "false", screenshot: str = "true", **kwargs):
        # actions may come as a JSON string or list
        if isinstance(actions, str):
            try:
                actions = DirtyJson.parse_string(actions)
            except Exception:
                actions = []
        if not isinstance(actions, list):
            actions = []

        should_reset = str(reset).lower().strip() == "true"
        take_shot = str(screenshot).lower().strip() != "false"

        state_key = "_simple_browser_state"
        browser: SimpleBrowser | None = self.agent.get_data(state_key)
        if browser and should_reset:
            await browser.close()
            browser = None
        if not browser:
            browser = SimpleBrowser(headless=True)
            self.agent.set_data(state_key, browser)

        # run actions
        for step in actions:
            if not isinstance(step, dict):
                continue
            op = str(step.get("do") or step.get("action") or step.get("op") or "").lower()
            index = step.get("index")
            index = int(index) if isinstance(index, (int, float, str)) and str(index).strip().isdigit() else None
            if op in ("goto", "open", "navigate"):
                url = step.get("url") or step.get("href") or ""
                if url:
                    await browser.goto(str(url))
            elif op == "click":
                sel = step.get("selector") or step.get("sel") or step.get("css")
                if sel:
                    if index is not None:
                        # Click nth via evaluating locator
                        await browser.wait_for_selector(str(sel))
                        await browser.page.locator(str(sel)).nth(index).click()
                    else:
                        await browser.click(str(sel))
            elif op in ("type", "fill"):
                sel = step.get("selector") or step.get("sel") or step.get("css")
                text = step.get("text") or step.get("value") or ""
                if sel is not None:
                    await browser.type(str(sel), str(text))
            elif op == "press":
                key = step.get("key") or step.get("keys") or "Enter"
                sel = step.get("selector") or None
                await browser.press(str(sel) if sel else None, str(key))
            elif op in ("wait", "wait_for_selector"):
                if "selector" in step:
                    await browser.wait_for_selector(str(step["selector"]))
                else:
                    # simple sleep
                    import asyncio
                    await asyncio.sleep(float(step.get("seconds", 1)))
            elif op in ("save_image", "download_image", "saveimg"):
                sel = step.get("selector") or step.get("sel") or step.get("css")
                path = step.get("path") or step.get("to") or step.get("file")
                if sel and path:
                    await browser.save_image_by_selector(str(sel), str(path), index=index)

        shot_path = ""
        if take_shot:
            shot_name = f"{uuid.uuid4()}.png"
            shot_path = files.get_abs_path(
                "tmp/chats",
                self.agent.context.id,
                "browser",
                "screenshots",
                shot_name,
            )
            await browser.screenshot(shot_path, full_page=False)

        msg = "Ran browser actions successfully."
        if shot_path:
            msg += f"\n\nScreenshot: {shot_path}"
        return Response(message=msg, break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"icon://captive_portal {self.agent.agent_name}: Browser Actions",
            content="",
            kvps=self.args,
        )
