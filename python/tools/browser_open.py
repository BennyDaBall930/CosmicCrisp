import uuid
from python.helpers.tool import Tool, Response
from python.helpers import files
from python.helpers.playwright_simple import SimpleBrowser


class BrowserOpen(Tool):
    async def execute(self, url: str = "", reset: str = "false", screenshot: str = "true", **kwargs):
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

        await browser.goto(url)

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

        msg = f"Opened URL: {url}"
        if shot_path:
            msg += f"\n\nScreenshot: {shot_path}"

        return Response(message=msg, break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"icon://captive_portal {self.agent.agent_name}: Open URL",
            content="",
            kvps=self.args,
        )

