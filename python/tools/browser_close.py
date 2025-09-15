from python.helpers.tool import Tool, Response
from python.helpers.playwright_simple import SimpleBrowser


class BrowserClose(Tool):
    async def execute(self, **kwargs):
        state_key = "_simple_browser_state"
        browser: SimpleBrowser | None = self.agent.get_data(state_key)
        if browser:
            await browser.close()
            self.agent.set_data(state_key, None)
            return Response(message="Browser session closed.", break_loop=False)
        return Response(message="No browser session to close.", break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"icon://captive_portal {self.agent.agent_name}: Close Browser",
            content="",
            kvps=self.args,
        )

