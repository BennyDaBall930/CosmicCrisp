from typing import Any
from python.helpers.tool import Tool, Response
from .browser_open import BrowserOpen
from .browser_do import BrowserDo


class Browser(Tool):
    async def execute(self, steps: Any = None, url: str = "", reset: str = "false", **kwargs):
        # Composite convenience: if url provided and no steps -> open
        if url and not steps:
            opener = BrowserOpen(
                agent=self.agent,
                name="browser_open",
                method=None,
                args=self.args,
                message=self.message,
                loop_data=self.loop_data,
            )
            await opener.before_execution(url=url, reset=reset)
            resp = await opener.execute(url=url, reset=reset)
            await opener.after_execution(resp)
            return Response(message=resp.message, break_loop=False)

        # Otherwise treat steps as actions
        doer = BrowserDo(
            agent=self.agent,
            name="browser_do",
            method=None,
            args=self.args,
            message=self.message,
            loop_data=self.loop_data,
        )
        await doer.before_execution(actions=steps, reset=reset)
        resp = await doer.execute(actions=steps, reset=reset)
        await doer.after_execution(resp)
        return Response(message=resp.message, break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"icon://captive_portal {self.agent.agent_name}: Browser Composite",
            content="",
            kvps=self.args,
        )

