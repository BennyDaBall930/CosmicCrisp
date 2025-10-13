import math

from agent import LoopData
from python.helpers.extension import Extension
from python.extensions.before_main_llm_call._10_log_for_stream import build_heading


class LogFromStream(Extension):

    async def execute(
        self,
        loop_data: LoopData = LoopData(),
        text: str = "",
        **kwargs,
    ):
        pipes = "|" * math.ceil(math.sqrt(len(text)))
        heading = build_heading(self.agent, f"Reasoning... {pipes}")

        log_item = loop_data.params_temporary.get("log_item_reasoning")
        if not log_item:
            log_item = self.agent.context.log.log(
                type="agent",
                heading=heading,
                content="",
            )
            loop_data.params_temporary["log_item_reasoning"] = log_item

        log_item.update(
            heading=heading,
            content=text,
            temp=False,
            reasoning=text,
        )

        # Persist the latest reasoning snapshot for downstream hooks/fallbacks.
        loop_data.params_temporary["reasoning_text"] = text
