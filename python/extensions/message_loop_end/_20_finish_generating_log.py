from python.helpers.extension import Extension
from agent import LoopData


class FinishGeneratingLog(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        try:
            # Do not add a visible "finished:true" kvp to agent logs.
            # The response tool uses this indicator for speech; agent logs do not.

            reasoning_log = loop_data.params_temporary.get("log_item_reasoning")
            reasoning_text = loop_data.params_temporary.get("reasoning_text", "")

            if reasoning_log:
                update_kwargs = {}
                if reasoning_text:
                    update_kwargs.update(
                        {
                            "content": reasoning_text,
                            "reasoning": reasoning_text,
                            "temp": False,
                        }
                    )
                reasoning_log.update(**update_kwargs)
            elif reasoning_text:
                fallback_log = self.agent.context.log.log(
                    type="agent",
                    heading=f"icon://network_intelligence {self.agent.agent_name}: Reasoning",
                    content=reasoning_text,
                )
                fallback_log.update(reasoning=reasoning_text, temp=False)
                loop_data.params_temporary["log_item_reasoning"] = fallback_log
        except Exception:
            # Never let UI bookkeeping break the loop
            pass
