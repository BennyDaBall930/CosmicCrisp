from agent import LoopData
from python.helpers.extension import Extension
from python.extensions.before_main_llm_call._10_log_for_stream import build_heading


class InitializeReasoningLog(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # Don’t create a visible card here. We’ll create the card lazily
        # when actual reasoning/thoughts arrive from streaming. This avoids
        # empty “Reasoning” blocks with only a Finished:true kvp.
        if "reasoning_text" not in loop_data.params_temporary:
            loop_data.params_temporary["reasoning_text"] = ""
