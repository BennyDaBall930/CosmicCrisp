from python.helpers import persist_chat, tokens
from python.helpers.extension import Extension
from agent import LoopData
import asyncio
from python.helpers.log import LogItem
from python.helpers import log
import math
from python.extensions.before_main_llm_call._10_log_for_stream import build_heading, build_default_heading


class LogFromStream(Extension):

    async def execute(
        self,
        loop_data: LoopData = LoopData(),
        text: str = "",
        parsed: dict = {},
        **kwargs,
    ):

        heading = build_default_heading(self.agent)
        if "headline" in parsed:
            heading = build_heading(self.agent, parsed["headline"])
        elif "tool_name" in parsed:
            heading = build_heading(self.agent, f"Using tool {parsed['tool_name']}")
        elif "thoughts" in parsed:
            thoughts = parsed.get("thoughts") or []
            joined = "\n".join(str(item) for item in thoughts if item)
            pipes = "|" * math.ceil(math.sqrt(len(joined))) if joined else ""
            heading = build_heading(self.agent, f"Thinking... {pipes}")
        
        # create log message and store it in loop data temporary params
        if "log_item_generating" not in loop_data.params_temporary:
            loop_data.params_temporary["log_item_generating"] = (
                self.agent.context.log.log(
                    type="agent",
                    heading=heading,
                )
            )

        log_item = loop_data.params_temporary["log_item_generating"]

        # keep reasoning from previous logs in kvps
        kvps = {}
        if log_item.kvps is not None and "reasoning" in log_item.kvps:
            kvps["reasoning"] = log_item.kvps["reasoning"]
        kvps.update(parsed)

        # Build a readable content block so the UI shows thoughts even when JSON
        # rows are collapsed. Prefer the explicit thoughts list, then reasoning,
        # otherwise fall back to the headline/tool summary.
        display_sections: list[str] = []
        thoughts = parsed.get("thoughts")
        if isinstance(thoughts, list) and thoughts:
            bullet_lines = "\n".join(f"• {str(item)}" for item in thoughts if item)
            display_sections.append(bullet_lines)

        reasoning = parsed.get("reasoning")
        if isinstance(reasoning, list) and reasoning:
            bullet_lines = "\n".join(f"• {str(item)}" for item in reasoning if item)
            display_sections.append(bullet_lines)
        elif isinstance(reasoning, str) and reasoning.strip():
            display_sections.append(reasoning)

        headline = parsed.get("headline")
        if not display_sections and isinstance(headline, str) and headline.strip():
            display_sections.append(headline.strip())

        display_content = "\n\n".join(display_sections).strip()
        if not display_content:
            display_content = text or ""

        log_item.update(heading=heading, content=display_content, kvps=kvps, temp=False)

        # If the parsed payload contains thoughts/reasoning, mirror it into the
        # dedicated reasoning log so the UI always shows a reasoning panel even
        # when providers don’t emit a native reasoning stream.
        try:
            extracted = None
            if isinstance(parsed, dict):
                if "reasoning" in parsed and parsed["reasoning"]:
                    val = parsed["reasoning"]
                    if isinstance(val, list):
                        extracted = "\n".join(str(x) for x in val if x)
                    elif isinstance(val, dict):
                        extracted = "\n".join(str(v) for v in val.values() if v)
                    else:
                        extracted = str(val)
                elif "thoughts" in parsed and parsed["thoughts"]:
                    val = parsed["thoughts"]
                    if isinstance(val, list):
                        extracted = "\n".join(str(x) for x in val if x)
                    elif isinstance(val, dict):
                        # Preserve common high/low/local order if present
                        order = ["high_level", "low_level", "local"]
                        if any(k in val for k in order):
                            pieces = []
                            for k in order:
                                if k in val and val[k]:
                                    pieces.append(str(val[k]))
                            # add any remaining items
                            for k, v in val.items():
                                if k not in order and v:
                                    pieces.append(str(v))
                            extracted = "\n".join(pieces)
                        else:
                            extracted = "\n".join(str(v) for v in val.values() if v)
                    else:
                        extracted = str(val)

            if extracted:
                loop_data.params_temporary["reasoning_text"] = extracted
                rlog = loop_data.params_temporary.get("log_item_reasoning")
                if not rlog:
                    rlog = self.agent.context.log.log(
                        type="agent",
                        heading=build_heading(self.agent, "Reasoning"),
                        content=extracted,
                    )
                    loop_data.params_temporary["log_item_reasoning"] = rlog
                else:
                    rlog.update(content=extracted, reasoning=extracted, temp=False)
        except Exception:
            pass
