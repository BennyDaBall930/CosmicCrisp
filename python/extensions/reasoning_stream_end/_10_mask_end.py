from python.helpers.extension import Extension


class MaskReasoningStreamEnd(Extension):
    async def execute(self, **kwargs):
        agent = kwargs.get("agent")
        if not agent:
            return

        loop_data = kwargs.get("loop_data")

        try:
            filter_key = "_reason_stream_filter"
            filter_instance = agent.get_data(filter_key)
            if filter_instance:
                tail = filter_instance.finalize()
                if tail:
                    from python.helpers.print_style import PrintStyle

                    PrintStyle().stream(tail)

                    if loop_data:
                        try:
                            combined = (loop_data.params_temporary.get("reasoning_text") or "") + tail
                            loop_data.params_temporary["reasoning_text"] = combined
                            log_item = loop_data.params_temporary.get("log_item_reasoning")
                            if log_item:
                                log_item.update(content=combined, reasoning=combined, temp=False)
                        except Exception:
                            pass
                agent.set_data(filter_key, None)
        except Exception:
            pass
