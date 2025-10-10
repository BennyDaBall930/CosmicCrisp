from python.helpers.extension import Extension


class MaskReasoningStreamEnd(Extension):
    async def execute(self, **kwargs):
        agent = kwargs.get("agent")
        if not agent:
            return

        try:
            filter_key = "_reason_stream_filter"
            filter_instance = agent.get_data(filter_key)
            if filter_instance:
                tail = filter_instance.finalize()
                if tail:
                    from python.helpers.print_style import PrintStyle

                    PrintStyle().stream(tail)
                agent.set_data(filter_key, None)
        except Exception:
            pass
