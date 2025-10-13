from python.helpers.tool import Tool, Response


class ResponseTool(Tool):

    async def execute(self, **kwargs):
        return Response(message=self.args["text"] if "text" in self.args else self.args["message"], break_loop=True)

    async def before_execution(self, **kwargs):
        # Ensure a response log exists even if streaming extensions didn't create it.
        try:
            if self.loop_data and "log_item_response" not in self.loop_data.params_temporary:
                self.loop_data.params_temporary["log_item_response"] = (
                    self.agent.context.log.log(
                        type="response",
                        heading=f"icon://chat {self.agent.agent_name}: Responding",
                        content=self.args.get("text") or self.args.get("message", ""),
                    )
                )
        except Exception:
            # Non-fatal: logging should not block tool execution
            pass

    async def after_execution(self, response, **kwargs):
        # do not add anything to the history or output

        if self.loop_data and "log_item_response" in self.loop_data.params_temporary:
            log = self.loop_data.params_temporary["log_item_response"]
            try:
                # Update the final content if it wasn't streamed, and mark finished
                if not getattr(log, "content", None):
                    log.update(content=response.message, finished=True)
                else:
                    log.update(finished=True)  # mark the message as finished
            except Exception:
                # Make a best-effort attempt to close out the message
                try:
                    log.update(finished=True)
                except Exception:
                    pass
        else:
            # Fallback: if no response log exists at all, create and finish one now
            try:
                self.agent.context.log.log(
                    type="response",
                    heading=f"icon://chat {self.agent.agent_name}: Responding",
                    content=response.message,
                    finished=True,
                )
            except Exception:
                pass
