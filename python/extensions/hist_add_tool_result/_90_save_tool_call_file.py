import os
from typing import Any

from python.helpers import files, persist_chat
from python.helpers.extension import Extension

LEN_MIN = 500


class SaveToolCallFile(Extension):
    async def execute(self, data: dict[str, Any] | None = None, **kwargs):
        if not data:
            return

        result = data.get("tool_result") if isinstance(data, dict) else None
        if result is None:
            return

        if len(str(result)) < LEN_MIN:
            return

        msgs_folder = persist_chat.get_chat_msg_files_folder(self.agent.context.id)
        os.makedirs(msgs_folder, exist_ok=True)

        next_index = len(os.listdir(msgs_folder)) + 1
        new_file = files.get_abs_path(msgs_folder, f"{next_index}.txt")
        files.write_file(new_file, result)
        data["file"] = new_file
