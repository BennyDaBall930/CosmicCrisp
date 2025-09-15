from python.helpers.api import ApiHandler, Request, Response
from python.helpers.file_browser import FileBrowser
from python.helpers import runtime

class GetWorkDirFiles(ApiHandler):

    @classmethod
    def get_methods(cls):
        return ["GET"]

    async def process(self, input: dict, request: Request) -> dict | Response:
        current_path = request.args.get("path", "")
        # Map the UI's placeholder to the project base directory
        if current_path == "$WORK_DIR":
            current_path = ""

        # browser = FileBrowser()
        # result = browser.get_files(current_path)
        result = await runtime.call_development_function(get_files, current_path)
        # Keep display label consistent for the root of the project
        if request.args.get("path", "") == "$WORK_DIR":
            result["current_path"] = "$WORK_DIR"

        return {"data": result}


async def get_files(path):
    # Support the special placeholder from the UI
    effective_path = "" if path in ("$WORK_DIR", "") else path
    browser = FileBrowser()
    result = browser.get_files(effective_path)
    # Keep UI label stable if user entered $WORK_DIR
    if path == "$WORK_DIR":
        result["current_path"] = "$WORK_DIR"
    return result
