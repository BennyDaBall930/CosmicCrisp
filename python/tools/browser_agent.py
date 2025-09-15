import asyncio
import time
import os
from typing import Optional
from agent import Agent, InterventionException
from pathlib import Path


import models
from python.helpers.tool import Tool, Response
from python.helpers import files, defer, persist_chat, strings
from python.helpers.browser_use import browser_use
from python.helpers.print_style import PrintStyle
from python.helpers.playwright import ensure_playwright_binary, get_playwright_cache_dir
from python.extensions.message_loop_start._10_iteration_no import get_iter_no
from pydantic import BaseModel
import uuid
from python.helpers.dirty_json import DirtyJson


class State:
    @staticmethod
    async def create(agent: Agent):
        state = State(agent)
        return state

    def __init__(self, agent: Agent):
        self.agent = agent
        self.browser_session: Optional[browser_use.BrowserSession] = None
        self.task: Optional[defer.DeferredTask] = None
        self.use_agent: Optional[browser_use.Agent] = None
        self.iter_no = 0
        self.start_url: Optional[str] = None

    def __del__(self):
        self.kill_task()

    async def _initialize(self):
        if self.browser_session:
            return

        # Ensure Playwright uses project-local browsers cache
        try:
            os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", get_playwright_cache_dir())
        except Exception:
            pass

        # Resolve user_data_dir under project tmp to avoid home permissions
        user_data_dir = (
            Path(
                os.environ.get(
                    "BROWSER_USE_CONFIG_DIR",
                    files.get_abs_path("tmp/browseruse"),
                )
            )
            / "profiles"
            / f"agent_{self.agent.context.id}"
        )
        try:
            os.makedirs(user_data_dir, exist_ok=True)
        except Exception as e:
            PrintStyle().warning(f"Could not create user_data_dir {user_data_dir}: {e}")

        # Only allow CDP if explicitly permitted
        allow_cdp = (os.environ.get("A0_ALLOW_CDP", "").lower() in ("1", "true", "yes"))
        a0_cdp = os.environ.get("A0_CHROME_CDP_URL") or os.environ.get("CHROME_CDP_URL")
        # Build launch args (no remote debugging to avoid interfering with user Chrome)
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ]

        self.browser_session = browser_use.BrowserSession(
            browser_profile=browser_use.BrowserProfile(
                headless=False if (allow_cdp and a0_cdp) else True,
                disable_security=True,
                chromium_sandbox=False,
                accept_downloads=True,
                downloads_dir=files.get_abs_path("tmp/downloads"),
                downloads_path=files.get_abs_path("tmp/downloads"),
                keep_alive=False,
                minimum_wait_page_load_time=1.0,
                wait_for_network_idle_page_load_time=2.0,
                maximum_wait_page_load_time=10.0,
                screen={"width": 1024, "height": 2048},
                viewport={"width": 1024, "height": 2048},
                # Prefer full Chromium that Playwright installs; add flags to reduce automation fingerprints
                args=(
                    (["--headless=new"] if not (allow_cdp and a0_cdp) else [])
                    + launch_args
                ),
                # Use a unique user data directory to avoid conflicts
                # Prefer project-local tmp dir to avoid home permission issues
                user_data_dir=str(user_data_dir),
            )
        )

        # Ensure a Chromium is available under the project cache
        try:
            ensure_playwright_binary()
        except Exception:
            pass

        await self.browser_session.start()

        # Optional pre-navigation to avoid initial about:blank
        if self.start_url:
            try:
                page = None
                # try to get current page from session
                try:
                    page = await self.browser_session.get_current_page()  # type: ignore
                except Exception:
                    page = None
                if page:
                    try:
                        await page.goto(self.start_url, wait_until="domcontentloaded", timeout=15000)
                        # Attempt to accept consent dialogs where present
                        try:
                            # try common consent buttons quickly; ignore failures
                            for sel in [
                                "button#L2AG",
                                "button#introAgreeButton",
                                "button:has-text('I agree')",
                                "button:has-text('Accept all')",
                                "form [type='submit']:has-text('I agree')",
                            ]:
                                try:
                                    await page.locator(sel).first.click(timeout=1500)
                                    break
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # ensure interactive
                        try:
                            await page.wait_for_load_state("networkidle", timeout=5000)
                        except Exception:
                            pass
                    except Exception:
                        # swallow navigation errors; the agent can still recover
                        pass
            except Exception as e:
                PrintStyle().warning(f"Pre-navigation failed for {self.start_url}: {e}")
        # self.override_hooks()

        # Add init script to the browser session
        if self.browser_session.browser_context:
            js_override = files.get_abs_path("lib/browser/init_override.js")
            await self.browser_session.browser_context.add_init_script(path=js_override)

    def start_task(self, task: str, start_url: Optional[str] = None):
        if self.task and self.task.is_alive():
            self.kill_task()

        self.task = defer.DeferredTask(
            thread_name="BrowserAgent" + self.agent.context.id
        )
        if self.agent.context.task:
            self.agent.context.task.add_child_task(self.task, terminate_thread=True)
        # record optional start_url for _initialize to use
        self.start_url = start_url
        self.task.start_task(self._run_task, task)
        return self.task

    def kill_task(self):
        if self.task:
            self.task.kill(terminate_thread=True)
            self.task = None
        if self.browser_session:
            try:
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.browser_session.close())
                except Exception:
                    pass
                # force kill leftover process if keep_alive prevented termination
                try:
                    if hasattr(self.browser_session, "kill"):
                        loop.run_until_complete(self.browser_session.kill())  # type: ignore
                except Exception:
                    pass
                loop.close()
            except Exception as e:
                PrintStyle().error(f"Error closing browser session: {e}")
            finally:
                self.browser_session = None
        self.use_agent = None
        self.iter_no = 0

    async def _run_task(self, task: str):
        await self._initialize()

        class DoneResult(BaseModel):
            # Make title optional with a safe default so LLMs that omit it
            # still validate against browser_use's DoneAction model.
            title: str = "Task Results"
            # Primary textual answer from the agent
            response: str
            # Page summary is helpful but not always provided
            page_summary: Optional[str] = None

        # Initialize controller
        controller = browser_use.Controller(output_model=DoneResult)

        # Register custom completion action with proper ActionResult fields
        @controller.registry.action("Complete task", param_model=DoneResult)
        async def complete_task(params: DoneResult):
            result = browser_use.ActionResult(
                is_done=True, success=True, extracted_content=params.model_dump_json()
            )
            return result

        # Build a browser_use native LLM wrapper to ensure correct tool/function calling
        def build_browser_use_llm():
            cfg = self.agent.config.browser_model
            provider = (cfg.provider or '').lower()
            name = cfg.name
            # normalize model id (strip provider prefixes like 'gemini/...')
            model_id = name.split('/')[-1] if name else ''
            from models import get_api_key
            try:
                if provider in ("google", "gemini"):
                    return browser_use.ChatGoogle(model=model_id or 'gemini-2.0-flash', api_key=get_api_key('google'))
                if provider in ("openai", "oai"):
                    return browser_use.ChatOpenAI(model=model_id or 'gpt-4o-mini', api_key=get_api_key('openai'))
                if provider in ("anthropic",):
                    return browser_use.ChatAnthropic(model=model_id or 'claude-3-5-sonnet-latest', api_key=get_api_key('anthropic'))
                if provider in ("groq",):
                    return browser_use.ChatGroq(model=model_id or 'llama-3.1-70b-versatile', api_key=get_api_key('groq'))
                if provider in ("azure", "azureopenai", "azure-openai"):
                    # Azure requires base_url and deployment config; fall back to OpenAI if not configured
                    return browser_use.ChatOpenAI(model=model_id or 'gpt-4o-mini', api_key=get_api_key('azure'))
                if provider in ("ollama",):
                    return browser_use.ChatOllama(model=model_id or 'llama3.1')
            except Exception as e:
                PrintStyle().warning(f"Failed to build native browser_use LLM for provider '{provider}': {e}")
            # Fallback to Google with env if unspecified
            return browser_use.ChatGoogle(model='gemini-2.0-flash', api_key=get_api_key('google'))

        model = build_browser_use_llm()

        try:
            self.use_agent = browser_use.Agent(
                task=task,
                browser_session=self.browser_session,
                llm=model,
                use_vision=self.agent.config.browser_model.vision,
                # avoid overriding system prompt to keep expected function schemas
                extend_system_message=None,
                controller=controller,
                enable_memory=False,  # Disable memory to avoid state conflicts
                # available_file_paths=[],
            )
        except Exception as e:
            raise Exception(
                f"Browser agent initialization failed. This might be due to model compatibility issues. Error: {e}"
            ) from e

        self.iter_no = get_iter_no(self.agent)

        async def hook(agent: browser_use.Agent):
            await self.agent.wait_if_paused()
            if self.iter_no != get_iter_no(self.agent):
                raise InterventionException("Task cancelled")

        # try:
        result = await self.use_agent.run(
            max_steps=50, on_step_start=hook, on_step_end=hook
        )
        return result
        # finally:
        #     # if self.browser_session:
        #     #     try:
        #     #         await self.browser_session.close()
        #     #     except Exception as e:
        #     #         PrintStyle().error(f"Error closing browser session in task cleanup: {e}")
        #     #     finally:
        #     #         self.browser_session = None
        #     pass

    # def override_hooks(self):
    #     def override_hook(func):
    #         async def wrapper(*args, **kwargs):
    #             await self.agent.wait_if_paused()
    #             if self.iter_no != get_iter_no(self.agent):
    #                 raise InterventionException("Task cancelled")
    #             return await func(*args, **kwargs)

    #         return wrapper

    #     if self.browser_session and hasattr(self.browser_session, "remove_highlights"):
    #         self.browser_session.remove_highlights = override_hook(
    #             self.browser_session.remove_highlights
    #         )

    async def get_page(self):
        if self.use_agent and self.browser_session:
            try:
                return await self.use_agent.browser_session.get_current_page()
            except Exception:
                # Browser session might be closed or invalid
                return None
        return None

    async def get_selector_map(self):
        """Get the selector map for the current page state."""
        if self.use_agent:
            await self.use_agent.browser_session.get_state_summary(
                cache_clickable_elements_hashes=True
            )
            return await self.use_agent.browser_session.get_selector_map()
        return {}


class BrowserAgent(Tool):

    async def execute(self, message="", reset="", start_url: str = "", **kwargs):
        self.guid = str(uuid.uuid4())
        reset = str(reset).lower().strip() == "true"
        await self.prepare_state(reset=reset)
        # normalize optional start_url
        start_url = str(start_url).strip()
        if not start_url:
            # also accept 'url' alias if provided
            start_url = str(kwargs.get("url", "")).strip()
        # try to extract URL from message if still empty (e.g., "go to google.com")
        if not start_url and isinstance(message, str):
            import re
            m = re.search(r"(https?://\S+)", message)
            if m:
                start_url = m.group(1)
            elif "google.com" in message.lower():
                # try to extract a search phrase and build a SERP URL to reduce friction
                q = ""
                mq = re.search(r"search\s+(?:for\s+)?'([^']+)'", message, flags=re.I)
                if not mq:
                    mq = re.search(r"search\s+(?:for\s+)?\"([^\"]+)\"", message, flags=re.I)
                if not mq:
                    mq = re.search(r"search\s+(?:for\s+)?([\w\s]+?)(?:[,\.;]|$)", message, flags=re.I)
                if mq:
                    q = mq.group(1).strip()
                if q:
                    from urllib.parse import quote_plus
                    start_url = (
                        "https://www.google.com/ncr?hl=en&gl=us&pws=0#safe=off&q=" + quote_plus(q)
                    )
                else:
                    start_url = "https://www.google.com/ncr?hl=en&gl=us&pws=0"
        task = self.state.start_task(message, start_url or None)

        # wait for browser agent to finish and update progress with timeout
        timeout_seconds = 300  # 5 minute timeout
        start_time = time.time()

        fail_counter = 0
        while not task.is_ready():
            # Check for timeout to prevent infinite waiting
            if time.time() - start_time > timeout_seconds:
                PrintStyle().warning(
                    f"Browser agent task timeout after {timeout_seconds} seconds, forcing completion"
                )
                break

            # Respect user stop/intervention immediately
            try:
                await self.agent.handle_intervention()
            except InterventionException:
                self.state.kill_task()
                return Response(
                    message="Browser agent stopped by user.",
                    break_loop=False,
                )
            await asyncio.sleep(1)
            try:
                if task.is_ready():  # otherwise get_update hangs
                    break
                try:
                    update = await asyncio.wait_for(self.get_update(), timeout=10)
                    fail_counter = 0  # reset on success
                except asyncio.TimeoutError:
                    fail_counter += 1
                    PrintStyle().warning(
                        f"browser_agent.get_update timed out ({fail_counter}/3)"
                    )
                    if fail_counter >= 3:
                        PrintStyle().warning(
                            "3 consecutive browser_agent.get_update timeouts, breaking loop"
                        )
                        break
                    continue
                log = update.get("log", get_use_agent_log(None))
                self.update_progress("\n".join(log))
                screenshot = update.get("screenshot", None)
                if screenshot:
                    self.log.update(screenshot=screenshot)
            except Exception as e:
                PrintStyle().error(f"Error getting update: {str(e)}")

        if not task.is_ready():
            PrintStyle().warning("browser_agent.get_update timed out, killing the task")
            self.state.kill_task()
            return Response(
                message="Browser agent task timed out, not output provided.",
                break_loop=False,
            )

        # final progress update
        if self.state.use_agent:
            log = get_use_agent_log(self.state.use_agent)
            self.update_progress("\n".join(log))

        # collect result with error handling
        try:
            result = await task.result()
        except Exception as e:
            PrintStyle().error(f"Error getting browser agent task result: {str(e)}")
            # Return a timeout response if task.result() fails
            answer_text = f"Browser agent task failed to return result: {str(e)}"
            self.log.update(answer=answer_text)
            return Response(message=answer_text, break_loop=False)
        # finally:
        #     # Stop any further browser access after task completion
        #     # self.state.kill_task()
        #     pass

        # Check if task completed successfully
        # Try to capture the latest extracted content (e.g., from extract_structured_data)
        extracted_saved_path = None
        try:
            if self.state.use_agent:
                def _is_extraction_block(text: str) -> bool:
                    t = text
                    return (
                        "Extracted content from" in t
                        or "Extracted Content:" in t
                        or ("\nPage Link:" in t and "\nQuery:" in t)
                        or "extracted_information" in t
                        or "```json" in t
                    )

                def _is_done_payload(text: str) -> bool:
                    t = text.strip()
                    # Avoid mistaking the final done payload (which is JSON with title/response/page_summary)
                    return (
                        t.startswith("{")
                        and '"title"' in t
                        and '"response"' in t
                        and '"page_summary"' in t
                        and '"extracted_information"' not in t
                    )

                latest_extract = None
                # Prefer the latest actual extraction (skip the final done payload)
                for item in reversed(self.state.use_agent.state.history.action_results()):
                    ec = (item.extracted_content or "").strip()
                    if not ec or _is_done_payload(ec):
                        continue
                    if _is_extraction_block(ec):
                        latest_extract = ec
                        break
                # As a fallback, search forward to catch an earlier extraction if needed
                if not latest_extract:
                    for item in self.state.use_agent.state.history.action_results():
                        ec = (item.extracted_content or "").strip()
                        if not ec or _is_done_payload(ec):
                            continue
                        if _is_extraction_block(ec):
                            latest_extract = ec
                            break
                if latest_extract:
                    # Persist into chat folder so users can open it reliably
                    extracted_saved_path = files.get_abs_path(
                        persist_chat.get_chat_folder_path(self.agent.context.id),
                        "browser",
                        "extracted_content_0.md",
                    )
                    files.make_dirs(extracted_saved_path)
                    try:
                        with open(extracted_saved_path, "w", encoding="utf-8") as f:
                            f.write(latest_extract)
                    except Exception:
                        extracted_saved_path = None
        except Exception:
            extracted_saved_path = None

        if result.is_done():
            answer = result.final_result()
            try:
                if answer and isinstance(answer, str) and answer.strip():
                    answer_data = DirtyJson.parse_string(answer)
                    answer_text = strings.dict_to_text(answer_data)  # type: ignore
                else:
                    answer_text = (
                        str(answer) if answer else "Task completed successfully"
                    )
            except Exception as e:
                answer_text = (
                    str(answer)
                    if answer
                    else f"Task completed with parse error: {str(e)}"
                )
        else:
            # Task hit max_steps without calling done(). Try to salvage any
            # structured extraction the agent may have produced, otherwise
            # return a clear, actionable message.
            fallback_text: Optional[str] = None
            try:
                if self.state.use_agent:
                    for item in reversed(self.state.use_agent.state.history.action_results()):
                        ec = (item.extracted_content or "").strip()
                        if not ec:
                            continue
                        # Prefer any JSON-like extraction blocks
                        if "extracted_information" in ec or ec.startswith("{") or "```json" in ec:
                            fallback_text = ec
                            break
            except Exception:
                fallback_text = None

            if fallback_text:
                # Save partial extract to file for convenience
                if not extracted_saved_path:
                    try:
                        extracted_saved_path = files.get_abs_path(
                            persist_chat.get_chat_folder_path(self.agent.context.id),
                            "browser",
                            "extracted_content_0.md",
                        )
                        files.make_dirs(extracted_saved_path)
                        with open(extracted_saved_path, "w", encoding="utf-8") as f:
                            f.write(fallback_text)
                    except Exception:
                        extracted_saved_path = None

                answer_text = "Partial results extracted before step limit was reached."
                if extracted_saved_path:
                    answer_text += f"\nSaved to: {extracted_saved_path}"
                else:
                    answer_text += "\n" + fallback_text
            else:
                urls = result.urls()
                current_url = urls[-1] if urls else "unknown"
                answer_text = (
                    f"Task reached step limit without completion. Last page: {current_url}. "
                    f"The browser agent may need clearer instructions on when to finish."
                )

        # update the log (without screenshot path here, user can click)
        self.log.update(answer=answer_text)

        # add screenshot to the answer if we have it
        if (
            self.log.kvps
            and "screenshot" in self.log.kvps
            and self.log.kvps["screenshot"]
        ):
            path = self.log.kvps["screenshot"].split("//", 1)[-1].split("&", 1)[0]
            answer_text += f"\n\nScreenshot: {path}"

        # Include extracted content path reference if saved
        if extracted_saved_path:
            answer_text += f"\n\nSaved extracted content: {extracted_saved_path}"

        # respond (with screenshot path)
        # ensure browser session is closed
        try:
            self.state.kill_task()
        except Exception:
            pass
        return Response(message=answer_text, break_loop=False)

    def get_log_object(self):
        return self.agent.context.log.log(
            type="browser",
            heading=f"icon://captive_portal {self.agent.agent_name}: Calling Browser Agent",
            content="",
            kvps=self.args,
        )

    async def get_update(self):
        await self.prepare_state()

        result = {}
        agent = self.agent
        ua = self.state.use_agent
        page = await self.state.get_page()

        if ua and page:
            try:

                async def _get_update():

                    # await agent.wait_if_paused() # no need here

                    log = []

                    # for message in ua.message_manager.get_messages():
                    #     if message.type == "system":
                    #         continue
                    #     if message.type == "ai":
                    #         try:
                    #             data = json.loads(message.content)  # type: ignore
                    #             cs = data.get("current_state")
                    #             if cs:
                    #                 log.append("AI:" + cs["memory"])
                    #                 log.append("AI:" + cs["next_goal"])
                    #         except Exception:
                    #             pass
                    #     if message.type == "human":
                    #         content = str(message.content).strip()
                    #         part = content.split("\n", 1)[0].split(",", 1)[0]
                    #         if part:
                    #             if len(part) > 150:
                    #                 part = part[:150] + "..."
                    #             log.append("FW:" + part)

                    # for hist in ua.state.history.history:
                    #     for res in hist.result:
                    #         log.append(res.extracted_content)
                    # log = ua.state.history.extracted_content()
                    # short_log = []
                    # for item in log:
                    #     first_line = str(item).split("\n", 1)[0][:200]
                    #     short_log.append(first_line)
                    result["log"] = get_use_agent_log(ua)

                    path = files.get_abs_path(
                        persist_chat.get_chat_folder_path(agent.context.id),
                        "browser",
                        "screenshots",
                        f"{self.guid}.png",
                    )
                    files.make_dirs(path)
                    await page.screenshot(path=path, full_page=False, timeout=3000)
                    result["screenshot"] = f"img://{path}&t={str(time.time())}"

                if self.state.task and not self.state.task.is_ready():
                    await self.state.task.execute_inside(_get_update)

            except Exception:
                pass

        return result

    async def prepare_state(self, reset=False):
        self.state = self.agent.get_data("_browser_agent_state")
        if reset and self.state:
            self.state.kill_task()
        if not self.state or reset:
            self.state = await State.create(self.agent)
        self.agent.set_data("_browser_agent_state", self.state)

    def update_progress(self, text):
        short = text.split("\n")[-1]
        if len(short) > 50:
            short = short[:50] + "..."
        progress = f"Browser: {short}"

        self.log.update(progress=text)
        self.agent.context.log.set_progress(progress)

    # def __del__(self):
    #     if self.state:
    #         self.state.kill_task()


def get_use_agent_log(use_agent: browser_use.Agent | None):
    result = ["🚦 Starting task"]
    if use_agent:
        action_results = use_agent.state.history.action_results()
        short_log = []
        for item in action_results:
            # final results
            if item.is_done:
                if item.success:
                    short_log.append(f"✅ Done")
                else:
                    short_log.append(
                        f"❌ Error: {item.error or item.extracted_content or 'Unknown error'}"
                    )

            # progress messages
            else:
                text = item.extracted_content
                if text:
                    first_line = text.split("\n", 1)[0][:200]
                    short_log.append(first_line)
        result.extend(short_log)
    return result
