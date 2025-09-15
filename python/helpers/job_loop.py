import asyncio
import time
from python.helpers.task_scheduler import TaskScheduler
from python.helpers.print_style import PrintStyle
from python.helpers import errors
from python.helpers import runtime, rfc, settings as settings_helper, dotenv


SLEEP_TIME = 60

keep_running = True
pause_time = 0
_pause_warned = False


async def run_loop():
    global pause_time, keep_running

    while True:
        # Only attempt RFC pause when explicitly configured via .env password
        # Avoid noisy errors when no RFC target is running by warning once.
        if runtime.is_development() and dotenv.get_dotenv_value(dotenv.KEY_RFC_PASSWORD):
            # Signal to RFC target (if available) to pause its loop.
            # Do not fall back locally — we don't want to pause our own loop.
            try:
                setts = settings_helper.get_settings()
                host = setts["rfc_url"]
                if "://" not in host:
                    host = "http://" + host
                if host.endswith("/"):
                    host = host[:-1]
                url = f"{host}:{setts['rfc_port_http']}/rfc"
                await rfc.call_rfc(
                    url=url,
                    password=runtime.get_rfc_password() or "",
                    module="python.helpers.job_loop",
                    function_name="pause_loop",
                    args=[],
                    kwargs={},
                )
            except Exception as e:
                global _pause_warned
                if not _pause_warned:
                    PrintStyle().warning(
                        "Could not reach RFC pause target (development). Suppressing further messages."
                    )
                    _pause_warned = True
        if not keep_running and (time.time() - pause_time) > (SLEEP_TIME * 2):
            resume_loop()
        if keep_running:
            try:
                await scheduler_tick()
            except Exception as e:
                PrintStyle().error(errors.format_error(e))
        await asyncio.sleep(SLEEP_TIME)  # TODO! - if we lower it under 1min, it can run a 5min job multiple times in it's target minute


async def scheduler_tick():
    # Get the task scheduler instance and print detailed debug info
    scheduler = TaskScheduler.get()
    # Run the scheduler tick
    await scheduler.tick()


def pause_loop():
    global keep_running, pause_time
    keep_running = False
    pause_time = time.time()


def resume_loop():
    global keep_running, pause_time
    keep_running = True
    pause_time = 0
