import argparse
import inspect
import secrets
from typing import TypeVar, Callable, Awaitable, Union, overload, cast
from python.helpers import dotenv, rfc, settings, files
import secrets
import asyncio
import threading
import queue

T = TypeVar('T')
R = TypeVar('R')

parser = argparse.ArgumentParser()
args = {}
dockerman = None
runtime_id = None


def initialize():
    global args
    if args:
        return
    parser.add_argument("--port", type=int, default=None, help="Web UI port")
    parser.add_argument("--host", type=str, default=None, help="Web UI host")
    parser.add_argument(
        "--cloudflare_tunnel",
        type=bool,
        default=False,
        help="Use cloudflare tunnel for public URL",
    )
    parser.add_argument(
        "--development", type=bool, default=False, help="Development mode"
    )

    known, unknown = parser.parse_known_args()
    args = vars(known)
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            args[key] = value

def get_arg(name: str):
    global args
    return args.get(name, None)

def has_arg(name: str):
    global args
    return name in args

def is_dockerized() -> bool:
    return bool(get_arg("dockerized"))

def is_development() -> bool:
    return not is_dockerized()

def get_local_url():
    if is_dockerized():
        return "host.docker.internal"
    return "127.0.0.1"

def get_runtime_id() -> str:
    global runtime_id
    if not runtime_id:
        runtime_id = secrets.token_hex(8)   
    return runtime_id

def get_persistent_id() -> str:
    id = dotenv.get_dotenv_value("A0_PERSISTENT_RUNTIME_ID")
    if not id:
        id = secrets.token_hex(16)
        dotenv.save_dotenv_value("A0_PERSISTENT_RUNTIME_ID", id)
    return id

@overload
async def call_development_function(func: Callable[..., Awaitable[T]], *args, **kwargs) -> T: ...

@overload
async def call_development_function(func: Callable[..., T], *args, **kwargs) -> T: ...

async def call_development_function(func: Union[Callable[..., T], Callable[..., Awaitable[T]]], *args, **kwargs) -> T:
    """Call a function in the dev RFC target if available; otherwise fallback locally.

    In some environments (e.g., running the UI without a separate RFC server),
    the RFC endpoint might not be available. In that case, we gracefully fall
    back to executing the function locally so core features (like the File
    Browser) continue to work.
    """
    if is_development():
        try:
            url = _get_rfc_url()
            password = _get_rfc_password()
            module = files.deabsolute_path(func.__code__.co_filename).replace("/", ".").removesuffix(".py")  # __module__ is not reliable
            result = await rfc.call_rfc(
                url=url,
                password=password,
                module=module,
                function_name=func.__name__,
                args=list(args),
                kwargs=kwargs,
            )
            return cast(T, result)
        except Exception:
            # RFC target not reachable or failed — execute locally
            pass

    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)  # type: ignore


async def handle_rfc(rfc_call: rfc.RFCCall):
    return await rfc.handle_rfc(rfc_call=rfc_call, password=_get_rfc_password())


def get_rfc_password():
    return dotenv.get_dotenv_value(dotenv.KEY_RFC_PASSWORD)


def _get_rfc_password() -> str:
    password = dotenv.get_dotenv_value(dotenv.KEY_RFC_PASSWORD)
    if not password:
        # Auto-provision a dev RFC password to avoid crashes in development flows.
        # Persist to .env so both caller and handler use the same secret.
        password = secrets.token_hex(16)
        try:
            dotenv.save_dotenv_value(dotenv.KEY_RFC_PASSWORD, password)
        except Exception:
            pass
    return password


def _get_rfc_url() -> str:
    set = settings.get_settings()
    url = set["rfc_url"]
    if not "://" in url:
        url = "http://"+url
    if url.endswith("/"):
        url = url[:-1]
    url = url+":"+str(set["rfc_port_http"])
    url += "/rfc"
    return url


def call_development_function_sync(func: Union[Callable[..., T], Callable[..., Awaitable[T]]], *args, **kwargs) -> T:
    # run async function in sync manner
    result_queue = queue.Queue()
    
    def run_in_thread():
        result = asyncio.run(call_development_function(func, *args, **kwargs))
        result_queue.put(result)
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join(timeout=30)  # wait for thread with timeout
    
    if thread.is_alive():
        raise TimeoutError("Function call timed out after 30 seconds")
    
    result = result_queue.get_nowait()
    return cast(T, result)


def get_web_ui_port():
    # Priority: CLI arg --port -> env WEB_UI_PORT -> .env -> default
    import os
    env_port = 0
    try:
        env_port = int(os.environ.get("WEB_UI_PORT", "0"))
    except Exception:
        env_port = 0
    web_ui_port = (
        get_arg("port")
        or env_port
        or int(dotenv.get_dotenv_value("WEB_UI_PORT", 0))
        or 8080
    )
    return web_ui_port

def get_tunnel_api_port():
    tunnel_api_port = (
        get_arg("tunnel_api_port")
        or int(dotenv.get_dotenv_value("TUNNEL_API_PORT", 0))
        or 55520
    )
    return tunnel_api_port
