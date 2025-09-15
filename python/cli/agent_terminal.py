import asyncio
import json
import os
import sys
import signal
import argparse
from typing import Optional

import aiohttp
from aiohttp import ClientSession, BasicAuth

try:
    from python.helpers import dotenv
except Exception:
    dotenv = None  # Optional; we can proceed without it


def _read_env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    if val:
        return val
    if dotenv:
        try:
            from python.helpers import dotenv as dz
            return dz.get_dotenv_value(key) or default
        except Exception:
            return default
    return default


async def fetch_csrf(session: ClientSession, base: str, auth: Optional[BasicAuth]) -> str:
    url = base.rstrip("/") + "/csrf_token"
    async with session.get(url, auth=auth) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["token"]


async def terminal_start(
    session: ClientSession,
    base: str,
    csrf: str,
    auth: Optional[BasicAuth],
    cwd: str,
    env: dict,
    rows: int,
    cols: int,
) -> str:
    url = base.rstrip("/") + "/terminal_start"
    payload = {"cwd": cwd, "env": env, "rows": rows, "cols": cols}
    headers = {"X-CSRF-Token": csrf}
    async with session.post(url, json=payload, headers=headers, auth=auth) as resp:
        resp.raise_for_status()
        data = await resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Failed to start terminal: {data}")
        return data["session_id"]


async def terminal_write(
    session: ClientSession, base: str, csrf: str, auth: Optional[BasicAuth], session_id: str, data: str
) -> dict:
    url = base.rstrip("/") + "/terminal_write"
    headers = {"X-CSRF-Token": csrf}
    payload = {"session_id": session_id, "data": data}
    async with session.post(url, json=payload, headers=headers, auth=auth) as resp:
        resp.raise_for_status()
        return await resp.json()


async def terminal_confirm(
    session: ClientSession, base: str, csrf: str, auth: Optional[BasicAuth], token: str
) -> dict:
    url = base.rstrip("/") + "/terminal_confirm"
    headers = {"X-CSRF-Token": csrf}
    payload = {"token": token}
    async with session.post(url, json=payload, headers=headers, auth=auth) as resp:
        resp.raise_for_status()
        return await resp.json()


async def stream_terminal(
    session: ClientSession, base: str, csrf: str, auth: Optional[BasicAuth], session_id: str
):
    url = base.rstrip("/") + "/terminal_stream"
    headers = {
        "X-CSRF-Token": csrf,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    payload = {"session_id": session_id}
    # Note: Endpoint expects POST and returns SSE stream
    async with session.post(url, json=payload, headers=headers, auth=auth) as resp:
        resp.raise_for_status()
        buffer = b""
        async for chunk in resp.content.iter_chunked(1024):
            if not chunk:
                continue
            buffer += chunk
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                for line in frame.split(b"\n"):
                    if line.startswith(b"data:"):
                        try:
                            data_str = line[len(b"data:"):].strip().decode("utf-8", errors="replace")
                            evt = json.loads(data_str)
                            t = evt.get("type")
                            d = evt.get("data", "")
                            if t == "stdout":
                                # Write without extra newline; data includes newlines
                                sys.stdout.write(d)
                                sys.stdout.flush()
                            elif t == "exit":
                                print("\n[session] terminal exited")
                                return
                            elif t == "error":
                                print(f"\n[error] {d}")
                            # ignore other event types
                        except Exception as e:
                            print(f"\n[parse-error] {e}")


async def read_input_and_write(
    session: ClientSession,
    base: str,
    csrf: str,
    auth: Optional[BasicAuth],
    session_id: str,
):
    loop = asyncio.get_running_loop()
    last_perm_token: Optional[str] = None

    def blocking_readline():
        return sys.stdin.readline()

    while True:
        line = await loop.run_in_executor(None, blocking_readline)
        if not line:
            await asyncio.sleep(0.01)
            continue

        # Handle approval shortcut: ":approve <token>" or just ":approve" to reuse last token
        if line.startswith(":approve"):
            parts = line.strip().split()
            token = parts[1] if len(parts) > 1 else last_perm_token
            if not token:
                print("[permission] No token to approve. Try again: :approve <token>")
                continue
            result = await terminal_confirm(session, base, csrf, auth, token)
            if result.get("success"):
                print("[permission] Approved.")
            else:
                print(f"[permission] Failed: {result.get('error')}")
            continue

        result = await terminal_write(session, base, csrf, auth, session_id, line)
        if not result.get("success"):
            print(f"[write-error] {result.get('error')}")
            continue
        if result.get("status") == "permission_required":
            last_perm_token = result.get("token")
            reason = result.get("reason")
            cmd = result.get("command")
            print(
                f"[permission] Command requires approval: '{cmd}'\n"
                f"           Reason: {reason}\n"
                f"           Approve with: :approve {last_perm_token}"
            )


async def amain():
    parser = argparse.ArgumentParser(description="Attach to Apple Zero terminal session")
    parser.add_argument("--base", type=str, default=None, help="Base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--cwd", type=str, default=None, help="Working directory for the session")
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=120)
    args = parser.parse_args()

    if dotenv:
        try:
            dotenv.load_dotenv()
        except Exception:
            pass

    base = args.base or f"http://127.0.0.1:{os.environ.get('WEB_UI_PORT','8080')}"
    cwd = args.cwd or os.getcwd()

    # BasicAuth if provided via env/.env
    user = _read_env("AUTH_LOGIN")
    pwd = _read_env("AUTH_PASSWORD")
    auth = BasicAuth(user, pwd) if user and pwd else None

    # Prepare venv env for PTY session
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    venv_dir = os.path.join(project_dir, "venv")
    venv_bin = os.path.join(venv_dir, "bin")
    env = {}
    if os.path.isdir(venv_dir):
        env["VIRTUAL_ENV"] = venv_dir
        env["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    env["TERM"] = os.environ.get("TERM", "xterm-256color")

    timeout = aiohttp.ClientTimeout(total=None, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        csrf = await fetch_csrf(session, base, auth)
        session_id = await terminal_start(session, base, csrf, auth, cwd=cwd, env=env, rows=args.rows, cols=args.cols)

        print(f"[session] Attached to {session_id}. Type commands. Approve with :approve <token> when prompted.")

        # Run stream and input tasks concurrently
        stream_task = asyncio.create_task(stream_terminal(session, base, csrf, auth, session_id))
        input_task = asyncio.create_task(read_input_and_write(session, base, csrf, auth, session_id))

        def _handle_sigint():
            for t in (stream_task, input_task):
                if not t.done():
                    t.cancel()

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        except NotImplementedError:
            pass

        done, pending = await asyncio.wait({stream_task, input_task}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

