### code_execution_tool

execute terminal commands python nodejs code for computation or software tasks
place code in "code" arg; escape carefully and indent properly
select "runtime" arg: "terminal" "python" "nodejs" "output" "reset"
select "session" number, 0 default, others for multitasking
if code runs long, use "output" to wait, "reset" to kill process
use "pip" "npm" "brew" in "terminal" to install packages
to output, use print() or console.log()
if tool outputs error, adjust code before retrying; 
important: check code for placeholders or demo data; replace with real variables; don't reuse snippets
don't use with other tools except thoughts; wait for response before using others
check dependencies before running code
never run 'sudo' or destructive commands without explicit user confirmation; request permission first
output may end with [SYSTEM: ...] information coming from framework, not terminal
working directory is the HoneyCrisp project root; prefer relative paths like `tmp/` or project subfolders
terminal shell is Bash; GNU bash features are available

path handling across runtimes (critical):
- terminal runtime: environment variables like `$HOME` are expanded by the shell; use `$HOME` in quoted paths and always quote paths with spaces
- python runtime: `$HOME`/`~` inside string literals are NOT expanded; use `pathlib.Path.home()` or `os.path.expanduser('~')` and join paths via `Path`/`os.path.join`
- when using environment variables in Python, call `os.environ['HOME']` or `os.path.expandvars()` explicitly
- after any filesystem change outside the project (e.g., creating files on Desktop), verify existence before reporting success
usage:

1 execute python code

~~~json
{
    "thoughts": [
        "Need to do...",
        "I can use...",
        "Then I can...",
    ],
    "headline": "Executing Python code to check current directory",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "python",
        "session": 0,
        "code": "import os\nprint(os.getcwd())",
    }
}
~~~

2 execute terminal command
~~~json
{
    "thoughts": [
        "Need to do...",
        "Need to install...",
    ],
    "headline": "Installing zip package via terminal",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "session": 0,
        "code": "brew install zip",
    }
}
~~~

3 bash tips for reliable scripts

- Use heredoc for multi-line or special characters: prefer `cat << 'EOF' > file` so content is literal (no expansion, history, or variable substitution).
- Avoid echo for long content: use `printf '%s'` or heredoc; `echo` may mangle strings and is not portable across shells.
- Prevent history expansion: if you must run commands containing `!`, either single-quote the string, use a quoted heredoc delimiter (`<< 'EOF'`), or disable histexpand for the session with `set +H`.
- Quote paths with spaces: wrap the whole path in quotes, e.g. `"$HOME/Desktop/TESTING HTML/index.html"` rather than escaping spaces.
- Prefer `$HOME` to `~` inside quoted strings and scripts (terminal runtime only). In Python, use `Path.home()`.
- For binary/complex payloads: write via the Python runtime or use `base64` encode/decode to avoid shell quoting issues.

examples:

1 write a multi-line html file safely

~~~json
{
    "thoughts": ["Create folder and write HTML with heredoc"],
    "headline": "Create HTML on Desktop",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "session": 0,
        "code": "mkdir -p \"$HOME/Desktop/TESTING HTML\" && cat << 'EOF' > \"$HOME/Desktop/TESTING HTML/index.html\"\n<!DOCTYPE html>\n<html><head><meta charset=\"UTF-8\"><title>Example</title></head><body>Hi!</body></html>\nEOF"
    }
}
~~~

2 temporarily disable history expansion (if needed)

~~~json
{
    "thoughts": ["Disable histexpand to allow ! in commands"],
    "headline": "Disable history expansion",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "session": 0,
        "code": "set +H && printf '%s\n' 'OK'"
    }
}
~~~

2.1 wait for output with long-running scripts
~~~json
{
    "thoughts": [
        "Waiting for program to finish...",
    ],
    "headline": "Waiting for long-running program to complete",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "output",
        "session": 0,
    }
}
~~~

2.2 reset terminal
~~~json
{
    "thoughts": [
        "code_execution_tool not responding...",
    ],
    "headline": "Resetting unresponsive terminal session",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "reset",
        "session": 0,
    }
}
~~~

 3 create a file on Desktop from Python (safe path handling + verification)

~~~json
{
    "thoughts": ["Use Path.home() and verify existence"],
    "headline": "Create Desktop file via Python",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "python",
        "session": 0,
        "code": "from pathlib import Path\nimport random\n\nbase = Path.home() / 'Desktop' / 'MUMBLES'\nbase.mkdir(parents=True, exist_ok=True)\nout = base / 'random_words.txt'\nwords = [str(i) for i in range(1000)]\nout.write_text(' '.join(words) + '.')\nprint(f'[OK] Created: {out.resolve()} exists={out.exists()}')"
    }
}
~~~

 4 verify filesystem results after actions (terminal)

~~~json
{
    "thoughts": ["Verify the file exists before reporting success"],
    "headline": "Verify output path",
    "tool_name": "code_execution_tool",
    "tool_args": {
        "runtime": "terminal",
        "session": 0,
        "code": "ls -l \"$HOME/Desktop/MUMBLES/random_words.txt\" || echo 'Not found'"
    }
}
~~~
