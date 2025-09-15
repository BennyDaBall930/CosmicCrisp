### browser:

composite convenience; if `url` is set, opens it; otherwise runs `steps` like browser_do

arguments:
- url: optional — open this URL
- steps: list of action objects (same schema as browser_do.actions)
- reset: true/false — close prior session first (default false)

usage (open URL):
{
  "tool_name": "browser",
  "tool_args": {
    "url": "https://example.com",
    "reset": "true"
  }
}

usage (actions):
{
  "tool_name": "browser",
  "tool_args": {
    "reset": "true",
    "steps": [
      {"do": "goto", "url": "https://google.com"},
      {"do": "type", "selector": "input[name='q']", "text": "example.com"},
      {"do": "press", "selector": "input[name='q']", "key": "Enter"}
    ]
  }
}

