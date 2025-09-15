### browser_do:

perform simple browser actions sequentially using Playwright (deterministic)

arguments:
- actions: list of steps; each step is an object with:
  - do: "goto"|"open"|"navigate"|"click"|"type"|"fill"|"press"|"wait"|"wait_for_selector"|"save_image"
  - selector: CSS selector (for click/type/press/wait_for_selector/save_image)
  - url/href: URL (for goto/open/navigate)
  - text/value: text to type (for type/fill)
  - key: keyboard key (for press)
  - seconds: number (for wait)
  - index: optional 0-based integer to target nth match (click/save_image)
  - path: filesystem path for save_image
- reset: true/false — close prior session first (default false)
- screenshot: true/false — attach screenshot (default true)

usage (Google search):
{
  "thoughts": ["Search Google for example.com"],
  "headline": "Perform Google search",
  "tool_name": "browser_do",
  "tool_args": {
    "reset": "true",
    "actions": [
      {"do": "goto", "url": "https://www.google.com"},
      {"do": "wait_for_selector", "selector": "input[name='q']"},
      {"do": "type", "selector": "input[name='q']", "text": "example.com"},
      {"do": "press", "selector": "input[name='q']", "key": "Enter"},
      {"do": "wait", "seconds": 2},
      {"do": "click", "selector": "a[href*='tbm=isch']"},
      {"do": "wait", "seconds": 2},
      {"do": "save_image", "selector": "img", "index": 2, "path": "tmp/downloads/example_3.jpg"}
    ]
  }
}
