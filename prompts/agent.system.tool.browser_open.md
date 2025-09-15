### browser_open:

open a URL in a lightweight Playwright browser and end

arguments:
- url: absolute URL to open (required)
- reset: true/false — close prior session first (default false)
- screenshot: true/false — attach screenshot (default true)

usage:
{
  "thoughts": ["Open a page to capture the state"],
  "headline": "Open URL",
  "tool_name": "browser_open",
  "tool_args": {
    "url": "https://www.google.com",
    "reset": "true",
    "screenshot": "true"
  }
}

