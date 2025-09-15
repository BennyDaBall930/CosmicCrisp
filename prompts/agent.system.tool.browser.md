### browser_agent:

subordinate agent controls playwright browser
message argument talks to agent give clear instructions credentials task based
reset argument spawns new agent
do not reset if iterating
be precise descriptive like: open google login and end task, log in using ... and end task
when following up start: considering open pages
dont use phrase wait for instructions use end task
downloads default to `tmp/downloads` inside the project folder

optional args:
- start_url (alias: url): pre-navigate to this URL before the agent starts; helps avoid initial about:blank

usage:
```json
{
  "thoughts": ["I need to log in to..."],
  "headline": "Opening new browser session for login",
  "tool_name": "browser_agent",
  "tool_args": {
    "message": "Open and log me into...",
    "reset": "true",
    "start_url": "https://www.google.com"
  }
}
```

```json
{
  "thoughts": ["I need to log in to..."],
  "headline": "Continuing with existing browser session",
  "tool_name": "browser_agent",
  "tool_args": {
    "message": "Considering open pages, click...",
    "reset": "false"
  }
}
```
