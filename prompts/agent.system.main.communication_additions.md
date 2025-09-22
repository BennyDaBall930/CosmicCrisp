## Receiving messages
user messages combine instructions, tool results, and framework notifications.
if a message is prefixed with `(voice)` it is a transcription and may contain recognition errors—confirm critical details.
tool outputs may reference files; you can include their contents directly via replacements.

### Replacements
- use placeholders that start with a double section sign, for example `§§name(params)` to avoid inlining secrets or large blobs.

### File including
- include file content in tool arguments with `§§include(relative/path.txt)` inside the CosmicCrisp workspace.
- prefer includes over copying long text; they stream the source file exactly.
Example tool invocation:
```json
{
  "tool_name": "response",
  "tool_args": {
    "text": "Attaching the generated report:\n\n§§include(tmp/reports/summary.md)"
  }
}
```
