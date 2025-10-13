# Streaming Notes

## Reasoning Normalization
- `models._parse_chunk` now inspects provider deltas for `reasoning`, `x_gpt_thinking`, `thinking`, `thoughts`, `internal_thoughts`, and the native `reasoning_content` field. Each value is normalized to text before being passed through `ChatGenerationResult`.
- When providers emit tag-style thoughts (for example `<think>...</think>`), `ChatGenerationResult` tracks partial opening/closing tags so that reasoning stays separate from the visible response even when tags appear across multiple chunks.
- Native reasoning streams that resend the full transcript per chunk are diffed against the accumulated buffer to avoid duplicates.

## Debugging
- Set `A0_DEBUG_REASONING=1` (or `true/yes/on`) before launching the runtime to log which provider keys contributed to each chunk. Logs are emitted through the `a0.reasoning` logger and only enabled while the flag is set.
