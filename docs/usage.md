# Apple Zero Usage Guide

Apple Zero is an AI agent framework based on the open-source project Agent Zero. This guide explains the most common features of the web interface.

## Basic Operations
- **Restart Framework** – use the restart button in the top bar.
- **Action Buttons** – run selected tools such as web search or code execution.
- **File Attachments** – drag files into the chat to provide additional context.

## Tool Usage
Apple Zero automatically selects tools based on the conversation. You can also trigger tools explicitly using the UI buttons or by prompting the agent.

## Example of Tools Usage – Web Search and Code Execution
1. Ask a question that requires external knowledge.
2. Apple Zero invokes web search and presents the result.
3. For code-related tasks Apple Zero opens a terminal session and returns the output.

## Multi-Agent Cooperation
Apple Zero supports spawning sub‑agents for complex tasks. Conversations show each agent’s messages so you can follow the reasoning chain.

## Prompt Engineering
Prompt files can be customized under the `prompts` directory. Select alternative prompt sets in the Settings page.

## Voice Interface
When enabled in Settings Apple Zero can speak responses using the built‑in text‑to‑speech engine and transcribe your microphone input.

## Mathematical Expressions
Apple Zero renders LaTeX expressions in responses.

## File Browser
Use the file browser in the sidebar to navigate your project files. Changes are applied immediately.

## Backup & Restore
A backup tab in Settings lets you export or restore configuration and data files.

For connectivity options see the [Connectivity Guide](connectivity.md). If you encounter issues review the [Troubleshooting](troubleshooting.md) page.
