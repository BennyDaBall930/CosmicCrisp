# Architecture Overview

Apple Zero is an AI agent framework based on the open-source project Agent Zero. The macOS port runs natively without Docker and keeps the original modular design for extensibility.

## System Architecture
This diagram illustrates the hierarchical relationship between agents and their interaction with tools, extensions, instruments, prompts, memory and knowledge base.

![Apple Zero Architecture](res/arch-01.svg)

The user or Agent 0 is at the top of the hierarchy, delegating tasks to subordinate agents. Each agent can utilize tools and access shared assets to perform its tasks.

## Runtime Architecture
Apple Zero executes directly on macOS.

1. **Backend** – Python services providing agents, tools and APIs.
2. **Frontend** – A browser-based UI served by `run_ui.py` on port 8080.
3. **Data directories** – User data lives beside the project in folders such as `agents`, `memory`, `knowledge`, `instruments`, `prompts`, `logs` and `tmp`.

This native approach removes container dependencies while preserving a consistent environment across machines.

## Implementation Details

### Directory Structure
| Directory | Description |
| --- | --- |
| `/agents` | Specialized agents with their prompts and tools |
| `/docs` | Documentation files and guides |
| `/instruments` | Custom scripts and tools |
| `/knowledge` | Knowledge base storage |
| `/logs` | HTML CLI-style chat logs |
| `/memory` | Persistent agent memory |
| `/prompts` | System and tool prompts |
| `/python` | Core Python codebase |
| `/tmp` | Temporary runtime data |
| `/webui` | Web interface components |

### Key Files
| File | Description |
| --- | --- |
| `.env` | Environment configuration |
| `agent.py` | Core agent implementation |
| `initialize.py` | Framework initialization |
| `models.py` | Model providers and configs |
| `run_ui.py` | Web UI launcher |

## Core Components
Apple Zero's architecture revolves around the following key components:

### 1. Agents
Agents receive instructions, reason, make decisions, and utilize tools to achieve their objectives. Agents operate within a hierarchical structure, with superior agents delegating tasks to subordinate agents.

### 2. Tools
Tools include web search, code execution, API calls and more. Both built-in and custom tools are supported.

### 3. Memory System
Apple Zero includes both long-term and short-term memory for context retention and recall.

### 4. Prompts
Prompt files control agent behavior and communication. Custom prompt sets can be selected in the Settings page.

### 5. Knowledge
User-provided data that agents can leverage. Supported formats include `.txt`, `.pdf`, `.csv`, `.html`, `.json`, and `.md`.

### 6. Instruments
Scripts or functions that extend Apple Zero without increasing the system prompt size. Instruments live in `instruments/custom` and are detected automatically.

### 7. Extensions
Modular Python scripts that hook into the agent lifecycle. Extensions live under `python/extensions` and execute in alphabetical order.

> Consider contributing valuable custom components to the main repository. See [Contributing](contribution.md) for more information.
