# Troubleshooting and FAQ
Apple Zero is an AI agent framework based on the open-source project Agent Zero. This page addresses frequently asked questions and provides troubleshooting steps for common issues encountered while using Apple Zero.

## Frequently Asked Questions
**1. How do I ask Apple Zero to work directly on my files or directories?**
- Place the files inside the project directory or use the file browser in the UI to navigate to them.

**2. When I input something in the chat, nothing happens. What's wrong?**
- Check if you have set up API keys in the Settings page. Without keys the application cannot access language models.

**3. How do I integrate open-source models with Apple Zero?**
- Refer to the [Installing and Using Ollama](installation.md#installing-and-using-ollama-local-models) section for configuring local models via Ollama or LM Studio.

**4. How can I make Apple Zero retain memory between sessions?**
- Use the backup tab in Settings to export your data before updating and restore it afterwards.

**5. Where can I find more documentation or tutorials?**
- Visit the project's GitHub repository for discussions and guides.

**6. My `code_execution` tool doesn't work, what's wrong?**
- Ensure the required system packages are installed and that `python` is available in your environment. Re-run `./setup.sh` if needed.

## Troubleshooting

**Installation**
- Re-run the setup script to ensure all dependencies are installed: `./setup.sh`.
- Verify your Python version is supported.

**Usage**
- Check the `logs/` directory for detailed error messages.
- Ensure API keys are valid and not rate-limited.

**Performance Issues**
- Large prompts or heavy tool usage may slow responses. Consider using faster models or reducing context size.

For further help consult the connectivity and development guides or open an issue on GitHub.
