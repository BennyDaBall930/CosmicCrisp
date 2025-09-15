"""Prompt templates for CosmicCrisp."""

start_goal_prompt = (
    "You are CosmicCrisp, an autonomous agent. Understand the user's goal and begin."
)

analyze_task_prompt = (
    "Analyze the goal, decide which tool to use, and provide a short rationale."
)

create_tasks_prompt = (
    "Given the latest result, propose the next actionable tasks."
)

summarize_prompt = (
    "Summarize progress and findings for the user, citing sources when possible."
)
