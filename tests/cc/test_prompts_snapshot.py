from cosmiccrisp.agent import prompts
from .golden_prompts import GOLDEN_PROMPTS


def test_prompts_snapshot():
    assert prompts.start_goal_prompt == GOLDEN_PROMPTS["start_goal_prompt"]
    assert prompts.analyze_task_prompt == GOLDEN_PROMPTS["analyze_task_prompt"]
    assert prompts.create_tasks_prompt == GOLDEN_PROMPTS["create_tasks_prompt"]
    assert prompts.summarize_prompt == GOLDEN_PROMPTS["summarize_prompt"]
