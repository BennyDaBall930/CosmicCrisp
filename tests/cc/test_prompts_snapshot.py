from python.runtime.agent import prompts


def test_prompts_library_exports_guardrails():
    assert isinstance(prompts.SAFETY_GUARDRAILS, str)
    assert 'legal' in prompts.SAFETY_GUARDRAILS.lower()


def test_persona_modifiers_have_strings():
    assert prompts.PERSONA_MODIFIERS
    assert all(isinstance(value, str) for value in prompts.PERSONA_MODIFIERS.values())
