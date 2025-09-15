import pytest
from pydantic import ValidationError

from cosmiccrisp.agent.task_parser import ToolCall


def test_invalid_tool():
    with pytest.raises(ValidationError):
        ToolCall(tool="invalid", args={})
