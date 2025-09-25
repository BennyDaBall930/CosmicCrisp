import pytest
from pydantic import ValidationError

from python.runtime.agent.task_parser import ToolCall


def test_invalid_tool():
    with pytest.raises(ValidationError):
        ToolCall(tool="invalid", args={})
