from __future__ import annotations

import json

from hypothesis import given, strategies as st

from python.helpers import dirty_json

primitive = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=40),
)

json_strategy = st.dictionaries(keys=st.text(min_size=1, max_size=20), values=primitive, max_size=5)


@given(json_strategy)
def test_round_trip_through_stringify(obj):
    serialised = dirty_json.stringify(obj)
    parsed = dirty_json.try_parse(serialised)
    assert parsed == obj


def test_try_parse_skips_json_comments():
    payload = '{// comment\n"foo": 1, /* multi */ "bar": "baz"}'
    parsed = dirty_json.try_parse(payload)
    assert parsed == {"foo": 1, "bar": "baz"}


@given(st.text(max_size=200))
def test_try_parse_never_raises_random_text(text: str):
    try:
        dirty_json.try_parse(text)
    except Exception as exc:  # pragma: no cover - guard for regression visibility
        raise AssertionError(f"dirty_json.try_parse raised {exc!r} for text={text!r}")
