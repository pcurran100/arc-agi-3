"""v3 prompts: contents, schemas, tolerant parsing."""

from __future__ import annotations

from arc3.v3.prompts import (
    ASSESS_SCHEMA,
    REPAIR_SCHEMA,
    Context,
    assess_messages,
    parse_json,
    repair_messages,
)


def ctx(**kw):
    base = dict(level=0, actions_on_level=12, human_baseline=22, available=["ACTION1", "ACTION2"],
                facts="OBJECTS now (1): #3 colour 3 ...", grid="y00 ....")
    base.update(kw)
    return Context(**base)


def test_assess_includes_facts_evidence_and_refuted_goals():
    m = assess_messages(ctx(predict_code="def predict(s, a):\n    return s", check="3/5 exact",
                            counterexamples=["t2 ACTION1: #3 predicted x1 -> actually x5"],
                            refuted_goals=["reach the red square"]))
    user = m[1]["content"]
    for needle in ("a human needed about 22", "FACTS", "3/5 exact", "t2 ACTION1",
                   "GOALS ALREADY REFUTED", "reach the red square", "TASK"):
        assert needle in user
    assert "predict(state, action)" in m[0]["content"]


def test_repair_is_narrow():
    user = repair_messages(ctx(predict_code="def predict(s, a): return s", check="0/2",
                               counterexamples=["t0 ACTION1: x"]))[1]["content"]
    assert "Fix predict()" in user and "shortest path" not in user


def test_schemas_require_the_core_fields():
    assert {"predict_code", "goal_code", "next", "hypotheses"} <= set(ASSESS_SCHEMA["required"])
    assert "predict_code" in REPAIR_SCHEMA["required"]


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('Sure! Here it is: {"a": 3} hope that helps') == {"a": 3}
    assert parse_json("no json") is None
