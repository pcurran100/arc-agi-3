"""Dynamics induction: learning (s, a) -> s' from observed transitions.

The subsystem D-002 stakes the project on, and the one the dynamics experiment
is the go/no-go for. Kept separable from ontology, goal and planner (D-004),
because we expect to replace at least one of them wholesale and this is the
most likely candidate.

Read `rules.py` first: it states what a hypothesis in this system is allowed to
say, and its list of what the language *cannot* express is the thing to consult
before believing any result from here.
"""

from .induce import ObjectEvent, TransitionObservation, induce, observe_transition
from .predict import (
    Score,
    predict_frame,
    score_replay_baseline,
    score_static_baseline,
    score_transitions,
)
from .rules import ATTRIBUTES, Effect, Rule, RuleSet, Selector

__all__ = [
    "ATTRIBUTES",
    "Effect",
    "ObjectEvent",
    "Rule",
    "RuleSet",
    "Score",
    "Selector",
    "TransitionObservation",
    "induce",
    "observe_transition",
    "predict_frame",
    "score_replay_baseline",
    "score_static_baseline",
    "score_transitions",
]
