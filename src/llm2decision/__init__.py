"""Any hosted LLM as a decision model, used like the TypeSafe (Jev) SDK.

    from llm2decision import DecisionClient, Noul, Choice, Score

    with DecisionClient("qwen3.8-27b@nebius") as client:
        r = client.system_one(state, {"team": Choice(criteria={"billing": "…", "other": None})})
        r.choices["team"].choice
"""
from .client import AsyncDecisionClient, DecisionClient
from .config import LLM2DecisionWarning, ModelInfo, bindings, list_models, providers
from .errors import (AuthError, ConfigError, LLM2DecisionError, MarksError, QuestionError, TransportError,
                     UnreadableAnswer)
from .marks import Marks, assign, read
from .types import (Answer, Choice, ChoiceAnswer, Meta, Noul, NoulAnswer, Score, ScoreAnswer, SystemOneResponse,
                    Tfu, TfuAnswer, Usage)

__all__ = ["DecisionClient", "AsyncDecisionClient", "list_models", "ModelInfo", "Noul", "Tfu", "Choice", "Score", "NoulAnswer", "TfuAnswer", "ChoiceAnswer",
           "ScoreAnswer", "Answer", "SystemOneResponse", "Usage", "Meta", "LLM2DecisionError", "QuestionError",
           "ConfigError", "AuthError", "TransportError", "UnreadableAnswer", "LLM2DecisionWarning", "bindings", "providers", "Marks", "MarksError", "assign", "read"]
__version__ = "0.1.0"
