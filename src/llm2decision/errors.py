"""Every error the package raises derives from `LLM2DecisionError`, so one `except` catches them all;
each also keeps the built-in base it had (`ValueError` for bad input, `RuntimeError` for a failure)."""


class LLM2DecisionError(Exception):
    """Base of every llm2decision error."""


class ConfigError(LLM2DecisionError, ValueError):
    """An unknown model, binding or provider, or settings that contradict each other."""


class AuthError(LLM2DecisionError, RuntimeError):
    """A key is needed and none was found, or one cannot be used. Says where it looked, never the key."""


class MarksError(LLM2DecisionError, ValueError):
    """Options that cannot be marked: too few, too many, or not distinct."""


class QuestionError(LLM2DecisionError, ValueError):
    """A question that cannot be asked, named by the key it was sent under."""

    def __init__(self, key: str, message: str):
        super().__init__(f"question {key!r}: {message}")
        self.key = key


class TransportError(LLM2DecisionError, RuntimeError):
    """The provider refused or failed after retries. Never carries the key."""


class UnreadableAnswer(LLM2DecisionError, RuntimeError):
    """The model answered, but not with any of the options (strict mode only; otherwise the answer
    comes back with `meta.answered = False`)."""

    def __init__(self, key: str, message: str):
        super().__init__(f"question {key!r}: {message}")
        self.key = key
