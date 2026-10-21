from __future__ import annotations

from typing import Protocol

from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.outcome import AnswerOutcome


class ClassifierPort(Protocol):
    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        """Return one terminal outcome. Do not raise for a model failure."""
