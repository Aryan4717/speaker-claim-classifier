from __future__ import annotations

from typing import Protocol
from uuid import UUID

from speaker_claims.domain.runs import StoredRun


class RunStorePort(Protocol):
    def save(self, run: StoredRun) -> None:
        """Persist one run, its answers, and its claims in a single transaction."""

    def get(self, run_id: UUID) -> StoredRun | None:
        """Return the joined run, or None when the id is unknown."""
