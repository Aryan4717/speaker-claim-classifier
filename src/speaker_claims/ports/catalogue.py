from __future__ import annotations

from typing import Protocol

from speaker_claims.domain.facts import BrandFact


class FactCataloguePort(Protocol):
    """The only catalogue operation the agent is allowed to request."""

    def get_brand_facts(self, brand_id: str) -> tuple[BrandFact, ...]:
        """Return every fact row for one known brand."""
