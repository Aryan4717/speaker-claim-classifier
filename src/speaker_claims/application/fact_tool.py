from __future__ import annotations

from speaker_claims.domain.errors import IllegalToolError, UnknownBrandError
from speaker_claims.domain.facts import BrandFact
from speaker_claims.ports.catalogue import FactCataloguePort

TOOL_NAME = "get_brand_facts"
FACT_VERSION = "speaker-catalogue-v4"


class FactToolExecutor:
    """Run the one legal catalogue tool. Answer text is never treated as code."""

    def __init__(self, catalogue: FactCataloguePort) -> None:
        self._catalogue = catalogue

    def execute(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        if tool_name != TOOL_NAME:
            raise IllegalToolError(tool_name)
        brand_id = arguments.get("brand_id")
        if not isinstance(brand_id, str) or brand_id == "":
            raise UnknownBrandError(brand_id if isinstance(brand_id, str) else "")
        facts = self._catalogue.get_brand_facts(brand_id)
        return _payload(brand_id, facts)


def _payload(brand_id: str, facts: tuple[BrandFact, ...]) -> dict[str, object]:
    return {
        "fact_version": FACT_VERSION,
        "brand_id": brand_id,
        "facts": [fact.as_record() for fact in facts],
    }
