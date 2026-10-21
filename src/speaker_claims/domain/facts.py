from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

ValueType = Literal["number", "boolean", "string"]

_BOOLEAN_TEXT = {"true": True, "false": False}


def parse_boolean(value: str) -> bool:
    """Parse catalogue booleans. Only the strings true and false are accepted."""
    try:
        return _BOOLEAN_TEXT[value]
    except KeyError as exc:
        raise ValueError("Boolean text must be true or false") from exc


@dataclass(frozen=True)
class BrandFact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    raw_value: str
    value_type: ValueType
    unit: str | None
    condition: str | None
    valid_from: date
    valid_to: date

    @property
    def typed_value(self) -> int | bool | str:
        if self.value_type == "boolean":
            return parse_boolean(self.raw_value)
        if self.value_type == "number":
            return int(self.raw_value)
        return self.raw_value

    def as_record(self) -> dict[str, str]:
        """Return the fact as catalogue text for a tool result."""
        return {
            "fact_id": self.fact_id,
            "brand_id": self.brand_id,
            "model_id": self.model_id,
            "field": self.field,
            "value": self.raw_value,
            "value_type": self.value_type,
            "unit": self.unit or "",
            "condition": self.condition or "",
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat(),
        }
