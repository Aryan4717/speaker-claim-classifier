from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import cast

from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.errors import CatalogueFormatError, UnknownBrandError
from speaker_claims.domain.facts import BrandFact, ValueType, parse_boolean

_FACT_COLUMNS = (
    "fact_id",
    "brand_id",
    "model_id",
    "field",
    "value",
    "value_type",
    "unit",
    "condition",
    "valid_from",
    "valid_to",
)
_ANSWER_COLUMNS = ("answer_id", "split", "brand_id", "question", "ai_answer")
_VALUE_TYPES = frozenset({"number", "boolean", "string"})

FACT_VERSION = "speaker-catalogue-v4"


def _blank_to_none(value: str) -> str | None:
    if value == "":
        return None
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CatalogueFormatError(f"{path.name} has no header")
        return list(reader)


def _require_columns(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    if not rows:
        raise CatalogueFormatError(f"{path.name} has no records")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise CatalogueFormatError(f"{path.name} is missing columns: {', '.join(missing)}")


def load_answers(path: Path) -> tuple[AnswerRecord, ...]:
    """Load answers without stripping or rewriting ai_answer."""
    rows = _read_rows(path)
    _require_columns(path, rows, _ANSWER_COLUMNS)
    return tuple(
        AnswerRecord(
            answer_id=row["answer_id"],
            split=row["split"],
            brand_id=row["brand_id"],
            question=row["question"],
            ai_answer=row["ai_answer"],
        )
        for row in rows
    )


def _parse_fact(row: dict[str, str]) -> BrandFact:
    value_type = row["value_type"]
    if value_type not in _VALUE_TYPES:
        raise CatalogueFormatError(f"Unknown value_type on {row.get('fact_id', '')}")
    typed = cast(ValueType, value_type)
    raw_value = row["value"]
    if typed == "boolean":
        parse_boolean(raw_value)
    elif typed == "number":
        if not raw_value.isdigit():
            raise CatalogueFormatError(f"Number value is not an integer on {row['fact_id']}")
    try:
        valid_from = date.fromisoformat(row["valid_from"])
        valid_to = date.fromisoformat(row["valid_to"])
    except ValueError as exc:
        raise CatalogueFormatError(f"Invalid date on {row['fact_id']}") from exc
    return BrandFact(
        fact_id=row["fact_id"],
        brand_id=row["brand_id"],
        model_id=row["model_id"],
        field=row["field"],
        raw_value=raw_value,
        value_type=typed,
        unit=_blank_to_none(row["unit"]),
        condition=_blank_to_none(row["condition"]),
        valid_from=valid_from,
        valid_to=valid_to,
    )


class CsvFactCatalogue:
    """In-memory catalogue. This is the only factual authority."""

    def __init__(self, facts: tuple[BrandFact, ...]) -> None:
        seen: set[str] = set()
        grouped: dict[str, list[BrandFact]] = {}
        for fact in facts:
            if fact.fact_id in seen:
                raise CatalogueFormatError(f"Duplicate fact_id: {fact.fact_id}")
            seen.add(fact.fact_id)
            grouped.setdefault(fact.brand_id, []).append(fact)
        self._ids = frozenset(seen)
        self._by_brand = {
            brand_id: tuple(sorted(brand_facts, key=lambda item: item.fact_id))
            for brand_id, brand_facts in grouped.items()
        }

    @classmethod
    def from_path(cls, path: Path) -> CsvFactCatalogue:
        rows = _read_rows(path)
        _require_columns(path, rows, _FACT_COLUMNS)
        return cls(tuple(_parse_fact(row) for row in rows))

    def get_brand_facts(self, brand_id: str) -> tuple[BrandFact, ...]:
        try:
            return self._by_brand[brand_id]
        except KeyError as exc:
            raise UnknownBrandError(brand_id) from exc

    def fact_ids(self) -> frozenset[str]:
        return self._ids
