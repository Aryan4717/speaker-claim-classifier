from __future__ import annotations

import csv
from pathlib import Path

from speaker_claims.domain.claims import LABELS
from speaker_claims.domain.errors import DomainError
from speaker_claims.domain.gold import GoldClaim

_COLUMNS = (
    "answer_id",
    "claim_id",
    "start_char",
    "end_char",
    "quote",
    "expected_label",
    "evidence_fact_ids",
    "explanation",
    "difficulty_tags",
)


class GoldFormatError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__("gold_format", message)


def load_gold(path: Path) -> tuple[GoldClaim, ...]:
    """Load harness labels. Callers must not pass this data to the model."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise GoldFormatError(f"{path.name} has no header")
        rows = list(reader)
    if not rows:
        raise GoldFormatError(f"{path.name} has no records")
    missing = [column for column in _COLUMNS if column not in rows[0]]
    if missing:
        raise GoldFormatError(f"{path.name} is missing columns: {', '.join(missing)}")
    return tuple(_parse(row) for row in rows)


def _parse(row: dict[str, str]) -> GoldClaim:
    label = row["expected_label"]
    if label not in LABELS:
        raise GoldFormatError(f"Unknown label on {row.get('claim_id', '')}")
    try:
        start = int(row["start_char"])
        end = int(row["end_char"])
    except ValueError as exc:
        raise GoldFormatError(f"Invalid offsets on {row.get('claim_id', '')}") from exc
    return GoldClaim(
        answer_id=row["answer_id"],
        claim_id=row["claim_id"],
        start_char=start,
        end_char=end,
        quote=row["quote"],
        expected_label=label,
        evidence_fact_ids=_split_ids(row["evidence_fact_ids"]),
        explanation=row["explanation"],
        difficulty_tags=_split_ids(row["difficulty_tags"]),
    )


def _split_ids(value: str) -> tuple[str, ...]:
    if value == "":
        return ()
    return tuple(part for part in value.split("|") if part != "")
