from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

runs = Table(
    "runs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("split", String, nullable=False),
    Column("agent_version", String, nullable=False),
    Column("prompt_version", String, nullable=False),
    Column("dataset_version", String, nullable=False),
    Column("evaluator_version", String, nullable=False),
    Column("fact_version", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("model", String, nullable=False),
    Column("settings", JSONB, nullable=False),
    Column("source_revision", String, nullable=True),
    Column("file_hashes", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("status", String, nullable=False),
    Column("answer_count", Integer, nullable=False),
    Column("answer_error_count", Integer, nullable=False),
    Column("accepted_prediction_count", Integer, nullable=False),
    Column("quality", JSONB, nullable=True),
)

answer_results = Table(
    "answer_results",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("run_id", UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
    Column("answer_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("error_code", String, nullable=True),
    Column("error_message", String, nullable=True),
    Column("latency_ms", Integer, nullable=True),
    Column("trace_id", String, nullable=True),
    UniqueConstraint("run_id", "answer_id", name="uq_answer_results_run_answer"),
)

claim_results = Table(
    "claim_results",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("run_id", UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
    Column("answer_id", String, nullable=False),
    Column("claim_id", String, nullable=False),
    Column("start_char", Integer, nullable=False),
    Column("end_char", Integer, nullable=False),
    Column("quote", String, nullable=False),
    Column("label", String, nullable=False),
    Column("fact_ids", JSONB, nullable=False),
    Column("reason", String, nullable=False),
    UniqueConstraint(
        "run_id",
        "answer_id",
        "start_char",
        "end_char",
        name="uq_claim_results_span",
    ),
    ForeignKeyConstraint(
        ["run_id", "answer_id"],
        ["answer_results.run_id", "answer_results.answer_id"],
        ondelete="CASCADE",
        name="fk_claim_results_answer",
    ),
)
