"""Evaluation run tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261021_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column("agent_version", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("dataset_version", sa.String(), nullable=False),
        sa.Column("evaluator_version", sa.String(), nullable=False),
        sa.Column("fact_version", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("settings", postgresql.JSONB(), nullable=False),
        sa.Column("source_revision", sa.String(), nullable=True),
        sa.Column("file_hashes", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("answer_count", sa.Integer(), nullable=False),
        sa.Column("answer_error_count", sa.Integer(), nullable=False),
        sa.Column("accepted_prediction_count", sa.Integer(), nullable=False),
        sa.Column("quality", postgresql.JSONB(), nullable=True),
    )
    op.create_table(
        "answer_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("answer_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "answer_id", name="uq_answer_results_run_answer"),
    )
    op.create_table(
        "claim_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("answer_id", sa.String(), nullable=False),
        sa.Column("claim_id", sa.String(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("quote", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("fact_ids", postgresql.JSONB(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["run_id", "answer_id"],
            ["answer_results.run_id", "answer_results.answer_id"],
            ondelete="CASCADE",
            name="fk_claim_results_answer",
        ),
        sa.UniqueConstraint(
            "run_id",
            "answer_id",
            "start_char",
            "end_char",
            name="uq_claim_results_span",
        ),
    )


def downgrade() -> None:
    op.drop_table("claim_results")
    op.drop_table("answer_results")
    op.drop_table("runs")
