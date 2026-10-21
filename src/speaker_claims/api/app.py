from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.errors import UnavailableAgentVersion, UnknownBrandError
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredClaim, StoredRun
from speaker_claims.ports.monitor import ErrorMonitorPort


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1)
    brand_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ai_answer: str = Field(min_length=1)
    agent_version: str = Field(pattern="^v[12]$")


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_version: str = Field(pattern="^v[12]$")
    split: str = Field(pattern="^(development|check)$")


def create_app(service: EvaluationService, monitor: ErrorMonitorPort | None = None) -> FastAPI:
    app = FastAPI(title="Speaker claim classifier")
    reporter = service.monitor if monitor is None else monitor

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        reporter.breadcrumb(
            category="http",
            message="validation",
            data={"status_code": 422, "path": request.url.path},
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if exc.status_code in {404, 422}:
            reporter.breadcrumb(
                category="http",
                message="client_error",
                data={"status_code": exc.status_code, "path": request.url.path},
            )
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        del request
        reporter.capture_exception(exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.post("/classify")
    def classify(body: ClassifyRequest) -> dict[str, object]:
        answer = AnswerRecord(
            answer_id=body.answer_id,
            split="",
            brand_id=body.brand_id,
            question=body.question,
            ai_answer=body.ai_answer,
        )
        try:
            outcome = service.classify(answer, body.agent_version)
        except UnavailableAgentVersion as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UnknownBrandError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if outcome.status == "error":
            raise HTTPException(status_code=_error_status(outcome), detail=_error_body(outcome))
        return {
            "answer_id": outcome.answer_id,
            "agent_version": outcome.agent_version,
            "claims": [_predicted_claim(claim) for claim in outcome.claims],
        }

    @app.post("/evaluation-runs", status_code=201)
    def start_run(body: EvaluationRequest) -> dict[str, object]:
        try:
            run = service.start_run(body.agent_version, body.split)
        except UnavailableAgentVersion as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _run_body(run)

    @app.get("/evaluation-runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        try:
            parsed = UUID(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Run id must be a UUID") from exc
        run = service.get_run(parsed)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_body(run)

    return app


def _error_status(outcome: AnswerOutcome) -> int:
    if outcome.error_code == "timeout":
        return 504
    return 502


def _error_body(outcome: AnswerOutcome) -> dict[str, str | None]:
    return {"error_code": outcome.error_code, "message": outcome.error_message}


def _predicted_claim(claim: object) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "start_char": claim.start_char,
        "end_char": claim.end_char,
        "quote": claim.quote,
        "label": claim.label,
        "fact_ids": list(claim.fact_ids),
        "reason": claim.reason,
    }


def _run_body(run: StoredRun) -> dict[str, object]:
    return {
        "run_id": str(run.run_id),
        "status": run.status,
        "split": run.split,
        "agent_version": run.agent_version,
        "prompt_version": run.prompt_version,
        "dataset_version": run.dataset_version,
        "evaluator_version": run.evaluator_version,
        "fact_version": run.fact_version,
        "provider": run.provider,
        "model": run.model,
        "settings": run.settings,
        "source_revision": run.source_revision,
        "file_hashes": run.file_hashes,
        "created_at": run.created_at.isoformat(),
        "answer_count": run.answer_count,
        "answer_error_count": run.answer_error_count,
        "accepted_prediction_count": run.accepted_prediction_count,
        "quality": run.quality,
        "answers": [
            {
                "answer_id": answer.answer_id,
                "status": answer.status,
                "error_code": answer.error_code,
                "error_message": answer.error_message,
                "latency_ms": answer.latency_ms,
                "trace_id": answer.trace_id,
                "claims": [_stored_claim(claim) for claim in answer.claims],
            }
            for answer in run.answers
        ],
    }


def _stored_claim(claim: StoredClaim) -> dict[str, object]:
    return _predicted_claim(claim)
