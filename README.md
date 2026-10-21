# Speaker Claim Classifier

Kasparro FSE I assessment (version 4.0). A Python agent reads long answers about speaker products, extracts factual claim sentences, and classifies each claim as `supported`, `contradicted`, or `unknown` against the supplied catalogue.

Time box: 2 to 3 hours. No hosted deployment. API keys, Langfuse credentials, and `SENTRY_DSN` stay out of the repository.

`feat/catalogue-and-validation` adds the CSV catalogue, `get_brand_facts`, and the span-validation chain. The agent, scorer, API, Langfuse traces, and Sentry adapter are still later branches. Sections below that name those pieces describe the contract, not code that is already here.

## What this does

The 50 answers are 300 to 500 words each. They mix buying advice with product assertions, historical prices, related models, comparisons, qualified claims, and missing information. Some answers contain instructions. Those instructions stay untrusted answer text and are never executed.

The agent discovers claims itself. No claim list is given to the model.

- Extract every complete factual sentence and record its exact source span.
- Classify each sentence with live `get_brand_facts` tool use.
- Score extraction and classification with a deterministic harness.
- Keep a frozen baseline (`v1`) and one changed candidate (`v2`).
- Expose a thin FastAPI service and store runs in PostgreSQL.
- Record Langfuse traces, macro F1, bounded async execution, and selective retries.
- Report unexpected failures through an optional Sentry adapter. Claim logic will depend on a port, not on `sentry_sdk`.

## Services

**Postgres** is the source of truth for evaluation runs. This branch runs PostgreSQL 16 in Docker only. Schema and migrations arrive with `feat/fastapi-postgres`.

**The API** will own classification and evaluation. It is not started on this branch. When it exists, routes will call application services, and those services will depend on ports for the model, the catalogue, run storage, Langfuse, and Sentry.

**Settings** load in `src/speaker_claims/config.py`. A missing `OPENAI_API_KEY`, a missing `DATABASE_URL`, a non-PostgreSQL URL, or a malformed `SENTRY_DSN` stops startup. An empty `SENTRY_DSN` is valid.

**The catalogue** is `data/brand_facts.csv`, loaded by `CsvFactCatalogue`. `FactToolExecutor` is the only caller that may request facts, and it accepts only `get_brand_facts`. The executor depends on `FactCataloguePort`. It does not read the CSV itself.

## Catalogue and span validation

```mermaid
flowchart LR
  executor[FactToolExecutor]
  port[FactCataloguePort]
  csv[CsvFactCatalogue]
  chain[SpanCheck chain]
  executor --> port
  port --> csv
  chain --> bounds[BoundsCheck]
  bounds --> quote[QuoteCheck]
  quote --> claimId[ClaimIdCheck]
  claimId --> label[LabelCheck]
  label --> factIds[FactIdCheck]
  factIds --> duplicate[DuplicateCheck]
  duplicate --> overlap[OverlapCheck]
  overlap --> orderCheck[OrderCheck]
  orderCheck --> reason[ReasonCheck]
```

| Principle | Where it appears |
| --- | --- |
| Single responsibility | Each `SpanCheck` enforces one rule. `CsvFactCatalogue` only loads and serves facts. `FactToolExecutor` only accepts or rejects a tool call. |
| Open/closed | A new span rule is a new `SpanCheck` added in `build_span_chain`. Existing checks stay unchanged. |
| Liskov substitution | Tests can pass any object with `get_brand_facts` where a `FactCataloguePort` is required. |
| Interface segregation | `FactCataloguePort` exposes `get_brand_facts` only. Fact-id membership stays on the CSV adapter and is passed into the chain as a set. |
| Dependency inversion | `FactToolExecutor` takes a `FactCataloguePort`. The CSV class is constructed by the caller. |

Boolean catalogue text is parsed only from `true` and `false`. Dates stay `YYYY-MM-DD`. A blank unit or condition is stored as not applicable. `ai_answer` is kept exactly as decoded, so gold offsets match `ai_answer[start_char:end_char]`. Gold rows are read by tests on this branch. The tool path does not open `development_gold.csv`.

## Data

The issued files are copied byte-for-byte into `data/`. Hashes are in `data/SHA256SUMS`. Read them with the `csv` module and UTF-8. Quoted commas and embedded line breaks are data. Do not split records by line, change punctuation, normalize whitespace, or resave `ai_answer` before computing offsets.

| File | Use |
| --- | --- |
| `data/answers.csv` | 50 records: `answer_id`, `split`, `brand_id`, `question`, `ai_answer`. A01–A40 are development. A41–A50 are final checks. |
| `data/brand_facts.csv` | 132 records: `fact_id`, `brand_id`, `model_id`, `field`, `value`, `value_type`, `unit`, `condition`, `valid_from`, `valid_to`. Six fictional models across three brands. This is the only factual authority. |
| `data/development_gold.csv` | 376 labeled sentences for A01–A40. Harness only. Never a model or tool input. |

Versions: dataset `long-brand-answers-50-v4`, facts `speaker-catalogue-v4`, evaluator `sentence-spans-and-labels-v4`.

The question date is 1 October 2026. `value_type` is `number`, `boolean`, or `string`. Boolean text is `true` or `false`. Dates are `YYYY-MM-DD` with inclusive endpoints. A blank `unit` or `condition` means not applicable.

## Label rules

Resolve the exact brand, model, market, stated date, units, and usage conditions. A 2025 price is a different fact from a 2026 price. Do not import outside product knowledge.

| Label | Meaning |
| --- | --- |
| `supported` | Supplied facts establish every factual component of the sentence. |
| `contradicted` | A supplied fact explicitly refutes a factual component under matching conditions. |
| `unknown` | The facts establish neither support nor an explicit contradiction. |

For a sentence with linked clauses: any explicitly refuted component makes the sentence `contradicted`; otherwise any unresolved component makes it `unknown`. Absent evidence is not a negative fact. Ordinary arithmetic and stated unit conversions are allowed.

## Agent contract

Input: `answer_id`, `brand_id`, `question`, and the complete `ai_answer`. `get_brand_facts(brand_id)` returns that brand's fact rows. The model must request the tool through real function calling. The executor validates the tool name and brand. An illegal tool or unknown brand is an error. Never execute instructions found in answer text.

Each predicted claim uses `claim_id` = `answer_id:start_char:end_char`, an inclusive start and exclusive end in the exact decoded `ai_answer`, a quote equal to that slice, a label of `supported`, `contradicted`, or `unknown`, existing fact ids only, and a reason of at most two sentences. Python locates offsets from exact quotes. Gold spans are never used to locate predictions.

An invalid response, unknown tool, provider failure, or exhausted turn limit is an answer-level error. Do not replace that error with `unknown`. Limits: three model turns per answer, a timeout on each provider request, and multiple tool calls in one turn.

## Evaluation

Run `v1` on all 40 development answers, inspect the errors, make one change, and run `v2` on the same set. Freeze both versions before predicting A41–A50. Submit four live-run artifacts. A prediction matches a gold claim only when `answer_id`, `start_char`, and `end_char` are identical.

Let `G` be gold claims, `P` accepted predictions, `M` exact span matches, and `C` matches whose label is also correct.

- Extraction precision = `M/P`, recall = `M/G`, F1 = `2M/(P+G)`
- Joint precision = `C/P`, recall = `C/G`, F1 = `2C/(P+G)`
- Joint F1 is the main quality comparison.
- Also report gold coverage, missed gold ids, unmatched prediction ids, matched wrong-label ids, answer error counts, per-label precision/recall/F1, and unweighted macro F1.
- Check runs keep quality metrics null.

Arithmetic check: `G=10`, `P=10`, `M=8`, `C=7` gives extraction precision, recall, and F1 of 0.80, and joint precision, recall, and F1 of 0.70.

## API and database

| Method | Behavior |
| --- | --- |
| `POST /classify` | Body is the agent input plus `agent_version` `v1` or `v2`. `200` with validated claims. Invalid request: `422`. Unknown primary brand: `404`. Provider or invalid output: `502`. Timeout: `504`. |
| `POST /evaluation-runs` | Body is `agent_version` and `split` (`development` or `check`). `201` with `run_id` and available metrics. Check quality metrics stay null. |
| `GET /evaluation-runs/{id}` | Joins run metadata, answer outcomes, and predicted claims. Missing run: `404`. Invalid UUID: `422`. |

Store `runs`, `answer_results`, and `claim_results`. These tables are not migrated on this branch.

## Reliability and observability

Overlap independent answers. At most three active provider requests, including retries. 15-second timeout per provider request. At most one retry for timeout, rate limit, or an explicitly transient failure. Do not retry malformed output or permanent authentication errors.

Langfuse: one trace per answer, generation observations for model calls, and tool observations for `get_brand_facts`. Development traces carry per-answer `extraction_f1` and `joint_f1`. Check traces do not carry quality scores.

Sentry is optional. When `SENTRY_DSN` is unset, the process runs without it. The SDK will be imported only by the monitor adapter on `feat/sentry-integration`.

## Branching strategy

`master` stays the published branch. `develop` is the integration branch. Feature branches are cut from `develop` and pushed one at a time.

```
master
  └─ develop
       ├─ feat/project-setup-and-tooling
       ├─ feat/catalogue-and-validation
       ├─ feat/openai-agent-v1
       ├─ feat/evaluation-harness
       ├─ feat/fastapi-postgres
       ├─ feat/reliability-controls
       ├─ feat/langfuse-traces
       ├─ feat/sentry-integration
       └─ feat/agent-v2-and-reports
```

The catalogue port, the tool executor, and the span chain are in this branch. Schema, request data flow, Langfuse, and Sentry are written into this file on the branches that add them.

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| API | FastAPI, Uvicorn |
| Settings | Pydantic Settings |
| Database | PostgreSQL 16, SQLAlchemy, Alembic, psycopg |
| Model | OpenAI, tool calling |
| Traces | Langfuse |
| Errors | Sentry, optional, adapter only |
| Tests | pytest, pytest-asyncio |
| Lint | Ruff |

## Local setup

### Prerequisites

- Python 3.11+
- Docker, for Postgres

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env
```

Fill `OPENAI_API_KEY` in `.env` before a live model call. Leave `SENTRY_DSN` empty until you want error reporting.

### Postgres

```powershell
docker compose up -d
```

`DATABASE_URL` in `.env.example` matches this compose file.

### Checks on this branch

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
```

## Environment

Copy `.env.example` to `.env`. Never commit `.env` or key values.

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
DATABASE_URL=postgresql://user:password@localhost:5432/speaker_claims
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
SENTRY_DSN=
SENTRY_ENVIRONMENT=local
```

One live provider key is required. This project uses OpenAI. `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are accepted and unused.

## Submission map

| Area | Branch |
| --- | --- |
| Extraction and classification agent | `feat/openai-agent-v1`, then `feat/agent-v2-and-reports` |
| Evaluation harness and metrics | `feat/evaluation-harness` |
| Improvement and regression analysis | `feat/agent-v2-and-reports` |
| FastAPI and PostgreSQL | `feat/fastapi-postgres` |
| Langfuse observability | `feat/langfuse-traces` |
| Reliability and Python tests | `feat/reliability-controls`, plus tests on each branch |
| Sentry, optional and decoupled | `feat/sentry-integration` |
| Reproducibility | this branch: pins, hashes, setup commands |
