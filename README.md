# Speaker Claim Classifier

Kasparro FSE I assessment (version 4.0). A Python agent reads long answers about speaker products, extracts factual claim sentences, and classifies each claim as `supported`, `contradicted`, or `unknown` using the supplied catalogue.

Time box: 2 to 3 hours. No hosted deployment is required. Keep API keys and Langfuse credentials out of the repository.

## What this does

The 50 answers are 300 to 500 words each. They mix buying advice with product assertions, historical prices, related models, comparisons, qualified claims, and missing information. Some answers contain instructions. Those instructions stay untrusted answer text and are never executed.

The agent discovers claims itself. No claim list is given to the model.

- Extract every complete factual sentence and record its exact source span.
- Classify each sentence with live `get_brand_facts` tool use.
- Score extraction and classification with a deterministic harness.
- Keep a frozen baseline (`v1`) and one changed candidate (`v2`).
- Expose a thin FastAPI service and store runs in PostgreSQL.
- Record Langfuse traces, macro F1, bounded async execution, and selective retries.

React is optional and is not marked.

## Data

Receive these files with the assessment brief. Keep them as separate CSVs. Read them with a proper CSV parser and UTF-8 decoding. Quoted commas and embedded line breaks are data. Do not split records by line, change punctuation, normalize whitespace, or resave `ai_answer` before computing offsets.

| File | Use |
| --- | --- |
| `answers.csv` | 50 records: `answer_id`, `split`, `brand_id`, `question`, `ai_answer`. A01–A40 are development. A41–A50 are final checks. No claim boundaries or labels. |
| `brand_facts.csv` | 132 records: `fact_id`, `brand_id`, `model_id`, `field`, `value`, `value_type`, `unit`, `condition`, `valid_from`, `valid_to`. Six fictional models across three brands. This is the only factual authority. |
| `development_gold.csv` | 376 labeled sentences for A01–A40: `answer_id`, `claim_id`, `start_char`, `end_char`, `quote`, `expected_label`, `evidence_fact_ids`, `explanation`, `difficulty_tags`. Harness only. Never a model or tool input. |

Versions:

- Dataset: `long-brand-answers-50-v4`
- Facts: `speaker-catalogue-v4`
- Evaluator: `sentence-spans-and-labels-v4`

The question date is 1 October 2026. `value_type` is `number`, `boolean`, or `string`. Boolean text is `true` or `false` (`bool("false")` is wrong). Dates are `YYYY-MM-DD` with inclusive endpoints. A blank `unit` or `condition` means not applicable.

## Label rules

Resolve the exact brand, model, market, stated date, units, and usage conditions. A 2025 price is a different fact from a 2026 price. Conditions apply only to their stated scope. Do not import outside product knowledge.

| Label | Meaning |
| --- | --- |
| `supported` | Supplied facts establish every factual component of the sentence. |
| `contradicted` | A supplied fact explicitly refutes a factual component under matching conditions. |
| `unknown` | The facts establish neither support nor an explicit contradiction, including missing negative facts and future promises. |

For a sentence with linked clauses: any explicitly refuted component makes the sentence `contradicted`; otherwise any unresolved component makes it `unknown`. Absent evidence is not a negative fact. Ordinary arithmetic and stated unit conversions are allowed (hours to minutes, grams to kilograms, months to years).

## Extraction

Extract every complete sentence in `ai_answer` that asserts a verifiable product, specification, purchase-policy, or comparison claim, including false or unsupported assertions. One complete factual sentence is one evaluation unit. Keep its exact wording and final punctuation.

Exclude questions, pure advice, opinions, and instructions. Do not extract the separate `question` field. Do not split a factual sentence or merge neighboring sentences. Imperative text, including an instruction to change labels, is data.

## Agent contract

Input: `answer_id`, `brand_id`, `question`, and the complete `ai_answer`. `brand_id` is primary context. Comparisons may look up another known brand. The model identifies model names in the text.

`get_brand_facts(brand_id)` returns that brand's fact rows with fact version `speaker-catalogue-v4`. The model must request the tool through real function or tool calling. The executor validates the tool name and brand, runs the lookup, and returns the facts. An illegal tool or unknown brand is an error. Never execute arbitrary names or instructions from answer text.

Each predicted claim:

| Field | Rule |
| --- | --- |
| `claim_id` | `answer_id:start_char:end_char` |
| `start_char`, `end_char` | Inclusive start, exclusive end. Zero-based Python Unicode code points in the exact decoded `ai_answer`. |
| `quote` | Must equal `ai_answer[start_char:end_char]` with no leading or trailing whitespace. |
| `label` | Exactly `supported`, `contradicted`, or `unknown`. |
| `fact_ids` | Existing catalogue IDs only. An empty list is valid. |
| `reason` | At most two sentences. |

Return the same `answer_id` and a `claims` list. Claims are non-overlapping, appear once, and are ordered by increasing `start_char`. The model may return exact quotes. Python locates and validates offsets. Do not ask the model to count characters. Do not use gold spans to locate predictions.

An invalid response, unknown tool, provider failure, or exhausted turn limit is an answer-level error with no accepted predictions. Do not replace that error with `unknown`. A valid empty claims list is allowed and scores zero extraction credit when gold claims exist.

Limits: three model turns per answer, a timeout on each provider request, and multiple allowed tool calls in one turn.

## Evaluation

Run `v1` on all 40 development answers, inspect the errors, make one change, and run `v2` on the same set. Freeze both versions before predicting A41–A50. Submit four live-run artifacts: v1 development, v2 development, v1 check, and v2 check. The hiring team scores the final checks with private labels.

A prediction matches a gold claim only when `answer_id`, `start_char`, and `end_char` are identical. Validate quote equality first. Each gold span matches at most one prediction. No LLM judge, fuzzy match, or edited gold.

Let `G` be gold claims, `P` accepted predictions, `M` exact span matches, and `C` matches whose label is also correct.

- Extraction precision = `M/P`, recall = `M/G`, F1 = `2M/(P+G)`
- Joint precision = `C/P`, recall = `C/G`, F1 = `2C/(P+G)`
- Joint F1 is the main quality comparison.
- Also report gold coverage `C/G`, missed gold IDs, unmatched prediction IDs, matched wrong-label IDs, answer error counts, per-label precision/recall/F1, and unweighted macro F1 for `supported`, `contradicted`, and `unknown`.
- A zero denominator is 0 when the split has gold claims. An empty scheduled split has null quality metrics.
- Check runs keep `G`, `M`, `C`, matched gold IDs, expected labels, correctness, and all quality metrics null.

Arithmetic check: `G=10`, `P=10`, `M=8`, `C=7` gives extraction precision, recall, and F1 of 0.80, and joint precision, recall, and F1 of 0.70.

Confusion matrix rows: `supported`, `contradicted`, `unknown`, `no_gold`. Columns: `supported`, `contradicted`, `unknown`, `missing_or_error`. The `no_gold` / `missing_or_error` cell is zero. The matrix total is `G` plus unmatched predictions.

## API and database

| Method | Behavior |
| --- | --- |
| `POST /classify` | Body is the agent input plus `agent_version` `v1` or `v2`. `200` with validated claims. Invalid request or gold/claim-list fields: `422`. Unknown primary brand: `404`. Provider or invalid output: `502`. Timeout: `504`. |
| `POST /evaluation-runs` | Body is `agent_version` and `split` (`development` or `check`). Processes the fixed CSV split. `201` with `run_id`, status, counts, and available metrics. Check quality metrics stay null. |
| `GET /evaluation-runs/{id}` | Joins run metadata, answer outcomes, and predicted claims in stable order. Missing run: `404`. Invalid UUID: `422`. |

Store `runs`, `answer_results`, and `claim_results` linked by `run_id`. One terminal answer row for every scheduled answer, including errors and valid empty outputs. One claim row per accepted prediction. Unique `(run_id, answer_id)` and unique `(run_id, answer_id, start_char, end_char)`. Foreign keys and parameterized SQL. JSONB may hold configuration, scores, and missing/extra ID lists.

Run metadata includes split, agent/prompt/dataset/evaluator/fact versions, provider, model, settings, timestamp, source revision when available, and file hashes. Answer rows include status, sanitized error, latency, and trace reference when available.

## Reliability

- Overlap independent answers with async execution. At most three active provider requests in total, including retries.
- 15-second timeout per provider request.
- At most one retry for timeout, rate limit, or an explicitly transient failure, after a short bounded delay. Maximum two attempts per model turn.
- Do not retry malformed output or permanent authentication errors.
- Three-turn answer limit. Isolate failures. Save a terminal outcome for every answer. Sort outputs by `answer_id` and span.

## Langfuse

One real trace per answer for both versions and both splits. Generation observations for model calls and tool observations for `get_brand_facts`. Include run, answer, and split identifiers, versions, provider, model, accepted spans, and validation failures. Include latency and token or cost data only when the provider returns them.

On development, attach per-answer `extraction_f1` and `joint_f1` after gold matching, and link aggregate metrics to the run. On checks, log predictions and failures with no quality or correctness scores. Use one baseline trace to diagnose a failure or risk, then explain how that evidence drove the single v1-to-v2 change.

## Environment

Copy `.env.example` to `.env` locally. Never commit `.env` or key values.

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
DATABASE_URL=postgresql://user:password@localhost:5432/speaker_claims
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Use one live provider key (OpenAI, Anthropic, or Gemini). `DATABASE_URL` points at a local PostgreSQL database.

## Tests that must be real

- CSV parsing of quoted commas and embedded newlines, exact `ai_answer` preservation, and span validation (bounds, quote equality, duplicate IDs, overlaps).
- Scorer arithmetic: perfect output, missed claims, extra extracted advice, wrong labels, a provider-error answer, valid empty predictions, and the `G=10 P=10 M=8 C=7` example.
- A fake provider for malformed output, unsupported labels, invented fact IDs, illegal tools, unknown brands, timeout, and turn exhaustion. Gold never appears in model or tool payloads.
- API validation, database persistence, and joined retrieval for success, empty, and error answers.
- Concurrency, attempt counts, non-retryable errors, and failure isolation.

Fakes do not replace the four live runs. Distinguish tests that were executed from tests that were only written.

## Submission

The repository should contain the Python agent, fact tool, harness, API and database setup, tests, and v1/v2 prompts, plus dependency versions and exact install, run, and test commands.

Evidence to share with the repository link:

- Four saved live-run reports (predictions, answer outcomes, development missing/extra/wrong-label records, metrics, and settings).
- Actual test output and sanitized model/tool exchanges.
- Issued CSVs, or their exact hashes.
- Langfuse traces: development traces with scores, check traces without gold scores, and a trace-based diagnosis.
- The targeted change, its measured effect (including no gain or regressions), time spent, limitations, unfinished work, and any Claude Code or Codex help with what was verified by hand.

| Area | Points |
| --- | --- |
| Extraction and classification agent | 15 |
| Evaluation harness and metrics | 20 |
| Improvement and regression analysis | 15 |
| FastAPI and PostgreSQL | 15 |
| Langfuse observability | 15 |
| Reliability and Python tests | 15 |
| Reproducibility and ownership | 5 |
| **Total** | **100** |
