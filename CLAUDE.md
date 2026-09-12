# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A SoftUni course project: a four-agent LangGraph fraud-investigation workflow for a fictional bank ("RevolutBank"), delivered as a Google Colab notebook that must complete unattended via **Runtime → Run all** with only an `ANTHROPIC_API_KEY` secret.

## Source of truth and the notebook

`fraud_multi_agent.py` (jupytext *percent* format, `# %%` cells) is the source of truth. `Fraud_Detection_Multi_Agent.ipynb` is **generated** from it — never edit the notebook directly. After changing the `.py`:

```bash
python build_notebook.py          # regenerate the notebook
python build_notebook.py --run    # regenerate, then execute (needs a live key in .env)
```

Never call `jupytext` or `jupyter execute` directly: on a Windows console with a legacy code page they silently rewrite every emoji as mojibake. `build_notebook.py` forces UTF-8 in the child processes and verifies the result; `tests/test_notebook_encoding.py` fails if a corrupted notebook is committed.

## Commands

```bash
pip install -r requirements-dev.txt        # setup (cp .env.example .env for a live key)
python -m pytest tests/ -v                 # all tests — no API key required
python -m pytest tests/test_triage.py -v   # one file
python -m pytest tests/ -k "sanctions" -v  # by keyword
```

Tests exercise the real graph (real `interrupt()`, real checkpointer, real routing) against scripted fake LLMs (`FakeLLM` classes inside the test files), so nothing needs the network.

## Test harness contract

`tests/conftest.py` sets `FRAUD_SKIP_DEMOS=1` **before** importing `fraud_multi_agent` — module import runs the demo cells otherwise — and redirects `FRAUD_AUDIT_LOG` / `FRAUD_CHECKPOINT_DB` into a temp dir so run artifacts stay out of the repo. Any new module-level side effect in `fraud_multi_agent.py` must be behind the `SKIP_DEMOS` flag or an env-var override, or the whole suite breaks on import. Use the `fresh_observer` fixture when a test touches `OBS` token/event state.

`tests/test_colab_runall.py` enforces the "upload, Run all, done" contract: exactly one install cell, no `input()` on the run-all path (test cases pass scripted `decisions` lists), no Drive mount, `ANTHROPIC_API_KEY` as the only required secret, and a working fallback for every optional dependency. Check those tests before adding any dependency, secret, or interactive step.

## Architecture

LangGraph `StateGraph` over `FraudWorkflowState` (a TypedDict with an `add_messages` reducer). Flow: `triage` → `fraud_analyst` → `compliance_officer` → `comms_draft` → `human_review` (an `interrupt()`) → on approve `execute_action` → `comms_deliver`. Guard branches end the run early: signal names no customer (`no_customer` / `followup_qa` for a checkpointed thread), unknown customer (`customer_not_found`), reject (`cancel_action`). Feedback loops back to `compliance_officer`, capped at `MAX_REVISIONS = 3`.

`execute_workflow(user_request, *, decisions=None, thread_id=None)` owns the pause/resume loop (`Command(resume=...)`); `start_workflow` / `resume_workflow` are the halves underneath. Durability comes from a SQLite checkpointer that is **probed with a real write** before being trusted (`make_checkpointer`), falling back to `MemorySaver`.

### Invariants the design depends on — do not weaken these

- **Decisions are computed, not generated.** Customer id comes from a regex over the request; the risk score from the `_score()` rule engine; sanctions screening re-runs `check_sanctions_list` on the *fetched* profile name (not whatever the model passed); a prior BLOCK deterministically escalates priority; a sanctions hit deterministically forces BLOCK regardless of the score.
- **One human gate covers every customer-visible side effect.** `comms_draft` runs *before* the interrupt so the reviewer approves the exact words; `comms_deliver` runs *after* `execute_action` and replays the approved text verbatim — the model must never rewrite a message a human signed off.
- **Tool output is untrusted.** Every tool result reaches the model inside an `UNTRUSTED TOOL OUTPUT` fence; `CUST-6006` is the adversarial test profile with an injected instruction in merchant names.
- **Agents cannot reach past their case.** `run_tool_loop(pin_args=...)` overrides the customer id on every tool call that declares one.
- **Observability re-raises.** `@observe_node` records and logs an exception, then re-raises it — never let instrumentation swallow a failure.

### Model/API constraints handled in code (each fails only at runtime with a live key)

- Claude 5 models reject `temperature` (400) — no sampling params anywhere.
- `EFFORT` must be omitted entirely (not `null`) on Haiku 4.5; only Sonnet 5 / Opus 5 accept it.
- Structured output uses `with_structured_output(schema, method="json_schema")` — LangChain's default forced tool calling conflicts with adaptive thinking.
- `langgraph-checkpoint-sqlite` is pinned `>=3.1,<4`: the 2.x line imports cleanly and then dies on the first checkpoint write. Keep the pin and the write-probe in `make_checkpointer` together.
- Cost guards: retries with backoff on transient faults only (429/529/5xx/timeouts fail-fast on 400/401/404), `timeout=120` per call, `MAX_USD_PER_NOTEBOOK` circuit breaker, preflight key check at the top of the notebook.

### Audit and reproducibility

`audit_log.jsonl` is append-only, joined by `case_id`, stamped with `PROMPT_VERSION` and the model id. **Bump `PROMPT_VERSION`** (top of `fraud_multi_agent.py`) whenever an agent system prompt changes. Customer names are PII-masked (`Viktor B.`) in logs and the audit trail. A labelled regression eval in section 8.1 fails the notebook if risk-engine decisions drift.

## Repo notes

- `revolutbank_cases.sqlite*` files in the root are run artifacts of the checkpointer, not source.
- Design spec and implementation plan live in `docs/superpowers/`; `production-migration-plan.md` is the plan this architecture was built from.
- The README's tables (test cases, tools, pricing) mirror the notebook — keep them in sync when behavior changes.
