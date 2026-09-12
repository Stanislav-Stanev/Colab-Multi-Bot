# Design: Observability & Logging for the Fraud Detection Workflow

> **Historical record — superseded in places.** This document says what was planned on
> 2026-09-10 and is kept unedited for that reason. The code has since moved on: the
> RevolutBank rebrand added a triage and a comms agent, and the harness-engineering
> review ([`hareness_improvment.md`](../../../hareness_improvment.md)) changed the tool
> signatures, the analyst's tool kit, the revision-exhaustion route and the audit format.
> For current behaviour read `fraud_multi_agent.py`, `README.md` and `CLAUDE.md`.

**Date:** 2026-09-10
**Extends:** [2026-09-10-fraud-detection-multi-agent-design.md](2026-09-10-fraud-detection-multi-agent-design.md)

## Problem

The workflow currently reports itself through 23 bare `print()` calls. That is readable in a
notebook and worthless everywhere else: no levels, no way to quiet it, nothing pytest can
capture, no timings, no per-node cost, no record of a run after the kernel dies, and an
exception inside a node surfaces only as a traceback with no context about which node,
which customer, or how far the run got.

`UsageTracker` already counts tokens globally, but cannot say which agent spent them.

## Approved approach

Self-contained structured observability (no new pip installs, nothing leaves the machine),
plus LangSmith tracing that stays off unless a key is present. The existing emoji traces
are kept verbatim — they are what a grader reads — but travel through the logger.

## Components

All of it lives in a new notebook section **1.5 Observability & logging**, after config and
before the tools, so every component downstream can use it.

### 1. `setup_logging(level="INFO") -> logging.Logger`

Configures a logger named `fraud` with one stream handler.

- At `INFO` the formatter emits **only the message**, so the trace reads exactly as it does
  today (`📥 intake: ...`).
- At `DEBUG`/`WARNING`/`ERROR` it prefixes level and node.
- **Idempotent**: existing handlers are removed first. Re-running a cell in Colab otherwise
  attaches a second handler and every subsequent line is printed twice.
- `propagate = False` so the root logger does not double-print.

### 2. `WorkflowObserver(BaseCallbackHandler)`

Absorbs and replaces `UsageTracker`. One instance, `OBS`, is the run's source of truth.

State:
- `events: list[dict]` — every record is `{ts, elapsed_s, run_id, thread_id, node, event, ...}`
- `nodes: dict[str, NodeStats]` — per node: `calls`, `seconds`, `input_tokens`,
  `output_tokens`, `errors`
- `current_node: str | None` — set by the decorator, read by `on_llm_end`, which is how an
  LLM call is attributed to the agent that made it.

Event types: `run_start`, `node_start`, `node_end`, `llm_call`, `tool_call`, `interrupt`,
`human_decision`, `error`, `run_end`.

Callback hooks used: `on_llm_end` (tokens), `on_llm_error` and `on_tool_error` (failures).

Methods:
- `start_run(thread_id, user_request)` — new `run_id`, resets the clock, keeps history
- `record(event, **fields)` — append a structured event
- `summary()` — per-node table: node, calls, seconds, tokens, cost
- `timeline()` — chronological events, one line each
- `errors()` — only the failures
- `to_json(path)` — the full structured log, for after the kernel is gone
- `cost_usd` — priced from `PRICING`, as before

### 3. `@observe_node(name)`

Wraps each of the 8 graph nodes:

1. sets `current_node`, records `node_start`
2. times the call with `perf_counter`
3. on exception: records `error` with the exception type and message, logs at `ERROR`,
   **re-raises** (observability must not swallow failures)
4. records `node_end` with duration and the tokens spent inside it
5. restores the previous `current_node`

### 4. `enable_langsmith() -> bool`

Looks for `LANGSMITH_API_KEY` through the existing secret chain **without raising**. If
found, sets `LANGCHAIN_TRACING_V2=true`, the API key, and the project name, and returns
True. If not, logs one line saying tracing is off and returns False. Off by default; no
data leaves the machine unless the user supplies a key.

## Known limitation (documented, not hidden)

Per-node token attribution assumes nodes execute one at a time, which is true for this
graph. If the graph ever gains parallel branches, concurrent LLM calls would be attributed
to whichever node happens to be current. A contextvar would fix it; it is not warranted for
a sequential graph and would obscure the code.

## Testing

Runs without an API key, using the existing scripted fake LLM:

- `setup_logging` twice leaves exactly one handler (the duplicate-output bug)
- log level is honoured; INFO messages carry no formatter noise
- a node's start/end events are recorded with a plausible duration
- an exception inside a node is recorded as `error` **and re-raised**
- tokens from an LLM call land on the node that was current
- `summary()` totals match the sum of the per-node rows
- `to_json()` round-trips and contains every event
- `enable_langsmith()` returns False and sets no env var when no key exists
- the full graph run produces one `run_start`, one `run_end`, and node events in order

## Deliverables

- New section 1.5 in `fraud_multi_agent.py`
- Nodes decorated; `print()` replaced by `log.info` with identical text
- A reporting cell after the test cases: `OBS.summary()` and the cost line
- `tests/test_observability.py`
- README section describing the logger, the report, and the LangSmith opt-in
