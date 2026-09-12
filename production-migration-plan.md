# RevolutBank Fraud Operations — Production Migration Plan

**Status: IMPLEMENTED.** All seven phases are done, 164 offline tests pass, and the notebook
has been executed end to end against the live Anthropic API with zero error outputs
(≈ $0.14 per full run on Haiku 4.5). §6 records what each phase delivered, including the
two defects the live run exposed.

This plan turns the current two-agent "Transaction Fraud Analyst" notebook into the
**RevolutBank Fraud Operations Workflow**: a four-agent LangGraph system that still satisfies
every requirement of the SoftUni assignment (`Project-Assignment-document.docx`) and, on top
of that, demonstrates enterprise production practices.

## 0. The hard constraint — "import, Run all, done"

Confirmed with the owner three times, and it overrides every other preference in this
document. The acceptance test for the whole migration is:

> Upload `Fraud_Detection_Multi_Agent.ipynb` to Google Colab, add `ANTHROPIC_API_KEY` in the
> 🔑 Secrets panel, press **Runtime → Run all**, and the notebook runs top to bottom
> unattended to the final cell. **Nothing else.** No mounting, no OAuth pop-up, no manual
> cell edits, no uploaded files, no second runtime, no interactive prompt to answer.

Rules that follow from it, and that every design decision below is checked against:

| Rule | Consequence |
|---|---|
| Only `pip install` from the first cell may add anything | No apt packages, no system services, no build steps |
| Every dependency must import cleanly on a stock Colab runtime | Any optional package is wrapped in `try/import` with a working fallback — a missing wheel must never break Run all |
| No interactive input during Run all | HITL decisions are **scripted** in the test cells (`decisions=[…]`); `input()` stays available for a human driving it manually, but no demo cell calls it |
| Google Drive is opt-in only | `mount()` triggers an OAuth pop-up, which would stall Run all — so Drive is **off by default**, behind a single flag documented in the runbook cell |
| State lives on the Colab VM disk | SQLite file next to the notebook; survives kernel restarts inside the session, which is exactly what the durability demo needs |
| The only outbound network call is the Anthropic API | No LangSmith unless the user adds a key, no mermaid.ink dependency for the diagram (fallback already in place), no external data fetches |
| No secret other than `ANTHROPIC_API_KEY` is required | Everything else (case history, notifications, screening) is a deterministic in-notebook mock |

Everything labelled "production practice" below was chosen *because* it can be demonstrated
under those rules — not adapted to them afterwards.

---

## 1. Goal and scope

| In scope | Out of scope |
|---|---|
| Rebrand the scenario to **RevolutBank** | Real Revolut branding/logos (fictional bank, name only) |
| Add **2 new agents**: Triage Agent and Customer Comms Agent | Any agent needing a live external service |
| Propose **top 3 architectures**, pick one | Implementing more than the chosen one |
| Production practices runnable in Colab | Real infra (Postgres, Kafka, K8s, Vault) |
| Keep all assignment requirements green | Changing the submission format (still one `.ipynb`) |

The deliverable of the *migration itself* (a later step, not this document) remains:
`fraud_multi_agent.py` (source of truth) → `build_notebook.py` → `Fraud_Detection_Multi_Agent.ipynb`.

---

## 2. Rebranding to RevolutBank

Pure renaming pass — no behavioural change:

1. **Scenario cell**: "A payments company" → "**RevolutBank**, a digital-first retail bank.
   Its Fraud Operations desk receives signals: issuer fraud alerts, chargeback disputes,
   AML referrals, and customer-reported suspicious activity."
2. **Mock systems get RevolutBank names**, so the tools read like a real integration surface:
   - `fetch_customer_transactions` → docstring/description says "RevolutBank Core Banking API".
   - `check_sanctions_list` → "RevolutBank Screening Service (EU consolidated list)".
   - Synthetic customers stay the same six profiles (CUST-1001 … CUST-9999) — the test matrix
     in the README keeps working.
3. **System prompts** of all agents open with "You work in the Fraud Operations department
   of RevolutBank…" — consistent persona across agents.
4. **README.md** title, scenario section, and architecture diagram updated to the new names
   (the diagram itself is replaced by a Mermaid version — see §5.7).
5. A visible disclaimer in the notebook: *RevolutBank is a fictional bank for an educational
   project; no affiliation with Revolut Ltd.* (keeps the submission clean).

---

## 3. New agents (2) — roles, prompts, tools, state

### 3.1 Triage Agent (new entry point)

**Role**: first responder. Classifies the incoming natural-language signal, extracts the
customer id, assigns priority, and routes the case.

- **Input**: raw `user_request`.
- **Output** (structured, `with_structured_output`): `signal_type`
  (`FRAUD_ALERT | CHARGEBACK | AML_REFERRAL | CUSTOMER_REPORT | GENERAL_QUESTION`),
  `customer_id | None`, `priority` (`P1 | P2 | P3`), `routing` decision.
- **Replaces/absorbs** the current `intake` node's routing (no-customer / follow-up branches
  move under Triage), so the graph gains a *reasoning* front door instead of a parsing one.
- **New tool it uses**: `lookup_case_history(customer_id)` — mock case-management store
  (in-memory dict seeded per run + appended to by finished runs) returning prior cases for
  the customer; prior fraud cases raise priority.
- **Why it is production-realistic**: every bank fraud desk has a triage/first-line layer;
  it also gives the graph a natural place for input validation and prompt-injection screening
  before any privileged tool is touched.

### 3.2 Customer Comms Agent (new exit stage)

**Role**: after the human decision is final (approved & executed, or rejected/cancelled),
drafts the customer-facing notification.

- **Input**: final decision, risk assessment summary, customer profile.
- **Output** (structured): `channel` (`push | email | none`), `language`, `subject`, `body` —
  tone rules in the system prompt (no accusatory language, no internal rule names, no score
  disclosure; regulatory phrasing for blocks).
- **New tool it uses**: `send_customer_notification(customer_id, channel, subject, body)` —
  mock RevolutBank Notification API; validates channel, logs the payload to the audit trail,
  returns a delivery id. **Never called before the HITL gate has closed.**
- **HITL interplay**: the notification *draft* is included in the package the human reviews,
  so the existing single interrupt keeps gating every external side effect (block + message).
  A second interrupt is *not* added — one gate, richer payload (keeps the revision loop simple).

### 3.3 State changes (`FraudWorkflowState`)

Add fields: `signal_type`, `priority`, `case_history`, `notification_draft`,
`notification_result`, `case_id` (correlation id, see §5). Existing fields untouched, so
the current 85 tests keep their shape and only gain new assertions.

### 3.4 Tool count after migration

5 custom tools (assignment requires ≥ 2): the existing three + `lookup_case_history` +
`send_customer_notification`. All mock, all deterministic, all Colab-safe.

---

## 4. Top 3 architectures for organising 4 agents

### Option A — Deterministic staged pipeline with a triage router ✅ RECOMMENDED

```text
START ─► triage ──┬─ GENERAL_QUESTION / follow-up ─► followup_qa ─► END
                  ├─ no / unknown customer ─► customer_not_found ─► END
                  └─ case ─► fraud_analyst ─► compliance_officer ─► human_review
                                                       ▲               │ feedback
                                                       └───────────────┤
                                              approve │                │ reject
                                                      ▼                ▼
                                            execute_action      cancel_action
                                                      └───► customer_comms ───► END
```

- Agents are **fixed stages**; the only dynamic decisions are conditional edges computed
  from *structured state fields*, not free-text LLM routing.
- **Pros**: fully auditable (a regulator can replay why each edge fired), predictable cost
  (≤ 4–5 LLM calls per case + revisions), trivially testable with the existing scripted-fake-LLM
  harness, natural extension of the current graph — lowest-risk migration.
- **Cons**: adding a new case type means adding edges by hand; no dynamic re-planning.
- **Why it wins for a bank**: in regulated fraud/AML flows, *determinism and auditability beat
  flexibility*. An LLM choosing the route is a compliance liability; an LLM filling in
  structured fields that deterministic edges consume is not. It is also the best fit for
  Colab: one compiled graph, one checkpointer, no orchestration layer.

### Option B — Supervisor (hub-and-spoke)

A fifth LLM node (Supervisor) receives every intermediate result and decides which specialist
to invoke next (`langgraph-supervisor` pattern: agents as tools/handoffs).

- **Pros**: most flexible — new agents are just new tools of the supervisor; handles messy,
  novel requests without new edges; the fashionable "agentic" answer.
- **Cons**: +1 LLM call per hop (~2× cost per case), routing is a model decision → harder to
  audit and to test deterministically; the current scripted-LLM test harness would need
  supervisor-turn scripts; risk of routing loops needs its own guard.
- **When to choose it**: if RevolutBank's desk had 8–10 specialists with genuinely
  unpredictable case shapes. At 4 agents with a well-understood flow, it buys nothing but cost.

### Option C — Hierarchical teams (subgraphs)

Two LangGraph **subgraphs**, each a mini-pipeline, composed by a thin parent graph:
*Investigation team* (Triage → Fraud Analyst) and *Resolution team*
(Compliance Officer → HITL → execute/cancel → Customer Comms).

- **Pros**: strongest modular boundaries — each team compiles, checkpoints and tests in
  isolation; mirrors a real org chart; teams are reusable in other workflows.
- **Cons**: highest structural complexity for 4 agents; interrupt-inside-a-subgraph and
  cross-subgraph state mapping are the fiddliest parts of LangGraph — more ways to be subtly
  wrong in a notebook the graders run top-to-bottom.
- **When to choose it**: at 3+ teams or when teams evolve independently. Premature here.

### Decision

**Option A.** The plan documents B and C in the notebook's architecture cell (one paragraph
each + this trade-off table) so the submission *shows* the architectural reasoning — that
comparison itself is an enterprise practice (an ADR). Option A is recorded as
**ADR-001: deterministic routing over LLM routing in regulated flows**.

---

## 5. Production-ready practices — every one runnable in Colab

Each practice below names its Colab-compatible mechanism. Nothing needs a server.

### 5.1 Reliability
- **Retries with exponential backoff + jitter** on every Anthropic call: `ChatAnthropic`'s own
  `max_retries` for HTTP-level faults, plus a ~15-line stdlib `retry_call` helper (no
  `tenacity`, no new dependency) around tool and LLM invocations, with explicit handling of
  429 / 529 (rate limit / overloaded).
- **Timeouts** on every LLM call (`timeout=`) so a hung request cannot freeze Run all.
- **Cost circuit breaker**: `OBS` gains a per-notebook budget (`MAX_USD_PER_NOTEBOOK`);
  crossing it raises a clean `BudgetExceeded` before the next LLM call instead of silently
  burning quota. The demo cells cost cents, so Run all never trips it — it is the guard, not
  a behaviour of the happy path.
- **Revision guard** already exists (`MAX_REVISIONS = 3`) — kept, documented as a loop breaker.

### 5.2 Durable state (Colab flavour, fallback-safe)
- Swap `MemorySaver` → **`SqliteSaver`** on a file next to the notebook
  (`langgraph-checkpoint-sqlite`, added to the first-cell `pip install`; SQLite itself is
  Python stdlib, so nothing external is contacted). Interrupted HITL cases then survive a
  kernel restart, and the notebook demonstrates *resume-after-restart* by re-attaching to a
  `thread_id` in a dedicated cell.
- **Fallback is mandatory, not optional**: the import is wrapped, and if the wheel is
  unavailable for any reason the workflow silently falls back to `MemorySaver` and logs one
  line saying durability is reduced. A missing package must never break Run all (§0).
- **Google Drive stays off by default.** `mount()` opens an OAuth pop-up that would stall
  Run all, so cross-session durability is behind a single `USE_DRIVE = False` flag that the
  runbook cell explains. Nobody has to touch it to run the notebook.
- The production mapping is documented in one sentence (SQLite ↔ Postgres checkpointer share
  the same interface) — knowledge shown, infra not required.

### 5.3 Security & compliance
- **Secrets**: unchanged chain (Colab Secrets → `.env` → env vars); plan adds a *fail-fast
  preflight cell* that verifies the key exists and the model responds before any workflow runs.
- **PII minimisation in logs**: customer names masked (`Maria P.`), card/transaction ids
  truncated in every log line and in `OBS` events; full data lives only in state.
- **Prompt-injection hygiene**: tool outputs (transaction memos are attacker-controlled text
  in real life) are wrapped in delimiters and the analyst prompt instructs the model to treat
  them as data — plus one adversarial test case (a transaction description containing an
  instruction) proving the gate holds.
- **Append-only audit trail**: a JSONL file (`audit_log.jsonl`) records every state
  transition, tool call, human decision and notification payload with `case_id` + timestamp —
  the banking-grade artefact, implemented with 20 lines and zero infra.

### 5.4 Observability
- Keep `WorkflowObserver` / `setup_logging` as-is; add **`case_id` correlation** to every
  event and log line (uuid per `execute_workflow` call).
- `OBS.to_json()` per run + the audit JSONL = full replayability after the kernel dies.
- `enable_langsmith()` stays opt-in (works from Colab if the user adds a key; silent otherwise).

### 5.5 Quality gates
- **Existing 85 offline tests extended** for: triage classification routing, comms drafting,
  notification-never-before-approval invariant, budget breaker, PII masking, injection case.
- **Eval cell** in the notebook: a small labelled set (the 6 synthetic customers × expected
  recommendation) scored automatically — a mini regression eval that runs on every
  "Run all", i.e. CI inside Colab.
- **Model/prompt pinning**: `MODEL_NAME` pinned to an exact model id; prompts kept as
  versioned constants (`PROMPT_VERSION`) recorded in the audit trail — reproducibility.

### 5.6 Operability
- **Runbook cell** at the end: how to resume a stuck case, how to read the audit log, how to
  raise the budget, what each error class means.
- **Graceful degradation matrix** documented per dependency (Anthropic down → fail fast with
  case parked in checkpointer; sanctions mock error → case proceeds flagged "screening
  unavailable", never silently clean).

### 5.7 README.md refresh with a Mermaid architecture diagram

The README is part of the submission and must reflect the new system:

- **Replace the ASCII architecture diagram with a Mermaid `flowchart TD`** showing the full
  Option A graph: `triage` → branches (`followup_qa`, `customer_not_found`) →
  `fraud_analyst` → `compliance_officer` → `human_review` (interrupt) with the
  `feedback` loop-back edge, `approve` → `execute_action` and `reject` → `cancel_action`,
  both converging into `customer_comms` → `END`. Agent nodes, deterministic router nodes and
  the HITL gate get distinct styling (Mermaid `classDef`) so the three kinds of node are
  visually distinguishable; GitHub renders Mermaid natively, so the diagram is interactive
  in the repo without any tooling.
- The same Mermaid diagram goes into the notebook's architecture markdown cell. Caveat to
  verify during execution: classic Colab does not always render Mermaid in markdown cells —
  if it does not, the notebook cell keeps a fenced ` ```mermaid ` block (renders on GitHub
  and in VS Code) *plus* the ASCII fallback right below it, so no environment shows a blank.
- README content updates in the same pass: RevolutBank scenario wording, the four-agent
  table (role + tools per agent), the 5-tool table, the new test-case matrix (9 cases),
  the SqliteSaver/durability note, and the fictional-bank disclaimer.
- A second, smaller Mermaid `sequenceDiagram` in the README shows one BLOCK case end-to-end
  (signal → triage → analysis → compliance → human approve → block + notification) — this is
  the diagram graders read to understand HITL at a glance.

### 5.8 Assignment compliance check (unchanged or improved)

| Assignment requirement | After migration |
|---|---|
| ≥ 2 distinct agents | **4** (Triage, Fraud Analyst, Compliance Officer, Customer Comms) |
| Defined state (TypedDict) | `FraudWorkflowState` + 6 new fields |
| ≥ 2 tools | **5** custom tools |
| Memory (checkpointer) | SqliteSaver + follow-up QA thread |
| HITL interrupt + resume | unchanged single gate, richer review package |
| `execute_workflow(user_request)` | unchanged signature |
| ≥ 5 test cases | existing 7 + 2 new (injection attempt, durable interrupt) = **9** |
| Keys never in code | unchanged + preflight check |
| Runs in Colab top-to-bottom | hard constraint of this whole plan (§0), guarded by tests |

---

## 6. Migration phases — all delivered

Each phase ended green: the whole offline suite passing and the notebook regenerating.

1. **P1 — Rebrand ✅** RevolutBank naming across prompts, tool descriptions, the scenario
   cell and the fictional-bank disclaimer. No logic change; two new tests pin the persona
   and the tool descriptions so the rebrand cannot silently rot.
2. **P2 — State + tools ✅** Six new state fields; `lookup_case_history` and
   `send_customer_notification`; `audit()` (append-only JSONL) and `mask_name()` for PII
   minimisation. 9 new tests, written first.
3. **P3 — Triage Agent ✅** `triage` replaces `intake`: bounded tool loop over
   `lookup_case_history`, structured `TriageDecision`, deterministic id extraction, P1
   escalation on a recorded prior BLOCK, `case_id` assignment, and zero tokens spent when
   no customer is named. Introduced `run_tool_loop(pin_args=…)` — a data-access boundary
   that stops an agent reaching another customer's file — and `fence_tool_evidence()`.
   15 tests.
4. **P4 — Customer Comms Agent ✅** Split into `comms_draft` (before the gate — the human
   approves the exact words) and `comms_deliver` (after `execute_action` — replays the
   approved text verbatim, suppresses on reject, skips on channel `none`). The HITL payload
   now carries the draft. 10 unit tests + 2 integration tests for the invariant.
5. **P5 — Durability ✅** `make_checkpointer()` with SQLite on the runtime disk,
   `build_graph()` so the graph can be rebuilt, Drive off by default. 7 tests.
6. **P6 — Hardening ✅** `retry_call` (stdlib, no `tenacity`), transient/permanent
   classification, timeouts, `BudgetExceeded` circuit breaker, the `_Guarded` wrapper that
   puts both on the path every agent actually uses, `preflight()`, `PROMPT_VERSION` in the
   audit trail, and the `CUST-6006` injection profile. 15 tests, plus 10 in
   `test_colab_runall.py` that guard the §0 contract itself.
7. **P7 — Docs, evals & live run ✅** Mermaid `flowchart` + `sequenceDiagram` in the README
   and the notebook (all three blocks validated by rendering them with mermaid-cli), the
   regression eval cell, the audit-trail cell, the runbook with a per-dependency
   degradation matrix, the rewritten README, and a full live execution.

### What the live run exposed (and the plan did not predict)

Two defects that only a real end-to-end execution could surface. Both are fixed, and both
now have a test that fails without the fix:

- **`langgraph-checkpoint-sqlite` 2.x imports cleanly and then dies on the first checkpoint
  write** (`'JsonPlusSerializer' object has no attribute 'dumps'`), erroring *every*
  scenario cell. The guarded import was not enough, because the import succeeded. Fixes:
  pin `>=3.1,<4`, **and** `_probe_checkpointer()` performs a real write/read before the
  saver is trusted, falling back to `MemorySaver` otherwise.
- **A watch-listed customer came back CLEAR.** The analyst called
  `check_sanctions_list(customer_name="CUST-4444")` *before* fetching the profile, so
  RevolutBank screened a customer id instead of *Viktor Baranov*. Fixes: the analyst
  re-screens the name the core banking system returned (screening is executed, not
  recalled), `sanctions_match` was removed from the model's schema, the prompt now fixes
  the tool order, and a match overrides the recommendation to BLOCK deterministically with
  the override written into both the report and the audit trail.

The second one is the more instructive: the original design already said "the score is
computed, not generated", and the same reasoning simply had not been applied to screening.

## 7. Risks & Colab limitations (accepted)

- **Colab ephemerality**: the SQLite file dies with the VM unless Drive-mounted —
  documented, optional flag provided, off by default because mounting stalls Run all.
- **No true parallelism**: Option A is sequential, so the known per-node-attribution
  limitation of `WorkflowObserver` stays a non-issue (already documented).
- **Mock boundary honesty**: all "RevolutBank APIs" are mocks; their signatures are
  service-shaped, so swapping in real HTTP clients is a leaf change.
- **Cost**: measured at **$0.14 per full notebook run** (72 LLM calls, ≈87k tokens) on
  Haiku 4.5; the breaker caps a runaway at $2.00.
- **The comms agent may choose silence.** On CLEAR and MONITOR outcomes it consistently
  drafts `channel: "none"`, so only blocks generate a customer message. That is defensible
  — and for an AML case, tipping off is actually prohibited — but it is the agent's
  judgement, not a rule, and a bank would want it pinned down per outcome type.

## 8. What a real production deployment would still need

Out of scope by the §0 constraint, listed so the boundary is explicit rather than implied:
a Postgres checkpointer (same interface as the SQLite one), the mocks replaced by real
service clients behind timeouts and circuit breakers, the audit trail in an append-only
store rather than a file, secrets from a vault instead of Colab Secrets, OpenTelemetry
export instead of an in-process observer, a queue in front of the graph, and the eval suite
running in CI on every prompt change rather than on every Run all.

### 8.1 Carrying the idempotency contract to the real executor

`send_customer_notification` already takes an `idempotency_key` derived from durable state
(`case_id:notify:revision_count`) and returns the original delivery on a repeat, because
LangGraph re-runs a node from its top when a case resumes mid-node. Two things extend that
to production:

- **The real Notification API must own the dedupe**, not the caller. The key is passed to
  it for exactly that reason — a mapping held in the workflow process is lost with the
  process, which is the moment it was needed.
- **`execute_action` needs the same treatment.** In the notebook the "action" renders a
  string, so a replay is invisible beyond a duplicated `action_executed` audit line. The
  moment it calls a real card-management API, it needs a key of the same shape
  (`case_id:execute:revision_count`) and an executor that dedupes on it. Blocking a card
  twice is survivable; releasing and re-blocking, or double-charging a reversal, is not.

### 8.2 Mapping the observer onto OpenTelemetry

`WorkflowObserver` records the right events; what it lacks is a vendor-neutral way out of
the process. The GenAI semantic conventions map onto it almost one-to-one, so the exporter
is a translation layer rather than a rewrite:

| `OBS` event | OpenTelemetry |
|---|---|
| `run_start` / `run_end` | the root span of the trace, `thread_id` and `case_id` as attributes |
| `node_start` / `node_end` | a child span per agent, named for the node |
| `llm_call` | a `gen_ai.client.inference` span: `gen_ai.system`, `gen_ai.request.model`, and the usage counters (`gen_ai.usage.input_tokens`, `output_tokens`, cache reads) |
| `tool_call` / `tool_unavailable` | `gen_ai.tool.execution` spans, the tool name as an attribute |
| `retry` | a span event on the call that was retried |
| `interrupt` / `human_decision` | span events on the node span — the pair a compliance audit actually reads |
| `error` | span status `ERROR` plus the recorded exception |

Attribution is already held in a `ContextVar`, which is what OpenTelemetry's own context
propagation uses, so spans nest correctly across parallel branches without further work.
LangSmith stays useful alongside it for prompt-level debugging; OTel is what a bank's
platform team will ask for first.

### 8.3 An SLA on cases parked at the human gate

The state records `awaiting_review_since`, and `hours_awaiting_review()` reads it, but
nothing acts on it: a P1 that reaches the gate on a Friday evening is still sitting there on
Monday. Production needs a sweeper over the checkpoint store that escalates or re-notifies
past a per-priority threshold. The timestamp it needs already exists; the scheduler does not.
