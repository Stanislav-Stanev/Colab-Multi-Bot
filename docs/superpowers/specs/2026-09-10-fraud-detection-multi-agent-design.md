# Design: Transaction Fraud Analyst — Multi-Agent LangGraph Workflow

**Date:** 2026-09-10
**Course:** SoftUni — AI Agents and Workflows for Developers (April 2026), Individual Project
**Deliverable:** Google Colab / VS Code compatible Jupyter notebook (`.ipynb`) + `README.md` + `.env.example`, packaged as ZIP.

## 1. Scenario

A **Transaction Fraud Analyst** system for a payments company (myPOS-style domain).
The user asks in natural language, e.g. *"Check customer CUST-1042 for suspicious activity."*
The system:

1. Fetches the customer's recent card transactions (mock API over a built-in synthetic dataset).
2. Analyzes them for fraud patterns and computes a deterministic risk score.
3. Drafts a compliance report with a recommended action: **BLOCK / MONITOR / CLEAR**.
4. **Pauses for human approval** before executing the action (blocking a card is critical).
5. On approval executes the (mock) action; on feedback, revises the report and asks again; on reject, cancels the action.

## 2. Requirements Mapping (assignment rubric)

| Requirement | How it is satisfied |
|---|---|
| ≥ 2 distinct agents | `fraud_analyst` and `compliance_officer`, each with own system prompt & role |
| Defined shared state | Pydantic `FraudWorkflowState` passed through the graph |
| ≥ 2 tools used | 3 custom tools: `fetch_customer_transactions`, `calculate_risk_score`, `check_sanctions_list` |
| Conversational memory | `MemorySaver` checkpointer + `thread_id` per conversation |
| Human-in-the-loop | `interrupt()` in `human_review` node; resume via `Command(resume=...)` |
| `execute_workflow(user_request)` | Implemented, plus `resume_workflow(thread_id, decision)` helper |
| ≥ 5 test cases | 6 test cases incl. approve, feedback→revise, reject, memory follow-up |
| Secrets safety | `get_secret()` reads Colab Secrets → `.env` → `os.environ`; no keys in code |
| Runs cleanly | Single `%pip install` cell at top; no other setup needed |

## 3. Architecture (Approach A — approved)

Linear `StateGraph` pipeline with conditional edges and a revision loop:

```
intake → fraud_analyst → compliance_officer → human_review (interrupt)
                              ↑                     │
                              └── feedback ─────────┤
                                                    ├── approve → execute_action → END
                                                    └── reject  → cancel_action  → END
```

Rejected alternatives: **B** supervisor/router agent (extra complexity, no rubric value), **C** two prebuilt `create_react_agent`s (hides the explicit graph the assignment asks to demonstrate).

### State (Pydantic model)

- `user_request: str`
- `customer_id: str | None`
- `transactions: list[dict]`
- `risk_assessment: dict | None` — score 0–100, triggered flags, analyst notes
- `report: str | None` — compliance report draft (markdown)
- `recommended_action: Literal["BLOCK", "MONITOR", "CLEAR"] | None`
- `human_decision: dict | None` — `{type: approve|feedback|reject, feedback: str | None}`
- `revision_count: int`
- `final_output: str | None`
- `messages: Annotated[list, add_messages]` — conversation history for memory

### Agents

1. **Fraud Analyst** (LLM node with bound tools): system prompt of a senior payments fraud analyst;
   must call the tools, then emit a structured `AnalystNarrative`. The risk score and triggered
   rules are taken from the scoring engine's own output rather than from the model, so the number
   that drives the decision is computed; the model contributes the prose and the sanctions verdict.
2. **Compliance Officer** (LLM node): writes the compliance report + recommendation from the analyst's assessment; on human feedback revises the report (revision loop, `revision_count` guard ≤ 3).

### Tools (all self-contained, no external APIs beyond Anthropic)

1. `fetch_customer_transactions(customer_id: str)` — mock core-banking API over a seeded synthetic dataset (~6 customer profiles: clean, card-testing burst, velocity anomaly, geo anomaly, high-amount outlier, unknown customer → error path).
2. `calculate_risk_score(transactions_json: str)` — deterministic scoring: velocity, geo spread, amount z-scores, risky MCCs, night-time activity. Returns score + triggered rules.
3. `check_sanctions_list(name: str)` — mock sanctions/watchlist lookup.

### HITL mechanics

Routing after `intake`: a named customer starts an investigation; a request with no customer id
on a thread that already holds a conversation is a follow-up answered from memory
(`followup_qa`); on a fresh thread it goes to `no_customer`, which asks for an id without
spending a token. `human_review` node calls `interrupt(payload)` with the report and recommended action. The caller receives `__interrupt__`, shows the report, collects a decision, and resumes with `Command(resume={"type": ..., "feedback": ...})`. Conditional edge routes to `execute_action` / `compliance_officer` / `cancel_action`.

### Core functions

- `execute_workflow(user_request: str, *, decisions=None, thread_id=None) -> dict` — the required
  entry point. Starts the graph and owns the whole HITL loop: on each interruption it shows the
  report, obtains the human's decision (interactively via `input()`, or from a scripted
  `decisions` list), resumes with `Command(resume=...)`, and repeats until the graph returns a
  final output. The extra parameters are keyword-only with defaults, so the assignment's
  `execute_workflow(user_request)` call works exactly as specified.
- `start_workflow(user_request, thread_id=None)` / `resume_workflow(thread_id, decision)` — the
  low-level halves, exposed so the notebook can show the pause explicitly.
- `run_scenario(title, user_request, decisions)` — banner + `execute_workflow`, for the demo cells.

## 4. Environment compatibility (Colab + VS Code)

- `get_secret(name)`: try `google.colab.userdata.get(name)` → fall back to `dotenv` `.env` → `os.environ`. Raises a clear message telling the user where to put the key in each environment.
- `%pip install -qU langgraph langchain langchain-anthropic python-dotenv` as the first code cell (works in both environments).
- Model: `claude-opus-5` via `langchain-anthropic` (configurable constant at the top).
- No other environment-specific code anywhere.

## 5. Notebook structure

1. Markdown: title, scenario description, architecture diagram (mermaid), rubric mapping.
2. `%pip install` cell.
3. Secrets/config cell (`get_secret`, model constant).
4. Synthetic dataset + tools.
5. State model.
6. Agent prompts + nodes.
7. Graph assembly + compile with `MemorySaver`; render graph image.
8. `execute_workflow` / `resume_workflow` / `run_scenario`.
9. Test cases 1–6 with printed agent traces.
10. Markdown: conclusions.

## 6. Test cases

Scores below are what the rule engine actually computes, so each scripted human decision reads
coherently against the report it responds to.

1. **CUST-1001**, score 0 → CLEAR, human approves.
2. **CUST-1042**, score 100 (card testing) → BLOCK, human **approves** → card blocked.
3. **CUST-1337**, score 40 (velocity) → MONITOR, human gives **feedback** that monitoring is too
   lenient → report revised upward → approve.
4. **CUST-2077**, score 40 (geo anomaly + crypto/gambling MCCs) → MONITOR, human **rejects**.
5. **CUST-9999** → unknown customer, graceful error handling.
6. **CUST-4444**, score 15 but a **sanctions match** → BLOCK, showing a rule that overrides the score.
7. **Memory test** — follow-up question in Test 2's `thread_id` answered from checkpointed history.

## 7. Error handling

- Unknown customer id → tool returns structured error; analyst reports honestly.
- LLM structured-output failure → one retry, then fail with readable message.
- Revision loop guard: max 3 revisions, then force final decision.
- Missing API key → actionable message naming both Colab Secrets and `.env` paths.

## 8. Repo layout

```
Colab-Multi-Bot/
├── Fraud_Detection_Multi_Agent.ipynb
├── README.md
├── .env.example
├── .gitignore            (.env excluded)
└── docs/superpowers/specs/…this file…
```
