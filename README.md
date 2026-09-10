# 🕵️ Transaction Fraud Analyst — Multi-Agent LangGraph Workflow

**SoftUni · AI Agents and Workflows for Developers — Individual Project**

A two-agent LangGraph system that reviews a payment customer for card fraud and **pauses for
human approval** before taking the critical action (blocking a card).

## Scenario

A payments company receives natural-language requests such as
*"Investigate CUST-1042 — we received a fraud alert from the issuer."*

1. **🔎 Fraud Analyst** — fetches the customer's card transactions from the (mock) core banking
   system, runs a deterministic risk-scoring engine, and checks the name against a sanctions
   list. Emits a structured `RiskAssessment` (score 0–100, triggered rules, analyst notes).
2. **🧑‍⚖️ Compliance Officer** — turns that assessment into a compliance report and recommends
   exactly one action: **BLOCK / MONITOR / CLEAR**.
3. **⏸️ Human review** — the graph interrupts. A human can **approve** (action executed),
   send **feedback** (the officer rewrites the report and the graph asks again), or
   **reject** (action cancelled, report archived).

## Architecture

```
                        ┌──────────────── feedback ────────────────┐
                        ▼                                          │
START ─► intake ─► fraud_analyst ─► compliance_officer ─► human_review ─ approve ─► execute_action ─► END
           │                                              (interrupt) │
           │                                                          └─ reject ──► cancel_action ──► END
           └─ no customer id + prior history ─► followup_qa ─► END
```

* **State:** `FraudWorkflowState` (`TypedDict`) carries the request, customer id, transactions,
  risk assessment, report, recommendation, human decision, revision count and the message
  history (`add_messages` reducer).
* **Memory:** `MemorySaver` checkpointer keyed by `thread_id` — follow-up questions in the same
  thread are answered from the checkpointed conversation (`followup_qa` node).
* **HITL:** `interrupt()` inside `human_review`, resumed with `Command(resume=decision)`.
* **Revision guard:** at most `MAX_REVISIONS = 3` feedback rounds, then the action is finalised.

## Tools (3 custom tools)

| Tool | What it does |
|---|---|
| `fetch_customer_transactions(customer_id)` | Mock core-banking API over a deterministic synthetic dataset of 6 customer profiles; returns a structured error for unknown ids. |
| `calculate_risk_score(transactions_json)` | Deterministic rule engine: card testing, velocity, geo anomaly, amount outlier, risky MCCs (gambling / crypto / money transfer), night-time activity. Returns score 0–100 + triggered rules. |
| `check_sanctions_list(customer_name)` | Mock EU-consolidated sanctions/watch-list lookup. |

## Test cases (section 8 of the notebook)

| # | Customer | Pattern | Human decision | Demonstrates |
|---|---|---|---|---|
| 1 | CUST-1001 | clean history | approve | happy path, CLEAR |
| 2 | CUST-1042 | card testing (micro-payments then a large one) | approve | BLOCK executed |
| 3 | CUST-1337 | velocity (6 payments in 8 minutes) | **feedback** → approve | report revision loop |
| 4 | CUST-2077 | geo anomaly + gambling/crypto MCCs | **reject** | action cancelled |
| 5 | CUST-9999 | unknown customer | approve | graceful error handling |
| 6 | — | follow-up in Test 2's thread | — | conversational memory |

## Running it

### Google Colab

1. Upload `Fraud_Detection_Multi_Agent.ipynb` (File → Upload notebook).
2. Open the 🔑 **Secrets** panel, add `ANTHROPIC_API_KEY`, and enable *Notebook access*.
3. **Runtime → Run all.** The first cell installs the dependencies; nothing else is needed.

### VS Code / local Jupyter

```bash
pip install -r requirements-dev.txt
cp .env.example .env        # then put your real key in .env
```

Then either open `Fraud_Detection_Multi_Agent.ipynb` and run all cells, or open
`fraud_multi_agent.py` — it is the same content in jupytext *percent* format, so VS Code
renders it as interactive cells (`# %%`) and you can run it cell by cell without Jupyter.

`get_secret()` tries Colab Secrets first, then `.env`, then plain environment variables, so
the identical notebook runs unmodified in both environments. **No key is ever written in code.**

## Tests

35 unit and integration tests run without an API key — the graph is exercised end to end
against a scripted fake LLM (real `interrupt()`, real checkpointer, real routing):

```bash
python -m pytest tests/ -v
```

## Regenerating the notebook

`fraud_multi_agent.py` is the source of truth; the notebook is generated from it:

```bash
python -m jupytext --to ipynb --set-kernel python3 fraud_multi_agent.py -o Fraud_Detection_Multi_Agent.ipynb
```

## Project layout

```
Fraud_Detection_Multi_Agent.ipynb   the submission notebook (generated)
fraud_multi_agent.py                same code as a VS Code / jupytext script
tests/                              35 tests (tools, rules, routers, full graph runs)
.env.example                        template for the local API key
requirements-dev.txt                dependencies for local development
docs/superpowers/                   design spec and implementation plan
```
