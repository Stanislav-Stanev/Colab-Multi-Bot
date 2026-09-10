# 🕵️ Transaction Fraud Analyst — Multi-Agent LangGraph Workflow

**SoftUni · AI Agents and Workflows for Developers — Individual Project**

A two-agent LangGraph system that reviews a payment customer for card fraud and **pauses for
human approval** before taking the critical action (blocking a card).

## Scenario

A payments company receives natural-language requests such as
*"Investigate CUST-1042 — we received a fraud alert from the issuer."*

1. **🔎 Fraud Analyst** — fetches the customer's card transactions from the (mock) core banking
   system, runs a deterministic risk-scoring engine, and checks the name against a sanctions
   list. The score and triggered rules come from the engine; the model writes the narrative.
2. **🧑‍⚖️ Compliance Officer** — turns that assessment into a compliance report and recommends
   exactly one action: **BLOCK / MONITOR / CLEAR**.
3. **⏸️ Human review** — the graph interrupts. A human can **approve** (action executed),
   send **feedback** (the officer rewrites the report and the graph asks again), or
   **reject** (action cancelled, report archived).

## Architecture

```text
                        ┌──────────────── feedback ────────────────┐
                        ▼                                          │
START ─► intake ─► fraud_analyst ─► compliance_officer ─► human_review ─ approve ─► execute_action ─► END
           │             │                                (interrupt) │
           │             │                                            └─ reject ──► cancel_action ──► END
           │             └─ customer not in the system ─► customer_not_found ─► END
           ├─ no customer id, prior history ─► followup_qa ─► END
           └─ no customer id, fresh thread  ─► no_customer ─► END
```

* **State:** `FraudWorkflowState` (`TypedDict`) carries the request, customer id, transactions,
  risk assessment, report, recommendation, human decision, revision count and the message
  history (`add_messages` reducer).
* **Memory:** `MemorySaver` checkpointer keyed by `thread_id` — follow-up questions in the same
  thread are answered from the checkpointed conversation (`followup_qa` node).
* **HITL:** `interrupt()` inside `human_review`, resumed with `Command(resume=decision)`.
  `execute_workflow` owns that loop: it pauses, collects the decision, resumes, and repeats
  until the graph produces a final output.
* **Revision guard:** at most `MAX_REVISIONS = 3` feedback rounds, then the action is finalised.
* **No action it cannot justify.** Two branches exist so the graph never reaches the human
  gate empty-handed: a request naming nobody, and a customer the core banking system has
  never heard of — the latter stops at `customer_not_found` rather than proposing to block
  a card that does not exist.
* **The score is computed, not generated.** `fraud_analyst` writes `_score()`'s output into
  the state and lets the model contribute only the narrative, so the number that drives
  BLOCK / MONITOR / CLEAR can never be an LLM transcription error.

## Tools (3 custom tools)

| Tool | What it does |
|---|---|
| `fetch_customer_transactions(customer_id)` | Mock core-banking API over a deterministic synthetic dataset of 6 customer profiles; returns a structured error for unknown ids. |
| `calculate_risk_score(transactions_json)` | Deterministic rule engine: card testing, velocity, geo anomaly, amount outlier, risky MCCs (gambling / crypto / money transfer), night-time activity. Returns score 0–100 + triggered rules. |
| `check_sanctions_list(customer_name)` | Mock EU-consolidated sanctions/watch-list lookup. |

## Test cases (section 8 of the notebook)

| # | Customer | Score → recommendation | Human decision | Demonstrates |
|---|---|---|---|---|
| 1 | CUST-1001 | 0 → CLEAR | approve | happy path |
| 2 | CUST-1042 | 100 → BLOCK (card testing) | approve | critical action executed |
| 3 | CUST-1337 | 40 → MONITOR (velocity) | **feedback** → approve | revision loop, human escalates |
| 4 | CUST-2077 | 40 → MONITOR (geo + crypto/gambling) | **reject** | action cancelled |
| 5 | CUST-9999 | unknown customer | *none needed* | stops before any action |
| 6 | CUST-4444 | 15, but sanctions hit → BLOCK | approve | a rule overriding the score |
| 7 | — | follow-up in Test 2's thread | — | conversational memory |

## The core function

```python
execute_workflow(user_request: str, *, decisions=None, thread_id=None) -> dict
```

One string in, the final output out. It initializes the graph, and on every human-in-the-loop
pause it shows the report, collects the reviewer's decision, and resumes the graph with it —
looping until the workflow completes. By default it asks the reviewer through `input()`;
the notebook's test cases pass a scripted `decisions` list so the whole notebook runs
top to bottom unattended. `start_workflow` / `resume_workflow` are the low-level halves
underneath, if you want to drive the pause yourself.

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

## Model configuration and cost

Two constants at the top of the notebook pick the model:

```python
MODEL_NAME = "claude-haiku-4-5"   # cheapest; "claude-sonnet-5" / "claude-opus-5" reason deeper
EFFORT     = None                 # None on Haiku; "low".."max" on Sonnet 5 / Opus 5
```

| Model | USD / 1M in | USD / 1M out | Reasoning effort |
|---|---|---|---|
| `claude-haiku-4-5` | $1 | $5 | not supported — must be `None` |
| `claude-sonnet-5` | $2 | $10 | `low` … `max` |
| `claude-opus-5` | $5 | $25 | `low` … `max` |

A `UsageTracker` callback counts every agent call, and the last cell prints the run's
token usage and its cost, so switching models shows its price immediately.

Three details of the current Claude models are handled in the code, each of which
otherwise fails only at runtime with a live key:

* **No `temperature`** — sampling parameters were removed on the Claude 5 models and the
  API rejects them with a 400. `langchain-anthropic` strips them only for `claude-fable-5`,
  so they would otherwise reach the API.
* **`effort` is omitted entirely unless set** — Haiku 4.5 answers a request carrying the
  effort parameter with `400 This model does not support the effort parameter`.
* **Native structured outputs** — the agents use
  `with_structured_output(schema, method="json_schema")` rather than LangChain's default
  forced tool calling, which conflicts with adaptive thinking.

## Tests

62 unit and integration tests run without an API key — the graph is exercised end to end
against a scripted fake LLM (real `interrupt()`, real checkpointer, real routing):

```bash
python -m pytest tests/ -v
```

## Regenerating the notebook

`fraud_multi_agent.py` is the source of truth; the notebook is generated from it:

```bash
python build_notebook.py          # regenerate
python build_notebook.py --run    # regenerate, then execute with a live key
```

Use that script rather than calling `jupytext` directly. Both jupytext and nbclient read
and write the notebook in the process's locale encoding, so on a Windows console set to a
legacy code page (cp1251, cp1252, …) every emoji in the notebook is silently rewritten as
mojibake — `📥` becomes `рџ“Ґ` — and a later `jupyter execute` fails outright with
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x98`. The script forces UTF-8 for
the child processes and refuses to leave a corrupted notebook behind; four tests in
`tests/test_notebook_encoding.py` fail if one slips through anyway.

## Project layout

```text
Fraud_Detection_Multi_Agent.ipynb   the submission notebook (generated)
fraud_multi_agent.py                same code as a VS Code / jupytext script
build_notebook.py                   regenerates the notebook with UTF-8 enforced
tests/                              62 tests (tools, rules, routers, request shape, graph, encoding)
.env.example                        template for the local API key
requirements-dev.txt                dependencies for local development
docs/superpowers/                   design spec and implementation plan
```
