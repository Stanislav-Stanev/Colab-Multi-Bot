# %% [markdown]
# # 🕵️ Transaction Fraud Analyst — Multi-Agent LangGraph Workflow
# **SoftUni · AI Agents and Workflows for Developers — Individual Project**
#
# ## Scenario
# A payments company receives natural-language requests like *"Check customer CUST-1042
# for suspicious activity."* Two AI agents collaborate:
#
# 1. **🔎 Fraud Analyst** — pulls the customer's transactions (mock core-banking API),
#    runs a deterministic risk-scoring engine and a sanctions check, and produces a
#    structured risk assessment.
# 2. **🧑‍⚖️ Compliance Officer** — turns the assessment into a compliance report and
#    recommends an action: **BLOCK / MONITOR / CLEAR**.
#
# Blocking a card is a critical action, so the graph **pauses (human-in-the-loop)**
# before executing it: a human can **approve**, give **feedback** (report is revised),
# or **reject** (action cancelled). Conversation memory (`MemorySaver`) lets follow-up
# questions in the same thread reuse the analysis.
#
# Works in **Google Colab** (secrets via 🔑 Colab Secrets) and **VS Code / local Jupyter**
# (secrets via `.env`).
#
# | Assignment requirement | Where it is satisfied |
# |---|---|
# | ≥ 2 distinct agents | `fraud_analyst`, `compliance_officer` (sections 4) |
# | Defined shared state | `FraudWorkflowState` (section 3) |
# | ≥ 2 tools | 3 custom tools (section 2) |
# | Conversational memory | `MemorySaver` + `thread_id` (sections 6, 8/Test 6) |
# | Human-in-the-loop | `interrupt()` in `human_review` (section 5) |
# | `execute_workflow(user_request)` | Section 7 |
# | ≥ 5 test cases | 6 scenarios (section 8) |
# | No API keys in code | `get_secret()` (section 1) |

# %%
# %pip install -qU langgraph langchain langchain-anthropic python-dotenv

# %% [markdown]
# ## 1. Configuration & secrets
# `get_secret()` looks in Colab Secrets first, then `.env` / environment variables —
# so the same notebook runs unmodified in Colab and VS Code. **No keys in code!**

# %%
import os

MODEL_NAME = "claude-sonnet-5"
SKIP_DEMOS = os.getenv("FRAUD_SKIP_DEMOS") == "1"   # lets automated tests import this file


def get_secret(name: str) -> str:
    """Fetch a secret: Colab Secrets -> .env file -> environment variable."""
    try:                                   # 1) Google Colab
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    try:                                   # 2) .env (VS Code / local)
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    value = os.getenv(name)                # 3) plain environment variable
    if value:
        return value
    raise RuntimeError(
        f"Secret '{name}' not found. In Google Colab: add it via the 🔑 Secrets panel "
        f"and enable notebook access. In VS Code/local: copy .env.example to .env and set it."
    )


_llm_cache = {}


def get_llm(temperature: float = 0.0):
    """Lazily build the ChatAnthropic client (so importing this file needs no API key)."""
    if temperature not in _llm_cache:
        from langchain_anthropic import ChatAnthropic
        _llm_cache[temperature] = ChatAnthropic(
            model=MODEL_NAME, temperature=temperature, max_tokens=4096,
            api_key=get_secret("ANTHROPIC_API_KEY"),
        )
    return _llm_cache[temperature]

# %% [markdown]
# ## 2. Mock data & custom tools
# Three **custom tools** (assignment requires ≥2). The dataset is synthetic and
# deterministic — six customer profiles covering clean behaviour and classic fraud
# patterns (card testing, velocity, geo anomaly, amount outlier).

# %%
import json
from datetime import datetime
from statistics import median
from langchain_core.tools import tool


def _tx(ts, amount, merchant, mcc, country, present=False):
    return {"timestamp": ts, "amount_eur": amount, "merchant": merchant,
            "mcc": mcc, "country": country, "card_present": present}


CUSTOMER_DB = {
    "CUST-1001": {  # clean profile: regular groceries/fuel in one country
        "name": "Maria Petrova", "home_country": "BG",
        "transactions": [
            _tx("2026-09-01T09:15:00", 42.30, "Fantastico Sofia", "5411", "BG", True),
            _tx("2026-09-02T18:40:00", 61.00, "OMV Fuel", "5541", "BG", True),
            _tx("2026-09-04T12:05:00", 18.90, "Cinema City", "7832", "BG", True),
            _tx("2026-09-06T10:22:00", 55.75, "Kaufland", "5411", "BG", True),
            _tx("2026-09-08T20:10:00", 34.60, "Happy Restaurant", "5812", "BG", True),
        ]},
    "CUST-1042": {  # card testing: burst of tiny online payments, then a big one
        "name": "Georgi Ivanov", "home_country": "BG",
        "transactions": [
            _tx("2026-09-08T03:02:00", 1.00, "onlinestore-a.com", "5999", "US"),
            _tx("2026-09-08T03:04:00", 1.10, "onlinestore-b.net", "5999", "US"),
            _tx("2026-09-08T03:05:30", 0.99, "digigoods.io", "5999", "GB"),
            _tx("2026-09-08T03:07:00", 1.50, "webshop-c.com", "5999", "US"),
            _tx("2026-09-08T03:09:00", 1.20, "quickpay-d.com", "5999", "US"),
            _tx("2026-09-08T03:30:00", 890.00, "lux-electronics.com", "5732", "US"),
        ]},
    "CUST-1337": {  # velocity: many mid-size payments within minutes
        "name": "Ivan Dimitrov", "home_country": "BG",
        "transactions": [
            _tx("2026-09-07T14:00:00", 120.00, "TechMart", "5732", "BG"),
            _tx("2026-09-07T14:02:00", 240.00, "TechMart", "5732", "BG"),
            _tx("2026-09-07T14:03:00", 180.00, "GadgetWorld", "5732", "BG"),
            _tx("2026-09-07T14:05:00", 310.00, "TechMart", "5732", "BG"),
            _tx("2026-09-07T14:06:00", 150.00, "GadgetWorld", "5732", "BG"),
            _tx("2026-09-07T14:08:00", 275.00, "TechMart", "5732", "BG"),
        ]},
    "CUST-2077": {  # geo anomaly: 4 countries in a few hours + gambling MCC
        "name": "Petar Stoyanov", "home_country": "BG",
        "transactions": [
            _tx("2026-09-08T08:00:00", 95.00, "Cafe Sofia", "5812", "BG", True),
            _tx("2026-09-08T10:30:00", 430.00, "e-casino.bet", "7995", "MT"),
            _tx("2026-09-08T12:45:00", 610.00, "luxwatches.hk", "5944", "HK"),
            _tx("2026-09-08T15:20:00", 380.00, "crypto-exchange.io", "6051", "SC"),
        ]},
    "CUST-3050": {  # amount outlier: normal history + one huge night-time payment
        "name": "Elena Georgieva", "home_country": "BG",
        "transactions": [
            _tx("2026-09-03T11:00:00", 25.00, "BooksBG", "5942", "BG", True),
            _tx("2026-09-04T13:30:00", 40.00, "Pharmacy 36.6", "5912", "BG", True),
            _tx("2026-09-05T17:15:00", 33.50, "H&M Sofia", "5651", "BG", True),
            _tx("2026-09-09T03:40:00", 2900.00, "goldenjewels-online.com", "5944", "TR"),
        ]},
    "CUST-4444": {  # sanctions-flagged name, otherwise quiet account
        "name": "Viktor Baranov", "home_country": "BG",
        "transactions": [
            _tx("2026-09-02T10:00:00", 500.00, "Wire-in remittance", "4829", "CY"),
            _tx("2026-09-06T10:00:00", 500.00, "Wire-in remittance", "4829", "CY"),
        ]},
}

SANCTIONS_LIST = {"viktor baranov", "acme offshore ltd", "ivan orlov"}
RISKY_MCC = {"7995": "gambling", "6051": "crypto", "4829": "money transfer"}


def _fetch(customer_id: str) -> dict:
    cust = CUSTOMER_DB.get(customer_id.strip().upper())
    if not cust:
        return {"error": f"Customer '{customer_id}' not found in the core banking system.",
                "known_ids": sorted(CUSTOMER_DB)}
    return {"customer": {"customer_id": customer_id.strip().upper(),
                         "name": cust["name"], "home_country": cust["home_country"]},
            "transactions": cust["transactions"]}


def _score(transactions: list) -> dict:
    """Deterministic rule-based risk scoring: returns score 0-100 + triggered rules."""
    if not transactions:
        return {"risk_score": 0, "triggered_rules": [], "summary": "No transactions."}
    rules, score = [], 0
    txs = sorted(transactions, key=lambda t: t["timestamp"])
    times = [datetime.fromisoformat(t["timestamp"]) for t in txs]
    amounts = [t["amount_eur"] for t in txs]

    # card testing: >=4 payments under 2 EUR within one hour
    small = [dt for dt, t in zip(times, txs) if t["amount_eur"] < 2.0]
    if len(small) >= 4 and (small[-1] - small[0]).total_seconds() <= 3600:
        rules.append("card_testing"); score += 35
    # velocity: >=5 transactions inside any 10-minute window
    for i in range(len(times)):
        if sum(1 for t2 in times if 0 <= (t2 - times[i]).total_seconds() <= 600) >= 5:
            rules.append("velocity"); score += 30
            break
    # geo anomaly: >=3 countries within the observed period
    if len({t["country"] for t in txs}) >= 3:
        rules.append("geo_anomaly"); score += 25
    # amount outlier: any amount > 10x median
    med = median(amounts)
    if med > 0 and max(amounts) > 10 * med:
        rules.append("amount_outlier"); score += 20
    # risky MCCs
    hit_mccs = sorted({RISKY_MCC[t["mcc"]] for t in txs if t["mcc"] in RISKY_MCC})
    if hit_mccs:
        rules.append("risky_mcc:" + ",".join(hit_mccs)); score += 15
    # night-time activity (02:00-05:59)
    if sum(1 for dt in times if 2 <= dt.hour < 6) >= 2:
        rules.append("night_activity"); score += 10

    score = min(score, 100)
    return {"risk_score": score, "triggered_rules": rules,
            "summary": f"{len(txs)} transactions, total {sum(amounts):.2f} EUR, "
                       f"score {score}/100, rules: {', '.join(rules) or 'none'}"}


@tool
def fetch_customer_transactions(customer_id: str) -> str:
    """Fetch a customer's profile and recent card transactions from the core banking system.
    Argument: customer_id like 'CUST-1042'. Returns JSON with customer info and transactions,
    or an error object if the customer does not exist."""
    return json.dumps(_fetch(customer_id))


@tool
def calculate_risk_score(transactions_json: str) -> str:
    """Run the deterministic fraud risk-scoring engine over a JSON array of transactions
    (as returned by fetch_customer_transactions). Returns JSON: risk_score 0-100,
    triggered_rules list, and a short summary."""
    try:
        data = json.loads(transactions_json)
        txs = data["transactions"] if isinstance(data, dict) else data
        return json.dumps(_score(txs))
    except Exception as e:
        return json.dumps({"error": f"Could not parse transactions: {e}"})


@tool
def check_sanctions_list(customer_name: str) -> str:
    """Check a customer name against the (mock) international sanctions/watch list.
    Returns JSON {match: bool, list: 'EU-consolidated-mock'}."""
    return json.dumps({"match": customer_name.strip().lower() in SANCTIONS_LIST,
                       "list": "EU-consolidated-mock"})


TOOLS = [fetch_customer_transactions, calculate_risk_score, check_sanctions_list]

# %% [markdown]
# ## 3. Shared workflow state
# A typed state object is passed between the agents (assignment requirement).
# `messages` uses the `add_messages` reducer so conversation history accumulates
# and is checkpointed by `MemorySaver` — this is the workflow's memory.

# %%
import re
from typing import Annotated, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class FraudWorkflowState(TypedDict):
    user_request: str
    customer_id: Optional[str]
    transactions: list
    risk_assessment: Optional[dict]
    report: Optional[str]
    recommended_action: Optional[str]      # BLOCK | MONITOR | CLEAR
    human_decision: Optional[dict]         # {"type": approve|feedback|reject, "feedback": str|None}
    revision_count: int
    final_output: Optional[str]
    messages: Annotated[list, add_messages]


class RiskAssessment(BaseModel):
    """Structured output of the Fraud Analyst agent."""
    risk_score: int = Field(ge=0, le=100)
    triggered_rules: list[str]
    sanctions_match: bool
    analyst_notes: str = Field(description="2-4 sentences explaining the patterns found")


class ComplianceReport(BaseModel):
    """Structured output of the Compliance Officer agent."""
    report_markdown: str = Field(description="Full compliance report in markdown")
    recommended_action: Literal["BLOCK", "MONITOR", "CLEAR"]
    justification: str

# %% [markdown]
# ## 4. The two agents (nodes)
# * **Fraud Analyst** runs a ReAct-style tool loop (it must actually *call* the tools),
#   then emits a structured `RiskAssessment`.
# * **Compliance Officer** writes the report; when the human sends feedback, the same
#   node runs again in *revision mode*.

# %%
FRAUD_ANALYST_PROMPT = """You are a senior fraud analyst at a European payments company.
Investigate the customer mentioned in the request. You MUST:
1. Call fetch_customer_transactions with the customer id.
2. Call calculate_risk_score on the returned transactions JSON.
3. Call check_sanctions_list with the customer's name.
If the customer does not exist, report that honestly instead of inventing data.
Be factual and concise; never exaggerate risk beyond what the tools show."""

COMPLIANCE_OFFICER_PROMPT = """You are a compliance officer at a European payments company.
Based on the fraud analyst's structured assessment, write a short professional compliance
report (markdown: Summary, Findings, Risk score, Recommendation) and recommend exactly one
action: BLOCK (score >= 70 or sanctions match), MONITOR (40-69), CLEAR (< 40).
You may deviate from these bands only with explicit justification."""


def _extract_customer_id(text: str) -> Optional[str]:
    match = re.search(r"cust-\d+", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def intake(state: FraudWorkflowState) -> dict:
    cid = _extract_customer_id(state["user_request"])
    print(f"📥 intake: request={state['user_request']!r} → customer_id={cid}")
    return {"customer_id": cid, "revision_count": state.get("revision_count") or 0,
            "messages": [HumanMessage(content=state["user_request"])]}


def route_after_intake(state) -> str:
    # No customer id but existing history -> it's a follow-up question over memory.
    if not state.get("customer_id") and state.get("messages"):
        return "followup_qa"
    return "fraud_analyst"


def fraud_analyst(state: FraudWorkflowState) -> dict:
    print("🔎 Fraud Analyst: investigating...")
    llm = get_llm().bind_tools(TOOLS)
    msgs = [SystemMessage(content=FRAUD_ANALYST_PROMPT),
            HumanMessage(content=state["user_request"])]
    tools_by_name = {t.name: t for t in TOOLS}
    transactions, evidence = [], []
    for _ in range(6):                                   # bounded ReAct loop
        ai = llm.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            print(f"   🛠️ tool call: {tc['name']}({json.dumps(tc['args'])[:120]})")
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            evidence.append(f"{tc['name']}({json.dumps(tc['args'])}) -> {result}")
            if tc["name"] == "fetch_customer_transactions":
                transactions = json.loads(result).get("transactions", [])
            msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Summarise the tool evidence as plain text: the structured-output call then runs on a
    # clean context, instead of replaying tool_use blocks the schema-only request can't resolve.
    summary = "\n".join(evidence) or "No tool evidence was collected."
    assessment = get_llm().with_structured_output(RiskAssessment).invoke([
        SystemMessage(content=FRAUD_ANALYST_PROMPT),
        HumanMessage(content=f"Original request: {state['user_request']}\n\n"
                             f"Tool evidence collected:\n{summary}\n\n"
                             f"Produce your final structured risk assessment based strictly "
                             f"on this evidence. If the customer was not found, set the score "
                             f"to 0 and say so in the notes."),
    ])
    print(f"   ✅ assessment: score={assessment.risk_score}, rules={assessment.triggered_rules}, "
          f"sanctions={assessment.sanctions_match}")
    return {"transactions": transactions,
            "risk_assessment": assessment.model_dump(),
            "messages": [AIMessage(content=f"[Fraud Analyst] {assessment.analyst_notes} "
                                           f"(score {assessment.risk_score}/100)")]}


def compliance_officer(state: FraudWorkflowState) -> dict:
    revision = state.get("human_decision") or {}
    feedback = revision.get("feedback") if revision.get("type") == "feedback" else None
    print("🧑‍⚖️ Compliance Officer:", "revising report after human feedback..." if feedback
          else "drafting report...")
    content = (f"Fraud analyst assessment (JSON): {json.dumps(state['risk_assessment'])}\n"
               f"Customer: {state.get('customer_id')}")
    if feedback:
        content += (f"\n\nPrevious report:\n{state['report']}\n\n"
                    f"HUMAN REVIEWER FEEDBACK (you must address it): {feedback}")
    result = get_llm().with_structured_output(ComplianceReport).invoke(
        [SystemMessage(content=COMPLIANCE_OFFICER_PROMPT), HumanMessage(content=content)])
    print(f"   ✅ recommendation: {result.recommended_action} — {result.justification[:100]}")
    return {"report": result.report_markdown,
            "recommended_action": result.recommended_action,
            "revision_count": (state.get("revision_count") or 0) + (1 if feedback else 0),
            "human_decision": None,
            "messages": [AIMessage(content=f"[Compliance Officer] Recommended action: "
                                           f"{result.recommended_action}. {result.justification}")]}

# %% [markdown]
# ## 5. Human-in-the-loop node + terminal actions
# `interrupt()` pauses the graph **before the critical action**. The value passed to
# `interrupt()` is shown to the human; the value the human resumes with
# (`Command(resume=...)`) becomes its return value.

# %%
def human_review(state: FraudWorkflowState) -> dict:
    print("⏸️ human_review: pausing for human approval...")
    decision = interrupt({
        "question": "Review the compliance report. Reply with one of: "
                    "{'type':'approve'} | {'type':'feedback','feedback':'...'} | {'type':'reject'}",
        "report": state["report"],
        "recommended_action": state["recommended_action"],
        "risk_score": (state.get("risk_assessment") or {}).get("risk_score"),
    })
    print(f"▶️ human_review: resumed with decision={decision}")
    return {"human_decision": decision,
            "messages": [HumanMessage(content=f"[Human reviewer] {json.dumps(decision)}")]}


MAX_REVISIONS = 3


def route_after_review(state) -> str:
    d = (state.get("human_decision") or {}).get("type")
    if d == "reject":
        return "cancel_action"
    if d == "feedback" and (state.get("revision_count") or 0) < MAX_REVISIONS:
        return "compliance_officer"
    return "execute_action"    # approve, or feedback budget exhausted


def execute_action(state: FraudWorkflowState) -> dict:
    action = state["recommended_action"]
    cid = state.get("customer_id") or "N/A"
    effects = {"BLOCK": f"🚫 Card of {cid} BLOCKED and case escalated to the fraud team.",
               "MONITOR": f"👀 {cid} placed on enhanced monitoring for 30 days.",
               "CLEAR": f"✅ {cid} cleared — no action taken."}
    out = (f"ACTION EXECUTED: {effects.get(action, action)}\n\n--- FINAL COMPLIANCE REPORT ---\n"
           f"{state['report']}")
    print(f"🏁 execute_action: {action} for {cid}")
    return {"final_output": out, "messages": [AIMessage(content=out)]}


def cancel_action(state: FraudWorkflowState) -> dict:
    out = (f"ACTION CANCELLED by human reviewer. No changes applied to "
           f"{state.get('customer_id')}. The draft report was archived for audit.")
    print("🛑 cancel_action")
    return {"final_output": out, "messages": [AIMessage(content=out)]}


def followup_qa(state: FraudWorkflowState) -> dict:
    print("💬 followup_qa: answering from conversation memory...")
    answer = get_llm().invoke(
        [SystemMessage(content="Answer the user's follow-up question strictly from the "
                               "conversation history of this fraud-review thread.")]
        + state["messages"])
    return {"final_output": answer.content, "messages": [answer]}

# %% [markdown]
# ## 6. Assembling the LangGraph
# ```
# intake ──► fraud_analyst ──► compliance_officer ──► human_review ──► execute_action ► END
#    │                                  ▲                  │ feedback        │reject
#    └──► followup_qa ► END             └──────────────────┤                 ▼
#                                                          └──────────► cancel_action ► END
# ```

# %%
_builder = StateGraph(FraudWorkflowState)
for _name, _fn in [("intake", intake), ("fraud_analyst", fraud_analyst),
                   ("compliance_officer", compliance_officer), ("human_review", human_review),
                   ("execute_action", execute_action), ("cancel_action", cancel_action),
                   ("followup_qa", followup_qa)]:
    _builder.add_node(_name, _fn)
_builder.add_edge(START, "intake")
_builder.add_conditional_edges("intake", route_after_intake,
                               {"fraud_analyst": "fraud_analyst", "followup_qa": "followup_qa"})
_builder.add_edge("fraud_analyst", "compliance_officer")
_builder.add_edge("compliance_officer", "human_review")
_builder.add_conditional_edges("human_review", route_after_review,
                               {"execute_action": "execute_action",
                                "compliance_officer": "compliance_officer",
                                "cancel_action": "cancel_action"})
for _terminal in ("execute_action", "cancel_action", "followup_qa"):
    _builder.add_edge(_terminal, END)

checkpointer = MemorySaver()                       # conversational memory (requirement)
graph = _builder.compile(checkpointer=checkpointer)

# %%
if not SKIP_DEMOS:
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        print(graph.get_graph().draw_ascii())      # offline fallback

# %% [markdown]
# ## 7. Core function: `execute_workflow` (+ resume helper)
# Required by the assignment. `execute_workflow(user_request)` starts the graph and runs
# until the human-in-the-loop interrupt (or completion). `resume_workflow` feeds the
# human's decision back via `Command(resume=...)`. `run_scenario` scripts the human's
# decisions so the notebook is reproducible end-to-end in Colab.

# %%
import uuid


def _result_of(raw: dict, thread_id: str) -> dict:
    if raw.get("__interrupt__"):
        return {"status": "awaiting_human_review", "thread_id": thread_id,
                "interrupt_payload": raw["__interrupt__"][0].value}
    return {"status": "completed", "thread_id": thread_id,
            "final_output": raw.get("final_output")}


def execute_workflow(user_request: str, thread_id: Optional[str] = None) -> dict:
    """Start the fraud-review workflow for a natural-language request.
    Runs until the graph pauses for human review or finishes."""
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    raw = graph.invoke({"user_request": user_request}, config=config)
    return _result_of(raw, thread_id)


def resume_workflow(thread_id: str, decision: dict) -> dict:
    """Resume an interrupted workflow with the human decision:
    {'type':'approve'} | {'type':'feedback','feedback':'...'} | {'type':'reject'}"""
    config = {"configurable": {"thread_id": thread_id}}
    raw = graph.invoke(Command(resume=decision), config=config)
    return _result_of(raw, thread_id)


def run_scenario(title: str, user_request: str, decisions: list,
                 thread_id: Optional[str] = None) -> dict:
    """Demo driver: executes a workflow and feeds scripted human decisions in order,
    printing everything a reviewer needs to see the HITL process."""
    print("=" * 88); print(f"🧪 {title}"); print(f"USER REQUEST: {user_request}"); print("=" * 88)
    result = execute_workflow(user_request, thread_id=thread_id)
    step = 0
    while result["status"] == "awaiting_human_review":
        payload = result["interrupt_payload"]
        print("\n" + "-" * 88)
        print(f"⏸️ GRAPH INTERRUPTED — report for human review "
              f"(recommended: {payload['recommended_action']}, score: {payload['risk_score']}):")
        print(payload["report"])
        decision = decisions[step] if step < len(decisions) else {"type": "approve"}
        step += 1
        print(f"\n🧑 HUMAN DECISION: {decision}")
        print("-" * 88)
        result = resume_workflow(result["thread_id"], decision)
    print("\n🏁 FINAL OUTPUT:\n" + (result.get("final_output") or "(none)"))
    return result

# %% [markdown]
# ## 8. Test cases (≥ 5, incl. approve / feedback-revision / reject / error / memory)
# Each cell runs one scenario. `run_scenario` scripts the human decisions so the whole
# notebook executes reproducibly; in a live setting a human would type the decision.

# %% [markdown]
# ### Test 1 — Clean customer → CLEAR, human approves

# %%
if not SKIP_DEMOS:
    t1 = run_scenario("Test 1: clean customer (approve)",
                      "Please review customer CUST-1001 for suspicious activity.",
                      decisions=[{"type": "approve"}])

# %% [markdown]
# ### Test 2 — Card-testing fraud → BLOCK, human approves

# %%
if not SKIP_DEMOS:
    t2 = run_scenario("Test 2: card-testing pattern (approve block)",
                      "Investigate CUST-1042 — we received a fraud alert from the issuer.",
                      decisions=[{"type": "approve"}])

# %% [markdown]
# ### Test 3 — Velocity fraud → human sends FEEDBACK → report revised → approve

# %%
if not SKIP_DEMOS:
    t3 = run_scenario("Test 3: velocity pattern (feedback → revision → approve)",
                      "Check CUST-1337, the terminal reported many rapid payments.",
                      decisions=[{"type": "feedback",
                                  "feedback": "Blocking is too aggressive for a first incident. "
                                              "Recommend MONITOR and describe what would escalate "
                                              "it to BLOCK."},
                                 {"type": "approve"}])

# %% [markdown]
# ### Test 4 — Geo anomaly + risky MCCs → human REJECTS the action

# %%
if not SKIP_DEMOS:
    t4 = run_scenario("Test 4: geo anomaly (human rejects)",
                      "Run a fraud review for customer CUST-2077.",
                      decisions=[{"type": "reject"}])

# %% [markdown]
# ### Test 5 — Unknown customer → graceful error handling

# %%
if not SKIP_DEMOS:
    t5 = run_scenario("Test 5: unknown customer (error path)",
                      "Please check CUST-9999 for suspicious transactions.",
                      decisions=[{"type": "approve"}])

# %% [markdown]
# ### Test 6 — Conversational memory: follow-up question in the same thread
# We reuse Test 2's `thread_id`; `MemorySaver` restores the whole conversation, and the
# graph routes a request without a customer id to the `followup_qa` node.

# %%
if not SKIP_DEMOS:
    t6 = execute_workflow("What was the final risk score and which rules were triggered?",
                          thread_id=t2["thread_id"])
    print("💬 Memory-based answer:\n", t6["final_output"])

# %% [markdown]
# ## 9. Conclusion
# The system demonstrates: two role-specialized agents over a shared typed state, three
# custom tools actually invoked by the analyst, `MemorySaver` conversational memory,
# a real `interrupt()`-based human-in-the-loop gate before the critical action
# (approve / revise / reject all shown), the required `execute_workflow` entry point,
# and six reproducible test cases — runnable both in Google Colab and VS Code.
