"""End-to-end runs of the real LangGraph with a scripted fake LLM.

Proves the graph mechanics (tool loop, interrupt/resume, revision loop, MemorySaver
memory, routing) without spending Anthropic tokens.
"""
import json
import uuid

import pytest
from langchain_core.messages import AIMessage

import fraud_multi_agent as m


class _StructuredFake:
    """Stands in for llm.with_structured_output(Schema)."""

    def __init__(self, schema, script):
        self.schema = schema
        self.script = script

    def invoke(self, messages):
        return self.script.next_structured(self.schema, messages)


class FakeLLM:
    """Scripted LLM: the analyst makes real tool calls, both agents emit structured output."""

    def __init__(self, script):
        self.script = script
        self._tools = None

    def bind_tools(self, tools):
        clone = FakeLLM(self.script)
        clone._tools = tools
        return clone

    def with_structured_output(self, schema):
        return _StructuredFake(schema, self.script)

    def invoke(self, messages):
        return self.script.next_chat(messages)


class Script:
    """Drives FakeLLM: one tool-calling round, then structured outputs."""

    def __init__(self, customer_id, recommended_action="BLOCK"):
        self.customer_id = customer_id
        self.recommended_action = recommended_action
        self.tool_round_done = False
        self.assessment_calls = []
        self.compliance_calls = []
        self.followup_calls = []

    def next_chat(self, messages):
        # First chat call = analyst asking for tools; second = analyst wrap-up.
        if not self.tool_round_done:
            self.tool_round_done = True
            return AIMessage(content="", tool_calls=[
                {"name": "fetch_customer_transactions",
                 "args": {"customer_id": self.customer_id}, "id": "tc1", "type": "tool_call"},
                {"name": "check_sanctions_list",
                 "args": {"customer_name": "Someone"}, "id": "tc2", "type": "tool_call"},
            ])
        # A chat call with no pending tool work is either the analyst's summary or followup_qa.
        if any("follow-up" in getattr(msg, "content", "") for msg in messages
               if isinstance(getattr(msg, "content", ""), str)):
            self.followup_calls.append(messages)
            return AIMessage(content="Recalled from memory: score 85, rules card_testing.")
        return AIMessage(content="Investigation complete.")

    def next_structured(self, schema, messages):
        if schema is m.RiskAssessment:
            self.assessment_calls.append(messages)
            scored = m._score(m._fetch(self.customer_id).get("transactions", []))
            return m.RiskAssessment(
                risk_score=scored["risk_score"],
                triggered_rules=scored["triggered_rules"],
                sanctions_match=False,
                analyst_notes="Scripted analyst notes.")
        if schema is m.ComplianceReport:
            self.compliance_calls.append(messages)
            revision_no = len(self.compliance_calls)
            action = self.recommended_action if revision_no == 1 else "MONITOR"
            return m.ComplianceReport(
                report_markdown=f"# Compliance report v{revision_no}\nAction: {action}",
                recommended_action=action,
                justification=f"Scripted justification v{revision_no}")
        raise AssertionError(f"unexpected schema {schema}")


@pytest.fixture
def scripted(monkeypatch):
    def install(customer_id, recommended_action="BLOCK"):
        script = Script(customer_id, recommended_action)
        monkeypatch.setattr(m, "get_llm", lambda temperature=0.0: FakeLLM(script))
        return script
    return install


def _thread():
    return f"test-{uuid.uuid4().hex[:8]}"


def test_approve_path_executes_block(scripted):
    scripted("CUST-1042", "BLOCK")
    started = m.execute_workflow("Investigate CUST-1042 for fraud.", thread_id=_thread())

    assert started["status"] == "awaiting_human_review"
    payload = started["interrupt_payload"]
    assert payload["recommended_action"] == "BLOCK"
    assert payload["risk_score"] >= 70
    assert "Compliance report v1" in payload["report"]

    done = m.resume_workflow(started["thread_id"], {"type": "approve"})
    assert done["status"] == "completed"
    assert "BLOCKED" in done["final_output"]
    assert "CUST-1042" in done["final_output"]


def test_reject_path_cancels_action(scripted):
    scripted("CUST-2077", "BLOCK")
    started = m.execute_workflow("Fraud review for CUST-2077.", thread_id=_thread())
    done = m.resume_workflow(started["thread_id"], {"type": "reject"})

    assert "CANCELLED" in done["final_output"]
    assert "BLOCKED" not in done["final_output"]


def test_feedback_path_revises_report_then_approves(scripted):
    script = scripted("CUST-1337", "BLOCK")
    started = m.execute_workflow("Check CUST-1337 rapid payments.", thread_id=_thread())
    assert "Compliance report v1" in started["interrupt_payload"]["report"]

    revised = m.resume_workflow(started["thread_id"],
                                {"type": "feedback", "feedback": "Too aggressive, prefer MONITOR."})
    assert revised["status"] == "awaiting_human_review"          # interrupts again after revision
    assert "Compliance report v2" in revised["interrupt_payload"]["report"]
    assert revised["interrupt_payload"]["recommended_action"] == "MONITOR"

    # the officer actually received the human's feedback text
    second_call = "".join(str(msg.content) for msg in script.compliance_calls[1])
    assert "Too aggressive" in second_call and "HUMAN REVIEWER FEEDBACK" in second_call

    done = m.resume_workflow(revised["thread_id"], {"type": "approve"})
    assert "monitoring" in done["final_output"].lower()


def test_unknown_customer_still_completes(scripted):
    scripted("CUST-9999", "CLEAR")
    started = m.execute_workflow("Please check CUST-9999.", thread_id=_thread())
    done = m.resume_workflow(started["thread_id"], {"type": "approve"})
    assert done["status"] == "completed"
    assert started["interrupt_payload"]["risk_score"] == 0


def test_memory_followup_uses_same_thread(scripted):
    script = scripted("CUST-1042", "BLOCK")
    thread = _thread()
    started = m.execute_workflow("Investigate CUST-1042.", thread_id=thread)
    m.resume_workflow(thread, {"type": "approve"})

    followup = m.execute_workflow("What was the final risk score again?", thread_id=thread)
    assert followup["status"] == "completed"
    assert "Recalled from memory" in followup["final_output"]

    # the follow-up node saw the earlier turns restored from the checkpointer
    history = "".join(str(msg.content) for msg in script.followup_calls[0])
    assert "CUST-1042" in history and "Compliance report v1" in history
    assert started["thread_id"] == thread


def test_threads_are_isolated(scripted):
    scripted("CUST-1001", "CLEAR")
    a = m.execute_workflow("Review CUST-1001.", thread_id=_thread())
    b = m.execute_workflow("Review CUST-1001.", thread_id=_thread())
    assert a["thread_id"] != b["thread_id"]
    m.resume_workflow(a["thread_id"], {"type": "approve"})
    # thread b is still paused and resumable independently
    done_b = m.resume_workflow(b["thread_id"], {"type": "reject"})
    assert "CANCELLED" in done_b["final_output"]


def test_analyst_really_invokes_tools(scripted):
    script = scripted("CUST-1042", "BLOCK")
    thread = _thread()
    m.execute_workflow("Investigate CUST-1042.", thread_id=thread)

    # transactions came back through the tool and were stored in the shared state
    state = m.graph.get_state({"configurable": {"thread_id": thread}})
    assert len(state.values["transactions"]) == 6
    assert state.values["risk_assessment"]["risk_score"] >= 70

    # the structured-output call receives plain-text tool evidence, never tool_use blocks
    prompt = "".join(str(msg.content) for msg in script.assessment_calls[0])
    assert "fetch_customer_transactions" in prompt and "check_sanctions_list" in prompt
    assert all(not getattr(msg, "tool_calls", None) for msg in script.assessment_calls[0])


def test_revision_budget_forces_finalisation(scripted):
    scripted("CUST-1337", "BLOCK")
    thread = _thread()
    result = m.execute_workflow("Check CUST-1337.", thread_id=thread)
    for _ in range(m.MAX_REVISIONS):
        assert result["status"] == "awaiting_human_review"
        result = m.resume_workflow(thread, {"type": "feedback", "feedback": "again please"})
    # budget exhausted: the next feedback no longer loops, the action is executed
    assert result["status"] == "awaiting_human_review"
    final = m.resume_workflow(thread, {"type": "feedback", "feedback": "one more"})
    assert final["status"] == "completed"
    assert "ACTION EXECUTED" in final["final_output"]
