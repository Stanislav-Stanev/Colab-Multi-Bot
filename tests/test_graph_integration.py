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

    def with_structured_output(self, schema, method=None):
        assert method == "json_schema", "must use native structured outputs, not forced tools"
        return _StructuredFake(schema, self.script)

    def invoke(self, messages):
        return self.script.next_chat(messages)


class Script:
    """Drives FakeLLM: one tool-calling round, then structured outputs."""

    def __init__(self, customer_id, recommended_action="BLOCK"):
        self.customer_id = customer_id
        self.recommended_action = recommended_action
        self.tool_round_done = False
        self.triage_calls = []
        self.assessment_calls = []
        self.compliance_calls = []
        self.comms_calls = []
        self.followup_calls = []

    @staticmethod
    def _system(messages) -> str:
        return "".join(str(msg.content) for msg in messages
                       if getattr(msg, "type", "") == "system")

    def next_chat(self, messages):
        # followup_qa is the only caller whose prompt mentions a follow-up question.
        if any("follow-up" in getattr(msg, "content", "") for msg in messages
               if isinstance(getattr(msg, "content", ""), str)):
            self.followup_calls.append(messages)
            return AIMessage(content="Recalled from memory: score 85, rules card_testing.")

        tools_already_ran = any(getattr(msg, "type", "") == "tool" for msg in messages)
        system = self._system(messages)

        if "triage officer" in system:
            if tools_already_ran:
                return AIMessage(content="Triage complete.")
            return AIMessage(content="", tool_calls=[
                {"name": "lookup_case_history", "args": {"customer_id": self.customer_id},
                 "id": "tr1", "type": "tool_call"}])

        if "customer communications" in system:
            if tools_already_ran:
                return AIMessage(content="Notification delivered.")
            return AIMessage(content="", tool_calls=[
                {"name": "send_customer_notification",
                 "args": {"customer_id": self.customer_id, "channel": "push",
                          "subject": "Scripted subject", "body": "Scripted body"},
                 "id": "cm1", "type": "tool_call"}])

        # The analyst asks for the tools until their results are in, then wraps up. Keyed on
        # the messages rather than a flag, so several runs can share one script.
        if not tools_already_ran:
            self.tool_round_done = True
            return AIMessage(content="", tool_calls=[
                {"name": "fetch_customer_transactions",
                 "args": {"customer_id": self.customer_id}, "id": "tc1", "type": "tool_call"},
                {"name": "calculate_risk_score",
                 "args": {"customer_id": self.customer_id}, "id": "tc2", "type": "tool_call"},
            ])
        return AIMessage(content="Investigation complete.")

    def next_structured(self, schema, messages):
        if schema is m.TriageDecision:
            self.triage_calls.append(messages)
            return m.TriageDecision(signal_type="FRAUD_ALERT", priority="P2",
                                    rationale="Scripted triage rationale.")
        if schema is m.CustomerMessage:
            self.comms_calls.append(messages)
            return m.CustomerMessage(channel="push", language="en",
                                     subject="Scripted subject", body="Scripted body")
        if schema is m.AnalystNarrative:
            self.assessment_calls.append(messages)
            return m.AnalystNarrative(
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
        monkeypatch.setattr(m, "get_llm", lambda: FakeLLM(script))
        return script
    return install


def _thread():
    return f"test-{uuid.uuid4().hex[:8]}"


def test_approve_path_executes_block(scripted):
    scripted("CUST-1042", "BLOCK")
    started = m.start_workflow("Investigate CUST-1042 for fraud.", thread_id=_thread())

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
    started = m.start_workflow("Fraud review for CUST-2077.", thread_id=_thread())
    done = m.resume_workflow(started["thread_id"], {"type": "reject"})

    assert "CANCELLED" in done["final_output"]
    assert "BLOCKED" not in done["final_output"]


def test_feedback_path_revises_report_then_approves(scripted):
    script = scripted("CUST-1337", "BLOCK")
    started = m.start_workflow("Check CUST-1337 rapid payments.", thread_id=_thread())
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


def test_unknown_customer_stops_before_any_action(scripted):
    """No account means no report and no human gate — there is nothing to approve."""
    scripted("CUST-9999", "CLEAR")
    result = m.start_workflow("Please check CUST-9999.", thread_id=_thread())

    assert result["status"] == "completed"          # never pauses for a human
    assert "REVIEW NOT POSSIBLE" in result["final_output"]
    assert "CUST-9999" in result["final_output"]
    assert "BLOCKED" not in result["final_output"]


def test_known_customer_is_marked_found(scripted):
    scripted("CUST-1001", "CLEAR")
    thread = _thread()
    m.start_workflow("Review CUST-1001.", thread_id=thread)
    state = m.graph.get_state({"configurable": {"thread_id": thread}})
    assert state.values["customer_found"] is True


def test_memory_followup_uses_same_thread(scripted):
    script = scripted("CUST-1042", "BLOCK")
    thread = _thread()
    started = m.start_workflow("Investigate CUST-1042.", thread_id=thread)
    m.resume_workflow(thread, {"type": "approve"})

    followup = m.start_workflow("What was the final risk score again?", thread_id=thread)
    assert followup["status"] == "completed"
    assert "Recalled from memory" in followup["final_output"]

    # the follow-up node saw the earlier turns restored from the checkpointer
    history = "".join(str(msg.content) for msg in script.followup_calls[0])
    assert "CUST-1042" in history and "Compliance report v1" in history
    assert started["thread_id"] == thread


def test_threads_are_isolated(scripted):
    scripted("CUST-1001", "CLEAR")
    a = m.start_workflow("Review CUST-1001.", thread_id=_thread())
    b = m.start_workflow("Review CUST-1001.", thread_id=_thread())
    assert a["thread_id"] != b["thread_id"]
    m.resume_workflow(a["thread_id"], {"type": "approve"})
    # thread b is still paused and resumable independently
    done_b = m.resume_workflow(b["thread_id"], {"type": "reject"})
    assert "CANCELLED" in done_b["final_output"]


def test_analyst_really_invokes_tools(scripted):
    script = scripted("CUST-1042", "BLOCK")
    thread = _thread()
    m.start_workflow("Investigate CUST-1042.", thread_id=thread)

    # transactions came back through the tool and were stored in the shared state
    state = m.graph.get_state({"configurable": {"thread_id": thread}})
    assert len(state.values["transactions"]) == 6
    assert state.values["risk_assessment"]["risk_score"] >= 70

    # the structured-output call receives plain-text tool evidence, never tool_use blocks
    prompt = "".join(str(msg.content) for msg in script.assessment_calls[0])
    assert "fetch_customer_transactions" in prompt and "calculate_risk_score" in prompt
    assert all(not getattr(msg, "tool_calls", None) for msg in script.assessment_calls[0])


def test_no_notification_leaves_the_bank_before_approval(scripted, tmp_path, monkeypatch):
    """The invariant the human gate exists for, proven through the real graph."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    scripted("CUST-1042", "BLOCK")
    started = m.start_workflow("Investigate CUST-1042.", thread_id=_thread())

    # paused, with the drafted message in front of the human
    assert started["status"] == "awaiting_human_review"
    draft = started["interrupt_payload"]["customer_notification"]
    assert draft["subject"] == "Scripted subject"
    assert "notification_sent" not in path.read_text(encoding="utf-8")

    done = m.resume_workflow(started["thread_id"], {"type": "approve"})
    assert done["status"] == "completed"
    sent = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event"] == "notification_sent"]
    assert len(sent) == 1 and sent[0]["subject"] == "Scripted subject"


def test_rejecting_sends_no_customer_message(scripted, tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    scripted("CUST-2077", "BLOCK")
    started = m.start_workflow("Fraud review for CUST-2077.", thread_id=_thread())
    done = m.resume_workflow(started["thread_id"], {"type": "reject"})

    assert "CANCELLED" in done["final_output"]
    assert "notification_sent" not in path.read_text(encoding="utf-8")


def test_every_audit_entry_of_a_case_carries_its_case_id(scripted, tmp_path, monkeypatch):
    """The audit trail is only useful if one case's entries can be joined together."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    scripted("CUST-1042", "BLOCK")
    started = m.start_workflow("Investigate CUST-1042.", thread_id=_thread())
    m.resume_workflow(started["thread_id"], {"type": "approve"})

    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {e["event"] for e in entries} >= {"case_opened", "human_decision",
                                             "action_executed", "notification_sent"}
    case_ids = {e.get("case_id") for e in entries}
    assert len(case_ids) == 1 and None not in case_ids, \
        f"audit entries are not all joined to one case: {case_ids}"


def test_triage_classifies_and_prioritises_through_the_graph(scripted):
    scripted("CUST-1042", "BLOCK")
    thread = _thread()
    m.start_workflow("Investigate CUST-1042 — issuer fraud alert.", thread_id=thread)
    state = m.graph.get_state({"configurable": {"thread_id": thread}})

    assert state.values["signal_type"] == "FRAUD_ALERT"
    assert state.values["priority"] == "P1"        # escalated: CUST-1042 has a prior BLOCK
    assert state.values["case_id"].startswith("CASE-")
    assert state.values["case_history"][0]["outcome"] == "BLOCK"


def test_case_history_priority_is_not_escalated_for_a_clean_customer(scripted):
    scripted("CUST-1001", "CLEAR")
    thread = _thread()
    m.start_workflow("Please review CUST-1001.", thread_id=thread)
    state = m.graph.get_state({"configurable": {"thread_id": thread}})
    assert state.values["priority"] == "P2"        # what the model scripted, no escalation
    assert state.values["case_history"] == []


def test_revision_budget_stops_the_loop_without_executing_the_action(scripted, tmp_path,
                                                                     monkeypatch):
    """The loop breaker fails closed (review F1), through the real graph.

    Exhausting the budget must not finalise the recommendation the reviewer kept
    objecting to: nothing is executed, nothing is sent, the case is parked.
    """
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    scripted("CUST-1337", "BLOCK")
    thread = _thread()
    result = m.start_workflow("Check CUST-1337.", thread_id=thread)
    for _ in range(m.MAX_REVISIONS):
        assert result["status"] == "awaiting_human_review"
        result = m.resume_workflow(thread, {"type": "feedback", "feedback": "again please"})
    # budget exhausted: the next feedback stops the loop, but stops it closed
    assert result["status"] == "awaiting_human_review"
    final = m.resume_workflow(thread, {"type": "feedback", "feedback": "one more"})

    assert final["status"] == "completed"
    assert "ACTION NOT EXECUTED" in final["final_output"]
    assert "ACTION EXECUTED" not in final["final_output"]

    trail = path.read_text(encoding="utf-8")
    assert "revision_budget_exhausted" in trail
    assert "action_executed" not in trail, "a contested action was carried out anyway"
    assert "notification_sent" not in trail, "a message went out on an unapproved action"
