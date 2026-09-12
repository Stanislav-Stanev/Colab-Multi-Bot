"""Sanctions screening is executed, not recalled — and a hit is a hard stop.

Observed in a live run: the analyst called `check_sanctions_list(customer_name="CUST-4444")`
before it had fetched the profile, so RevolutBank screened a customer id instead of
"Viktor Baranov", found nothing, and recommended CLEAR on a watch-listed customer. A
regulatory control must not depend on the order in which a model happens to call its tools.
"""
import json

import pytest
from langchain_core.messages import AIMessage

import fraud_multi_agent as m


class _StructuredFake:
    def __init__(self, schema, holder):
        self.schema = schema
        self.holder = holder

    def invoke(self, messages):
        return self.holder.reply_for(self.schema, messages)


class AnalystFakeLLM:
    """An analyst that screens the *wrong* name, exactly as the live model did."""

    def __init__(self, screened_name, action="CLEAR"):
        self.screened_name = screened_name
        self.action = action
        self.narrative_prompts = []
        self.compliance_prompts = []

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, method=None):
        return _StructuredFake(schema, self)

    def invoke(self, messages):
        if any(getattr(msg, "type", "") == "tool" for msg in messages):
            return AIMessage(content="Investigation complete.")
        return AIMessage(content="", tool_calls=[
            {"name": "check_sanctions_list",
             "args": {"customer_name": self.screened_name}, "id": "s1", "type": "tool_call"},
            {"name": "fetch_customer_transactions",
             "args": {"customer_id": "CUST-4444"}, "id": "f1", "type": "tool_call"}])

    def reply_for(self, schema, messages):
        if schema is m.AnalystNarrative:
            self.narrative_prompts.append(messages)
            return m.AnalystNarrative(analyst_notes="Two quiet remittances.")
        if schema is m.ComplianceReport:
            self.compliance_prompts.append(messages)
            return m.ComplianceReport(report_markdown="# Report", recommended_action=self.action,
                                      justification="Score is low.")
        raise AssertionError(f"unexpected schema {schema}")


@pytest.fixture
def analyst_llm(monkeypatch):
    def install(screened_name, action="CLEAR"):
        fake = AnalystFakeLLM(screened_name, action)
        monkeypatch.setattr(m, "get_llm", lambda: fake)
        return fake
    return install


def test_screening_uses_the_name_the_core_banking_system_returned(analyst_llm):
    """Even when the model screened the customer id, the hit is still found."""
    analyst_llm(screened_name="CUST-4444")
    out = m.fraud_analyst({"user_request": "Review CUST-4444.", "customer_id": "CUST-4444"})
    assert out["risk_assessment"]["sanctions_match"] is True
    assert out["risk_assessment"]["screened_name"] == "Viktor Baranov"


def test_screening_is_clean_for_a_customer_not_on_the_list(analyst_llm):
    analyst_llm(screened_name="anything")
    out = m.fraud_analyst({"user_request": "Review CUST-1001.", "customer_id": "CUST-1001"})
    assert out["risk_assessment"]["sanctions_match"] is False


def test_the_analyst_narrative_no_longer_carries_the_screening_result():
    """The model contributes prose; facts come from the tools that produced them."""
    assert "sanctions_match" not in m.AnalystNarrative.model_fields


def test_an_unknown_customer_is_not_screened(analyst_llm):
    analyst_llm(screened_name="CUST-9999")
    out = m.fraud_analyst({"user_request": "Review CUST-9999.", "customer_id": "CUST-9999"})
    assert out["customer_found"] is False
    assert out["risk_assessment"]["sanctions_match"] is False
    assert out["risk_assessment"]["screened_name"] is None


def test_a_sanctions_hit_forces_block_over_a_lenient_recommendation(analyst_llm, tmp_path,
                                                                    monkeypatch):
    """A watch-list match is a legal hard stop, not a judgement call."""
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    analyst_llm(screened_name="CUST-4444", action="CLEAR")
    out = m.compliance_officer({
        "customer_id": "CUST-4444", "case_id": "CASE-X",
        "risk_assessment": {"risk_score": 15, "triggered_rules": [], "sanctions_match": True,
                            "screened_name": "Viktor Baranov", "analyst_notes": "quiet"},
        "revision_count": 0})
    assert out["recommended_action"] == "BLOCK"
    assert "sanctions" in out["report"].lower()

    entries = [json.loads(line) for line
               in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    overrides = [e for e in entries if e["event"] == "recommendation_overridden"]
    assert len(overrides) == 1
    assert overrides[0]["from"] == "CLEAR" and overrides[0]["to"] == "BLOCK"


def test_a_clean_screening_leaves_the_recommendation_alone(analyst_llm):
    analyst_llm(screened_name="x", action="MONITOR")
    out = m.compliance_officer({
        "customer_id": "CUST-1337", "case_id": "CASE-Y",
        "risk_assessment": {"risk_score": 40, "triggered_rules": ["velocity"],
                            "sanctions_match": False, "screened_name": "Ivan Dimitrov",
                            "analyst_notes": "velocity"},
        "revision_count": 0})
    assert out["recommended_action"] == "MONITOR"
