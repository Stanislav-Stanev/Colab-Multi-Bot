"""Customer Comms Agent: drafting before the gate, delivery only after it (plan §3.2).

The agent drafts the customer-facing message with the LLM, and that draft is part of what
the human reviews. Delivery replays the approved text verbatim — a model must not be able
to rewrite a message a human already signed off.
"""
import json

import pytest

import fraud_multi_agent as m


class _StructuredFake:
    def __init__(self, schema, holder):
        self.schema = schema
        self.holder = holder

    def invoke(self, messages):
        self.holder.calls.append((self.schema, messages))
        return self.holder.message

class CommsFakeLLM:
    def __init__(self, message):
        self.message = message
        self.calls = []

    def with_structured_output(self, schema, method=None):
        assert method == "json_schema"
        assert schema is m.CustomerMessage
        return _StructuredFake(schema, self)


@pytest.fixture
def comms_llm(monkeypatch):
    def install(**overrides):
        fields = {"channel": "push", "language": "en",
                  "subject": "Your card has been temporarily blocked",
                  "body": "We noticed unusual activity and paused your card."}
        fields.update(overrides)
        fake = CommsFakeLLM(m.CustomerMessage(**fields))
        monkeypatch.setattr(m, "get_llm", lambda: fake)
        return fake
    return install


def _case(**overrides):
    state = {"customer_id": "CUST-1042", "case_id": "CASE-TEST01",
             "recommended_action": "BLOCK", "report": "# Report",
             "risk_assessment": {"risk_score": 85, "triggered_rules": ["card_testing"]},
             "signal_type": "FRAUD_ALERT", "priority": "P1"}
    state.update(overrides)
    return state


# ---- drafting (before the human gate) -------------------------------------------------


def test_draft_puts_the_message_in_state_without_sending_it(comms_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    comms_llm()
    out = m.comms_draft(_case())
    draft = out["notification_draft"]
    assert draft["channel"] == "push"
    assert "card" in draft["subject"].lower()
    assert out.get("notification_result") is None
    # nothing left the bank: no delivery was recorded
    assert not (tmp_path / "audit.jsonl").exists() or "notification_sent" not in \
        (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_draft_is_told_the_action_and_never_the_raw_score(comms_llm):
    fake = comms_llm()
    m.comms_draft(_case())
    prompt = "".join(str(msg.content) for _, msgs in fake.calls for msg in msgs)
    assert "BLOCK" in prompt
    assert "85" not in prompt, "the customer-facing agent must not see the internal score"
    assert "card_testing" not in prompt, "internal rule names must not reach the drafter"


def test_comms_prompt_forbids_internal_detail_and_accusation():
    for rule in ("never", "score", "internal"):
        assert rule in m.CUSTOMER_COMMS_PROMPT.lower()
    assert "RevolutBank" in m.CUSTOMER_COMMS_PROMPT


def test_customer_message_rejects_an_unsupported_channel():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        m.CustomerMessage(channel="sms", language="en", subject="s", body="b")


def test_no_notification_is_drafted_for_a_cleared_customer(comms_llm):
    """A clean review is not news the customer needs — the agent may choose to stay silent."""
    comms_llm(channel="none", subject="", body="")
    out = m.comms_draft(_case(recommended_action="CLEAR"))
    assert out["notification_draft"]["channel"] == "none"


# ---- delivery (after the human gate) --------------------------------------------------


def test_delivery_sends_the_approved_text_verbatim(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    state = _case(human_decision={"type": "approve"},
                  notification_draft={"channel": "email", "language": "en",
                                      "subject": "Approved subject", "body": "Approved body"},
                  final_output="ACTION EXECUTED: blocked")
    out = m.comms_deliver(state)

    assert out["notification_result"]["status"] == "queued"
    sent = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event"] == "notification_sent"]
    assert len(sent) == 1
    assert sent[0]["subject"] == "Approved subject"
    assert sent[0]["body"] == "Approved body"
    assert sent[0]["channel"] == "email"


def test_delivery_is_skipped_when_the_draft_says_no_channel(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    out = m.comms_deliver(_case(human_decision={"type": "approve"},
                                notification_draft={"channel": "none", "language": "en",
                                                    "subject": "", "body": ""},
                                final_output="ACTION EXECUTED: cleared"))
    assert out["notification_result"]["status"] == "skipped"
    assert "notification_sent" not in path.read_text(encoding="utf-8")


def test_delivery_refuses_without_an_approval(tmp_path, monkeypatch):
    """The invariant the whole HITL gate exists for: no approval, no outbound message."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    out = m.comms_deliver(_case(human_decision={"type": "reject"},
                                notification_draft={"channel": "push", "language": "en",
                                                    "subject": "s", "body": "b"}))
    assert out["notification_result"]["status"] == "suppressed"
    assert "notification_sent" not in path.read_text(encoding="utf-8")


def test_delivery_appends_the_outcome_to_the_final_output(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    out = m.comms_deliver(_case(human_decision={"type": "approve"},
                                notification_draft={"channel": "push", "language": "en",
                                                    "subject": "Card blocked", "body": "b"},
                                final_output="ACTION EXECUTED: blocked"))
    assert "ACTION EXECUTED" in out["final_output"]
    assert "Card blocked" in out["final_output"]


def test_delivery_reports_a_rejected_channel_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    out = m.comms_deliver(_case(human_decision={"type": "approve"},
                                notification_draft={"channel": "fax", "language": "en",
                                                    "subject": "s", "body": "b"},
                                final_output="ACTION EXECUTED"))
    assert out["notification_result"]["status"] == "failed"
    assert "error" in out["notification_result"]


# ---- approve-with-edits (review F11) ----------------------------------------------------


def test_the_reviewer_can_edit_the_message_and_the_edit_is_what_ships(tmp_path, monkeypatch):
    """The strongest form of "a human signs the exact words": the words are the human's.

    Feedback costs another officer round and another redraft; a reviewer who only wants
    two words changed can change them at the gate, and the harness delivers that text
    verbatim under the same approval.
    """
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    m._DELIVERED.clear()
    out = m.comms_deliver(_case(
        human_decision={"type": "approve",
                        "edited_body": "Your card is paused. Call us on 0700 11 222."},
        notification_draft={"channel": "push", "language": "en",
                            "subject": "Card paused", "body": "Original model draft."},
        final_output="ACTION EXECUTED: blocked"))

    assert out["notification_result"]["status"] == "queued"
    sent = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event"] == "notification_sent"]
    assert len(sent) == 1
    assert sent[0]["body"] == "Your card is paused. Call us on 0700 11 222."
    assert sent[0]["body"] != "Original model draft."
    assert out["final_output"].count("Original model draft.") == 0

    # the trail keeps both sides of the edit: an auditor must be able to see what the
    # model proposed and what the reviewer decided to send instead
    edits = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if json.loads(line)["event"] == "message_edited_by_reviewer"]
    assert len(edits) == 1 and edits[0]["case_id"] == "CASE-TEST01"
    assert edits[0]["original_body"] == "Original model draft."
    assert edits[0]["edited_body"] == "Your card is paused. Call us on 0700 11 222."


def test_an_edit_without_an_approval_ships_nothing(tmp_path, monkeypatch):
    """`edited_body` rides on an approval; on any other decision it changes nothing."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    out = m.comms_deliver(_case(
        human_decision={"type": "reject", "edited_body": "should never leave"},
        notification_draft={"channel": "push", "language": "en",
                            "subject": "s", "body": "b"}))
    assert out["notification_result"]["status"] == "suppressed"
    assert "notification_sent" not in path.read_text(encoding="utf-8")


def test_an_empty_edit_falls_back_to_the_approved_draft(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(m, "AUDIT_LOG_PATH", str(path))
    m._DELIVERED.clear()
    out = m.comms_deliver(_case(
        human_decision={"type": "approve", "edited_body": "   "},
        notification_draft={"channel": "email", "language": "en",
                            "subject": "s", "body": "The approved draft."},
        final_output="ACTION EXECUTED"))
    sent = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event"] == "notification_sent"]
    assert out["notification_result"]["status"] == "queued"
    assert sent[0]["body"] == "The approved draft."
