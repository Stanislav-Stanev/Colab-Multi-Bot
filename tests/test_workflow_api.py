import pytest

import fraud_multi_agent as m


class _Interrupt:
    def __init__(self, value):
        self.value = value


def _payload(report="draft", action="BLOCK", score=80):
    return {"report": report, "recommended_action": action, "risk_score": score}


def test_start_workflow_detects_interrupt(monkeypatch):
    class FakeGraph:
        def invoke(self, inp, config=None):
            assert inp == {"user_request": "check CUST-1042"}
            assert config["configurable"]["thread_id"]
            return {"__interrupt__": [_Interrupt(_payload())]}

    monkeypatch.setattr(m, "graph", FakeGraph())
    out = m.start_workflow("check CUST-1042")
    assert out["status"] == "awaiting_human_review"
    assert out["interrupt_payload"]["report"] == "draft"
    assert out["thread_id"]


def test_start_workflow_honours_given_thread_id(monkeypatch):
    monkeypatch.setattr(m, "graph", type("G", (), {
        "invoke": lambda self, i, config=None: {"final_output": "ok"}})())
    assert m.start_workflow("hi", thread_id="t-42")["thread_id"] == "t-42"


def test_resume_workflow_completes(monkeypatch):
    class FakeGraph:
        def invoke(self, cmd, config=None):
            assert cmd.resume == {"type": "approve"}
            return {"final_output": "done"}

    monkeypatch.setattr(m, "graph", FakeGraph())
    out = m.resume_workflow("t-1", {"type": "approve"})
    assert out == {"status": "completed", "thread_id": "t-1", "final_output": "done"}


class ScriptedGraph:
    """Interrupts `pauses` times, then completes; records every resume value."""

    def __init__(self, pauses):
        self.pauses = pauses
        self.resumes = []

    def invoke(self, arg, config=None):
        if hasattr(arg, "resume"):
            self.resumes.append(arg.resume)
        if self.pauses > 0:
            self.pauses -= 1
            return {"__interrupt__": [_Interrupt(_payload(f"draft {self.pauses}"))]}
        return {"final_output": "final report"}


def test_execute_workflow_takes_one_string_and_returns_the_final_output(monkeypatch):
    """The assignment's core function: one request in, final output out, HITL handled."""
    fake = ScriptedGraph(pauses=1)
    monkeypatch.setattr(m, "graph", fake)
    monkeypatch.setattr("builtins.input", lambda _="": "approve")

    result = m.execute_workflow("Investigate CUST-1042.")

    assert result["status"] == "completed"
    assert result["final_output"] == "final report"
    assert fake.resumes == [{"type": "approve"}]


def test_execute_workflow_drives_every_interruption_to_completion(monkeypatch):
    fake = ScriptedGraph(pauses=3)
    monkeypatch.setattr(m, "graph", fake)
    decisions = [{"type": "feedback", "feedback": "softer"},
                 {"type": "feedback", "feedback": "again"},
                 {"type": "approve"}]

    result = m.execute_workflow("check CUST-1337", decisions=decisions)

    assert result["status"] == "completed"
    assert fake.resumes == decisions          # every scripted decision was delivered, in order


def test_execute_workflow_falls_back_to_asking_when_decisions_run_out(monkeypatch):
    fake = ScriptedGraph(pauses=2)
    monkeypatch.setattr(m, "graph", fake)
    monkeypatch.setattr("builtins.input", lambda _="": "reject")

    m.execute_workflow("check CUST-2077", decisions=[{"type": "approve"}])

    assert fake.resumes == [{"type": "approve"}, {"type": "reject"}]


def test_execute_workflow_never_prompts_when_fully_scripted(monkeypatch):
    monkeypatch.setattr(m, "graph", ScriptedGraph(pauses=1))

    def explode(_=""):
        raise AssertionError("input() must not be called when decisions are supplied")

    monkeypatch.setattr("builtins.input", explode)
    assert m.execute_workflow("x", decisions=[{"type": "approve"}])["status"] == "completed"


@pytest.mark.parametrize("typed,expected", [
    ("", {"type": "approve"}),
    ("approve", {"type": "approve"}),
    ("Y", {"type": "approve"}),
    ("reject", {"type": "reject"}),
    ("n", {"type": "reject"}),
    ("please soften the wording", {"type": "feedback", "feedback": "please soften the wording"}),
])
def test_ask_human_parses_operator_input(monkeypatch, typed, expected):
    monkeypatch.setattr("builtins.input", lambda _="": typed)
    assert m.ask_human(_payload()) == expected


# ---- approve-with-edits reaches the operator (review F11) -------------------------------


def test_ask_human_offers_an_edit_path(monkeypatch):
    """Typing `edit: ...` approves and replaces the text, in one step."""
    monkeypatch.setattr("builtins.input", lambda _="": "edit: Your card is paused.")
    assert m.ask_human(_payload()) == {"type": "approve",
                                       "edited_body": "Your card is paused."}


def test_edit_is_not_mistaken_for_feedback(monkeypatch):
    """Free text is still feedback; only the explicit prefix means "send exactly this"."""
    monkeypatch.setattr("builtins.input", lambda _="": "please edit the wording")
    assert m.ask_human(_payload())["type"] == "feedback"


def test_the_gate_tells_the_reviewer_the_edit_option_exists():
    prompt = m.HUMAN_REVIEW_QUESTION.lower()
    for option in ("approve", "feedback", "reject", "edit"):
        assert option in prompt
