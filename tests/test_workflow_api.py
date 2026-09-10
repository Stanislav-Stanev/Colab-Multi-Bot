import fraud_multi_agent as m


class _Interrupt:
    def __init__(self, value):
        self.value = value


def test_execute_workflow_detects_interrupt(monkeypatch):
    class FakeGraph:
        def invoke(self, inp, config=None):
            assert inp == {"user_request": "check CUST-1042"}
            assert config["configurable"]["thread_id"]
            return {"__interrupt__": [_Interrupt({"report": "draft"})]}

    monkeypatch.setattr(m, "graph", FakeGraph())
    out = m.execute_workflow("check CUST-1042")
    assert out["status"] == "awaiting_human_review"
    assert out["interrupt_payload"] == {"report": "draft"}
    assert out["thread_id"]


def test_execute_workflow_honours_given_thread_id(monkeypatch):
    monkeypatch.setattr(m, "graph", type("G", (), {
        "invoke": lambda self, i, config=None: {"final_output": "ok"}})())
    assert m.execute_workflow("hi", thread_id="t-42")["thread_id"] == "t-42"


def test_resume_workflow_completes(monkeypatch):
    class FakeGraph:
        def invoke(self, cmd, config=None):
            assert cmd.resume == {"type": "approve"}
            return {"final_output": "done"}

    monkeypatch.setattr(m, "graph", FakeGraph())
    out = m.resume_workflow("t-1", {"type": "approve"})
    assert out == {"status": "completed", "thread_id": "t-1", "final_output": "done"}


def test_run_scenario_feeds_decisions_in_order(monkeypatch, capsys):
    calls = []

    class FakeGraph:
        def __init__(self):
            self.step = 0

        def invoke(self, arg, config=None):
            self.step += 1
            if self.step <= 2:  # interrupt twice, then finish
                return {"__interrupt__": [_Interrupt(
                    {"report": f"draft {self.step}", "recommended_action": "BLOCK",
                     "risk_score": 80})]}
            calls.append(getattr(arg, "resume", None))
            return {"final_output": "final report"}

    fake = FakeGraph()
    monkeypatch.setattr(m, "graph", fake)
    decisions = [{"type": "feedback", "feedback": "softer"}, {"type": "approve"}]
    result = m.run_scenario("t", "check CUST-1337", decisions)

    assert result["status"] == "completed"
    assert calls == [{"type": "approve"}]  # last resume that completed the run
    printed = capsys.readouterr().out
    assert "GRAPH INTERRUPTED" in printed and "softer" in printed


def test_run_scenario_defaults_to_approve_when_decisions_exhausted(monkeypatch):
    seen = []

    class FakeGraph:
        def __init__(self):
            self.first = True

        def invoke(self, arg, config=None):
            if self.first:
                self.first = False
                return {"__interrupt__": [_Interrupt(
                    {"report": "d", "recommended_action": "CLEAR", "risk_score": 5})]}
            seen.append(arg.resume)
            return {"final_output": "done"}

    monkeypatch.setattr(m, "graph", FakeGraph())
    m.run_scenario("t", "check CUST-1001", decisions=[])
    assert seen == [{"type": "approve"}]
