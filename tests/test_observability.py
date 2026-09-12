"""Observability: logging setup, the event recorder, and per-node attribution.

Runs without an API key.
"""
import json
import logging

import pytest

import fraud_multi_agent as m


@pytest.fixture(autouse=True)
def _isolate_observer(fresh_observer):
    """Every test in this module gets the isolated observer from conftest."""
    return fresh_observer


# --------------------------------------------------------------------- logging
def test_setup_logging_is_idempotent():
    """Re-running the config cell in Colab must not print every later line twice."""
    for _ in range(3):
        logger = m.setup_logging("INFO")
    assert len(logger.handlers) == 1
    assert logger.propagate is False, "the root logger would print the message again"


def test_setup_logging_honours_the_level():
    assert m.setup_logging("WARNING").level == logging.WARNING
    assert m.setup_logging("INFO").level == logging.INFO


def test_info_lines_carry_no_formatter_noise(capsys):
    m.setup_logging("INFO")
    m.log.info("📥 intake: plain")
    assert capsys.readouterr().out.strip() == "📥 intake: plain"


def test_operational_levels_are_labelled(capsys):
    m.setup_logging("DEBUG")
    m.log.warning("something odd")
    out = capsys.readouterr().out
    assert "WARNING" in out and "fraud" in out
    m.setup_logging("INFO")          # restore the notebook's default


# ------------------------------------------------------------------- recording
def test_start_run_assigns_ids_and_records(fresh_observer):
    run_id = fresh_observer.start_run("thread-1", "check CUST-1042")
    assert run_id.startswith("run-")
    first = fresh_observer.events[0]
    assert first["event"] == "run_start"
    assert first["thread_id"] == "thread-1"
    assert first["user_request"] == "check CUST-1042"


def test_events_carry_the_current_node(fresh_observer):
    fresh_observer.start_run("t", "r")
    fresh_observer.current_node = "fraud_analyst"
    entry = fresh_observer.record("tool_call", tool="fetch_customer_transactions")
    assert entry["node"] == "fraud_analyst"
    assert entry["tool"] == "fetch_customer_transactions"
    assert entry["elapsed_s"] >= 0


# ------------------------------------------------------------- node instrumenting
def test_observe_node_times_the_call_and_records_both_ends(fresh_observer):
    @m.observe_node("slow_node")
    def node(state):
        return {"ok": True}

    assert node({}) == {"ok": True}

    events = [e["event"] for e in fresh_observer.events]
    assert events == ["node_start", "node_end"]
    stats = fresh_observer.nodes["slow_node"]
    assert stats.calls == 1 and stats.errors == 0
    assert stats.seconds >= 0


def test_observe_node_records_the_failure_and_re_raises(fresh_observer, caplog):
    @m.observe_node("broken")
    def node(state):
        raise ValueError("tool exploded")

    with pytest.raises(ValueError, match="tool exploded"):
        node({})

    assert fresh_observer.nodes["broken"].errors == 1
    error = fresh_observer.errors()[0]
    assert error["error_type"] == "ValueError" and "exploded" in error["message"]
    # the node still ends cleanly, so a failure cannot lose its timing
    assert [e["event"] for e in fresh_observer.events] == ["node_start", "error", "node_end"]


def test_a_human_in_the_loop_pause_is_not_an_error(fresh_observer):
    """`interrupt()` suspends the graph by raising; counting that as a failure would
    report an error for every clean HITL run and bury the real ones."""
    from langgraph.errors import GraphInterrupt

    @m.observe_node("human_review")
    def node(state):
        raise GraphInterrupt(("paused for review",))

    with pytest.raises(GraphInterrupt):
        node({})

    assert fresh_observer.nodes["human_review"].errors == 0
    assert fresh_observer.errors() == []
    assert [e["event"] for e in fresh_observer.events] == [
        "node_start", "node_paused", "node_end"]


def test_is_control_flow_only_matches_the_pause_signal():
    from langgraph.errors import GraphInterrupt
    assert m.is_control_flow(GraphInterrupt(("x",))) is True
    assert m.is_control_flow(ValueError("real bug")) is False
    assert m.is_control_flow(RuntimeError("real bug")) is False


def test_error_messages_are_truncated(fresh_observer):
    """A node failing mid-report must not dump the whole report into the log."""
    @m.observe_node("noisy")
    def node(state):
        raise ValueError("x" * 5000)

    with pytest.raises(ValueError):
        node({})
    assert len(fresh_observer.errors()[0]["message"]) <= 500


def test_observe_node_restores_the_previous_node(fresh_observer):
    @m.observe_node("inner")
    def inner(state):
        return {}

    @m.observe_node("outer")
    def outer(state):
        inner(state)
        return {"current_during_outer": fresh_observer.current_node}

    assert outer({})["current_during_outer"] == "outer"
    assert fresh_observer.current_node is None


def test_decorator_keeps_the_function_identity():
    """@observe_node wraps with functools.wraps, so nodes keep their name and docstring."""
    assert m.triage.__name__ == "triage"
    assert "Classify the incoming signal" in (m.triage.__doc__ or "")
    assert m.comms_deliver.__name__ == "comms_deliver"


# ------------------------------------------------------------------ attribution
class _Response:
    def __init__(self, inp, out):
        message = type("M", (), {"usage_metadata": {"input_tokens": inp,
                                                    "output_tokens": out}})()
        self.generations = [[type("G", (), {"message": message})()]]


def test_tokens_are_attributed_to_the_node_that_spent_them(fresh_observer):
    fresh_observer.start_run("t", "r")

    @m.observe_node("fraud_analyst")
    def analyst(state):
        fresh_observer.on_llm_end(_Response(1000, 200))
        return {}

    @m.observe_node("compliance_officer")
    def officer(state):
        fresh_observer.on_llm_end(_Response(500, 300))
        return {}

    analyst({})
    officer({})

    assert fresh_observer.nodes["fraud_analyst"].input_tokens == 1000
    assert fresh_observer.nodes["compliance_officer"].output_tokens == 300
    assert fresh_observer.input_tokens == 1500 and fresh_observer.output_tokens == 500


def test_tokens_spent_outside_a_node_are_not_silently_dropped(fresh_observer):
    fresh_observer.on_llm_end(_Response(10, 5))
    assert fresh_observer.nodes["unattributed"].input_tokens == 10


def test_totals_match_the_sum_of_the_rows(fresh_observer, monkeypatch):
    monkeypatch.setattr(m, "MODEL_NAME", "claude-haiku-4-5")     # $1 / $5 per 1M
    fresh_observer.current_node = "a"
    fresh_observer.on_llm_end(_Response(2000, 400))
    fresh_observer.current_node = "b"
    fresh_observer.on_llm_end(_Response(1000, 100))

    per_node = sum(s.cost_usd("claude-haiku-4-5") for s in fresh_observer.nodes.values())
    assert fresh_observer.cost_usd == pytest.approx(per_node)
    assert fresh_observer.cost_usd == pytest.approx(3000 / 1e6 + 500 / 1e6 * 5)
    assert fresh_observer.calls == 2


# -------------------------------------------------------------------- reporting
def test_summary_lists_every_node_and_a_total(fresh_observer):
    fresh_observer.current_node = "fraud_analyst"
    fresh_observer.on_llm_end(_Response(100, 50))
    fresh_observer.stats("fraud_analyst").calls = 1

    summary = fresh_observer.summary()
    assert "fraud_analyst" in summary and "TOTAL" in summary
    assert "cost $" in summary


def test_to_json_round_trips_every_event(fresh_observer, tmp_path):
    fresh_observer.start_run("t-9", "check CUST-1001")
    fresh_observer.current_node = "intake"
    fresh_observer.record("node_end", seconds=0.2)
    path = tmp_path / "events.json"

    fresh_observer.to_json(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["model"] == m.MODEL_NAME
    assert [e["event"] for e in payload["events"]] == ["run_start", "node_end"]
    assert payload["events"][1]["node"] == "intake"


def test_timeline_is_chronological_and_bounded(fresh_observer):
    fresh_observer.start_run("t", "r")
    for i in range(30):
        fresh_observer.record("node_start", i=i)
    assert len(fresh_observer.timeline(limit=5).splitlines()) == 5


# --------------------------------------------------------------------- langsmith
def test_langsmith_stays_off_without_a_key(monkeypatch):
    monkeypatch.setattr(m, "get_secret_or_none", lambda name: None)
    for var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
        monkeypatch.delenv(var, raising=False)

    assert m.enable_langsmith() is False
    import os
    assert "LANGCHAIN_TRACING_V2" not in os.environ, "tracing must not be enabled silently"


def test_langsmith_turns_on_with_a_key(monkeypatch):
    monkeypatch.setattr(m, "get_secret_or_none", lambda name: "ls-key")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    assert m.enable_langsmith("my-project") is True
    import os
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-project"


def test_get_secret_or_none_does_not_raise(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_XYZ", raising=False)
    assert m.get_secret_or_none("DEFINITELY_NOT_SET_XYZ") is None
