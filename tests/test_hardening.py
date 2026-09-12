"""Production hardening that needs nothing outside Colab (plan §5.1, §5.3).

Retries on transient API faults, a cost circuit breaker, and a preflight check that fails
with an actionable message instead of halfway through Run all.
"""
import pytest

import fraud_multi_agent as m


# ---- retries --------------------------------------------------------------------------


def test_retry_returns_the_first_success_without_sleeping():
    calls = []

    def ok():
        calls.append(1)
        return "done"

    assert m.retry_call(ok, description="test") == "done"
    assert len(calls) == 1


def test_retry_recovers_from_a_transient_overload(monkeypatch):
    slept = []
    monkeypatch.setattr(m.time, "sleep", slept.append)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("529 overloaded_error: Anthropic is temporarily overloaded")
        return "recovered"

    assert m.retry_call(flaky, description="test") == "recovered"
    assert len(attempts) == 3
    assert len(slept) == 2, "each retry must back off before trying again"
    assert slept[1] > slept[0], "backoff must grow, not hammer a struggling API"


def test_retry_gives_up_after_the_budget_and_reraises(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)

    def always_rate_limited():
        raise RuntimeError("429 rate_limit_error")

    with pytest.raises(RuntimeError, match="429"):
        m.retry_call(always_rate_limited, description="test", attempts=3)


def test_retry_does_not_retry_a_permanent_error(monkeypatch):
    """A 400 will fail identically every time; retrying it only wastes the reviewer's time."""
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    attempts = []

    def bad_request():
        attempts.append(1)
        raise ValueError("400 invalid_request_error: temperature is not supported")

    with pytest.raises(ValueError):
        m.retry_call(bad_request, description="test")
    assert len(attempts) == 1


def test_transient_classification_covers_the_codes_that_actually_happen():
    for text in ("429 rate_limit_error", "529 overloaded_error", "500 internal server error",
                 "Connection reset by peer", "Read timed out"):
        assert m.is_transient(RuntimeError(text)), text
    for text in ("400 invalid_request_error", "401 authentication_error",
                 "404 not_found_error"):
        assert not m.is_transient(RuntimeError(text)), text


# ---- cost circuit breaker -------------------------------------------------------------


def test_budget_guard_passes_under_the_limit(fresh_observer, monkeypatch):
    monkeypatch.setattr(m, "MAX_USD_PER_NOTEBOOK", 1.0)
    fresh_observer.stats("triage").input_tokens = 1_000
    m.check_budget()          # nowhere near the limit — must not raise


def test_budget_guard_trips_before_spending_more(fresh_observer, monkeypatch):
    monkeypatch.setattr(m, "MAX_USD_PER_NOTEBOOK", 0.001)
    fresh_observer.stats("fraud_analyst").input_tokens = 10_000_000
    with pytest.raises(m.BudgetExceeded) as excinfo:
        m.check_budget()
    assert "MAX_USD_PER_NOTEBOOK" in str(excinfo.value)


def test_budget_guard_can_be_switched_off(fresh_observer, monkeypatch):
    monkeypatch.setattr(m, "MAX_USD_PER_NOTEBOOK", None)
    fresh_observer.stats("fraud_analyst").input_tokens = 10_000_000
    m.check_budget()


def test_the_default_budget_is_generous_enough_for_a_whole_run():
    """Run all must never trip the breaker: the demos cost cents, the cap is dollars."""
    assert m.MAX_USD_PER_NOTEBOOK is not None
    assert m.MAX_USD_PER_NOTEBOOK >= 1.0


# ---- preflight -------------------------------------------------------------------------


def test_preflight_reports_a_missing_key_without_raising(monkeypatch):
    def no_key(_name):
        raise RuntimeError("Secret 'ANTHROPIC_API_KEY' not found.")

    monkeypatch.setattr(m, "get_secret", no_key)
    ok, message = m.preflight()
    assert ok is False
    assert "ANTHROPIC_API_KEY" in message
    assert "Secrets" in message          # tells the reader how to fix it in Colab


def test_preflight_passes_with_a_key_present(monkeypatch):
    monkeypatch.setattr(m, "get_secret", lambda _name: "sk-ant-fake")
    ok, message = m.preflight()
    assert ok is True
    assert m.MODEL_NAME in message


def test_preflight_never_prints_the_key(monkeypatch):
    monkeypatch.setattr(m, "get_secret", lambda _name: "sk-ant-secret-value")
    _ok, message = m.preflight()
    assert "sk-ant-secret-value" not in message


# ---- PII in logs ------------------------------------------------------------------------


def test_customer_names_are_masked_in_the_analyst_trace(caplog):
    """A full name in a log line is a PII copy nobody asked for (plan §5.3)."""
    with caplog.at_level("INFO", logger=m.LOGGER_NAME):
        m.log.info("screening %s", m.mask_name("Viktor Baranov"))
    assert "Viktor B." in caplog.text
    assert "Baranov" not in caplog.text


# ---- the guards are actually in the agents' path ---------------------------------------


class _FlakyOnce:
    """An LLM that fails transiently on its first call and succeeds on the second."""

    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("529 overloaded_error")
        from langchain_core.messages import AIMessage
        return AIMessage(content="recovered from memory")

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, method=None):
        raise AssertionError("not needed for this test")


def test_a_node_survives_a_transient_api_fault(monkeypatch):
    """Decoration is not protection: the guard has to be on the path the agents use."""
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)
    flaky = _FlakyOnce()
    monkeypatch.setattr(m, "get_llm", lambda: flaky)

    out = m.followup_qa({"messages": [], "user_request": "what was the score?"})
    assert flaky.calls == 2
    assert "recovered" in out["final_output"]


def test_the_budget_breaker_stops_a_node_before_it_spends(monkeypatch, fresh_observer):
    monkeypatch.setattr(m, "MAX_USD_PER_NOTEBOOK", 0.0001)
    fresh_observer.stats("followup_qa").input_tokens = 10_000_000
    flaky = _FlakyOnce()
    monkeypatch.setattr(m, "get_llm", lambda: flaky)

    with pytest.raises(m.BudgetExceeded):
        m.followup_qa({"messages": [], "user_request": "again?"})
    assert flaky.calls == 0, "the breaker must trip before the paid call, not after"
