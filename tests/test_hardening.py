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


# ---- classification reads the exception, not the prose (review F3) ---------------------


class _ApiError(Exception):
    """Shaped like the SDK's APIStatusError: the status code is an attribute."""

    def __init__(self, status_code, message="api error"):
        super().__init__(message)
        self.status_code = status_code


def test_a_status_code_decides_before_any_text_matching():
    assert m.is_transient(_ApiError(429)) is True
    assert m.is_transient(_ApiError(529)) is True
    assert m.is_transient(_ApiError(503)) is True
    assert m.is_transient(_ApiError(400)) is False
    assert m.is_transient(_ApiError(401)) is False
    assert m.is_transient(_ApiError(404)) is False


def test_a_wrapped_api_error_is_still_classified_by_its_cause():
    """LangChain wraps provider errors; the status code survives on `__cause__`."""
    try:
        try:
            raise _ApiError(429, "rate limited")
        except _ApiError as cause:
            raise RuntimeError("Error calling the model") from cause
    except RuntimeError as wrapped:
        assert m.is_transient(wrapped) is True

    try:
        try:
            raise _ApiError(400, "temperature is not supported")
        except _ApiError as cause:
            raise RuntimeError("Error calling the model") from cause
    except RuntimeError as wrapped:
        assert m.is_transient(wrapped) is False


def test_a_number_inside_a_message_is_not_a_status_code():
    """The false positive substring matching produces (review F3).

    "1500.00" contains "500"; a declined payment is not a server error, and retrying it
    three times only makes the reviewer wait for the same answer.
    """
    assert m.is_transient(ValueError("payment of 1500.00 EUR was declined")) is False
    assert m.is_transient(ValueError("expected 4290 tokens, got 5040")) is False
    # a real code in the same prose still reads as transient
    assert m.is_transient(ValueError("HTTP 500 while scoring 1500.00 EUR")) is True


def test_retrying_does_not_swallow_a_keyboard_interrupt(monkeypatch):
    """`except BaseException` would catch Ctrl-C and retry through it (review F3)."""
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)

    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        m.retry_call(interrupted, description="test")


def test_only_one_layer_retries(monkeypatch):
    """Stacked retries multiply: 3 client attempts x 3 of ours is 9 waits, not 3.

    `retry_call` owns the policy — it can tell a 429 from a 400 and it records the
    attempt — so the SDK's own retry loop is switched off underneath it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-no-request-is-made")
    m._llm_cache.clear()
    try:
        assert m.get_llm().max_retries == 0
    finally:
        m._llm_cache.clear()


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


# ---- the tool loop cannot be taken down by what the model asks for ---------------------


class _AsksForAToolItDoesNotHave:
    """A model that calls a tool outside its kit, then settles down."""

    def __init__(self, tool_name):
        self.tool_name = tool_name
        self.rounds = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.rounds += 1
        if self.rounds == 1:
            return AIMessage(content="", tool_calls=[
                {"name": self.tool_name, "args": {"customer_name": "whoever"},
                 "id": "x1", "type": "tool_call"}])
        return AIMessage(content="Done.")


def test_an_unavailable_tool_is_refused_not_crashed(monkeypatch, fresh_observer):
    """A tool name the harness does not serve must not take the case down.

    `check_sanctions_list` was removed from the analyst's kit (review F10); a model that
    still asks for it — or for anything else — gets a structured refusal back and carries
    on, instead of ending the investigation with a KeyError.
    """
    fake = _AsksForAToolItDoesNotHave("check_sanctions_list")
    monkeypatch.setattr(m, "get_llm", lambda: fake)

    from langchain_core.messages import HumanMessage
    calls, evidence = m.run_tool_loop(m.TOOLS, [HumanMessage(content="investigate")])

    assert calls == [], "a tool outside the kit must not be reported as executed"
    assert any("not available" in line for line in evidence)
    assert fake.rounds == 2, "the loop carried on after refusing"
    refusals = [e for e in fresh_observer.events if e["event"] == "tool_unavailable"]
    assert len(refusals) == 1 and refusals[0]["tool"] == "check_sanctions_list"


def test_a_tool_loop_cut_short_by_its_round_limit_says_so(monkeypatch, fresh_observer):
    """Distinguish "the model was finished" from "the harness stopped it" (review F14)."""
    class _NeverStops:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage
            return AIMessage(content="", tool_calls=[
                {"name": "fetch_customer_transactions", "args": {"customer_id": "CUST-1001"},
                 "id": "loop", "type": "tool_call"}])

    monkeypatch.setattr(m, "get_llm", lambda: _NeverStops())
    from langchain_core.messages import HumanMessage
    m.run_tool_loop(m.TOOLS, [HumanMessage(content="go")], max_rounds=3)

    exhausted = [e for e in fresh_observer.events if e["event"] == "tool_loop_exhausted"]
    assert len(exhausted) == 1 and exhausted[0]["rounds"] == 3


# ---- how long a case has been waiting at the gate (review F12) --------------------------


def test_a_case_records_when_it_reached_the_human_gate(monkeypatch):
    """"Which P1 cases have been waiting more than a day" must be answerable.

    The timestamp is written by the node *before* the gate: `human_review` suspends by
    raising, so nothing it returns on the first pass is ever checkpointed.
    """
    class _CommsFake:
        def with_structured_output(self, schema, method=None):
            return type("S", (), {"invoke": lambda _self, _msgs: m.CustomerMessage(
                channel="push", language="en", subject="s", body="b")})()

    monkeypatch.setattr(m, "get_llm", lambda: _CommsFake())
    out = m.comms_draft({"customer_id": "CUST-1042", "case_id": "CASE-W",
                         "recommended_action": "BLOCK", "signal_type": "FRAUD_ALERT"})

    assert out["awaiting_review_since"]
    from datetime import datetime
    stamped = datetime.fromisoformat(out["awaiting_review_since"])
    assert stamped.tzinfo is not None, "a naive timestamp cannot be compared across a restart"


def test_waiting_time_is_reported_from_the_checkpointed_stamp():
    from datetime import datetime, timedelta, timezone
    two_days_ago = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    assert m.hours_awaiting_review({"awaiting_review_since": two_days_ago}) == pytest.approx(
        48, abs=0.1)
    assert m.hours_awaiting_review({}) is None
    assert m.hours_awaiting_review({"awaiting_review_since": "not a timestamp"}) is None


# ---- a refusal is diagnosed, not left as a traceback (review F14) ------------------------


def test_a_model_refusal_is_explained_before_it_is_re_raised(monkeypatch, caplog):
    """`stop_reason: "refusal"` through structured output is otherwise opaque."""
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)

    class _Refusing:
        def invoke(self, *_a, **_k):
            raise ValueError("Invalid response: stop_reason='refusal' (category: cyber)")

    guarded = m._Guarded(_Refusing(), "ComplianceReport")
    with caplog.at_level("ERROR", logger=m.LOGGER_NAME):
        with pytest.raises(ValueError):
            guarded.invoke([])

    assert "refus" in caplog.text.lower()
    assert "PROMPT_VERSION" in caplog.text or "prompt" in caplog.text.lower()


def test_an_ordinary_failure_gets_no_refusal_diagnosis(monkeypatch, caplog):
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)

    class _Broken:
        def invoke(self, *_a, **_k):
            raise ValueError("400 invalid_request_error: max_tokens is too large")

    with caplog.at_level("ERROR", logger=m.LOGGER_NAME):
        with pytest.raises(ValueError):
            m._Guarded(_Broken(), "x").invoke([])
    assert "refus" not in caplog.text.lower()
