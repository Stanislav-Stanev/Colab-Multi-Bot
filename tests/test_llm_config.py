"""Guards on the request we send to Anthropic.

These lock in the fixes for two failure modes that only show up on a live call:
sampling parameters are rejected with a 400 on the Claude 5 models, and LangChain's
default structured-output method uses forced tool calling.
"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import fraud_multi_agent as m


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-no-request-is-made")
    m._llm_cache.clear()
    client = m.get_llm()
    yield client
    m._llm_cache.clear()


MESSAGES = [SystemMessage(content="sys"), HumanMessage(content="hi")]


def test_no_sampling_parameters_are_sent(llm):
    payload = llm._get_request_payload(MESSAGES)
    for removed in ("temperature", "top_p", "top_k"):
        assert removed not in payload, f"{removed} returns a 400 on the Claude 5 models"


def test_model_and_max_tokens(llm):
    payload = llm._get_request_payload(MESSAGES)
    assert payload["model"] == m.MODEL_NAME
    assert payload["max_tokens"] == 16000


def test_effort_is_only_sent_when_the_model_supports_it(llm):
    """Haiku 4.5 answers a request carrying `effort` with a 400, so it must be omitted."""
    payload = llm._get_request_payload(MESSAGES)
    if m.EFFORT is None:
        assert "effort" not in payload.get("output_config", {})
    else:
        assert payload["output_config"]["effort"] == m.EFFORT


def test_every_model_in_pricing_is_priced():
    assert m.MODEL_NAME in m.PRICING, "unknown model would report a $0.00 cost"
    for name, (price_in, price_out) in m.PRICING.items():
        assert price_in > 0 and price_out > price_in, name


def test_usage_tracker_accumulates_and_prices(monkeypatch):
    class FakeGeneration:
        def __init__(self, inp, out):
            self.message = type("M", (), {
                "usage_metadata": {"input_tokens": inp, "output_tokens": out}})()

    tracker = m.UsageTracker()
    tracker.on_llm_end(type("R", (), {"generations": [[FakeGeneration(1000, 200)],
                                                      [FakeGeneration(500, 100)]]})())
    assert tracker.calls == 2
    assert tracker.input_tokens == 1500 and tracker.output_tokens == 300

    monkeypatch.setattr(m, "MODEL_NAME", "claude-haiku-4-5")   # $1 / $5 per 1M
    assert tracker.cost_usd == pytest.approx(1500 / 1e6 * 1.0 + 300 / 1e6 * 5.0)
    assert "2 LLM calls" in tracker.report()


def test_usage_tracker_survives_a_response_without_usage():
    tracker = m.UsageTracker()
    tracker.on_llm_end(type("R", (), {"generations": [[object()]]})())
    assert tracker.calls == 0


def test_structured_output_uses_native_json_schema_not_forced_tools(llm):
    bound = m.structured(m.AnalystNarrative)
    inner = getattr(bound, "first", bound)
    payload = llm._get_request_payload(MESSAGES, **getattr(inner, "kwargs", {}))

    output_config = payload["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    if m.EFFORT is not None:
        assert output_config["effort"] == m.EFFORT      # effort survives the merge
    assert payload.get("tool_choice") is None           # never forced tool calling
    assert "temperature" not in payload


def test_tool_binding_sends_all_three_tools_unforced(llm):
    payload = llm._get_request_payload(MESSAGES, **llm.bind_tools(m.TOOLS).kwargs)
    assert {t["name"] for t in payload["tools"]} == {
        "fetch_customer_transactions", "calculate_risk_score", "check_sanctions_list"}
    assert payload.get("tool_choice") is None
    assert "temperature" not in payload


def test_llm_client_is_cached(llm):
    assert m.get_llm() is llm
